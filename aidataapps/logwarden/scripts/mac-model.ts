import { execFileSync, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { sha256 } from "../src/hash.js";
import {
  loadMacModelRegistry,
  resolveMacEmbeddingProfile,
  resolveMacProfile,
} from "../src/models-mac.js";
import { LAB_ROOT, atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";

// Drives Azure Foundry Local as the mac-profile serving plane (docs/MAC_PROFILE.md).
// One Foundry endpoint serves chat and embeddings; the lab pins it to CHAT_PORT
// so CHAT_BASE_URL and EMBEDDING_BASE_URL in .env agree with the daemon.
//
//   npm run mac:model -- status
//   npm run mac:model -- up --profile qwen3-4b-foundry
//   npm run mac:model -- up --profile qwen3-embedding-0.6b-foundry
//   npm run mac:model -- down

const port = Number(process.env.CHAT_PORT ?? 8010);
const baseUrl = `http://127.0.0.1:${port}/v1`;
const command = process.argv[2] ?? "status";

interface ServerStatus {
  running: boolean;
  state: string;
  pid?: number;
  webUrls?: string[];
}

function foundry(args: string[], timeout = 600_000): string {
  return execFileSync("foundry", args, { encoding: "utf8", timeout });
}

function serverStatus(): ServerStatus {
  try {
    return JSON.parse(foundry(["server", "status", "-o", "json"], 30_000)) as ServerStatus;
  } catch {
    return { running: false, state: "unknown" };
  }
}

async function ensureServer(): Promise<ServerStatus> {
  let status = serverStatus();
  const onPinnedPort = (candidate: ServerStatus) =>
    (candidate.webUrls ?? []).some((url) => url.endsWith(`:${port}`));
  if (status.running && onPinnedPort(status)) return status;
  if (status.running) {
    console.log(`Foundry daemon is on ${status.webUrls?.join(", ")}; restarting on :${port}`);
    spawn("foundry", ["server", "restart", "-p", String(port)], { detached: true, stdio: "ignore" }).unref();
  } else {
    console.log(`Starting Foundry daemon on :${port}`);
    spawn("foundry", ["server", "start", "-p", String(port)], { detached: true, stdio: "ignore" }).unref();
  }
  for (let attempt = 0; attempt < 60; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 2_000));
    status = serverStatus();
    if (status.running && status.state === "ready" && onPinnedPort(status)) return status;
  }
  throw new Error(`Foundry daemon did not become ready on :${port}`);
}

function resolveVariantId(alias: string, pinned: string): string {
  const info = JSON.parse(foundry(["model", "info", alias, "-o", "json"], 60_000)) as {
    model: { id: string; cached: boolean };
  };
  if (pinned && info.model.id !== pinned) {
    console.warn(`Pinned variant ${pinned} differs from catalog default ${info.model.id}; using the pin.`);
    return pinned;
  }
  return pinned || info.model.id;
}

interface CanaryResult {
  kind: "chat" | "embedding";
  variantId: string;
  latencyMs: number;
  ok: boolean;
  detail: Record<string, unknown>;
}

async function chatCanary(variantId: string): Promise<CanaryResult> {
  const startedAt = Date.now();
  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: variantId,
      messages: [{
        role: "user",
        content:
          "/no_think You are an incident-triage agent. Answer with exactly one JSON object, no prose, shaped " +
          '{"kind":"decision","incidentClass":"<one of: log_full, blocking, deadlock, login_failure>",' +
          '"severity":"<low|medium|high>","rationale":"<max 20 words>"}. ' +
          "Evidence: error 9002, severity 17: The transaction log for database LW_orders is full due to ACTIVE_TRANSACTION.",
      }],
      max_tokens: 200,
      temperature: 0,
    }),
  });
  const latencyMs = Date.now() - startedAt;
  const body = await response.json() as {
    choices?: Array<{ message?: { content?: string } }>;
    usage?: { completion_tokens?: number };
  };
  const raw = body.choices?.[0]?.message?.content ?? "";
  let text = raw.trim();
  let repairKind = "none";
  const thinkStripped = text.replace(/^<think>[\s\S]*?<\/think>\s*/u, "");
  if (thinkStripped !== text) { text = thinkStripped; repairKind = "think_strip"; }
  const fenceMatch = /^```(?:json)?\s*([\s\S]*?)```$/u.exec(text);
  if (fenceMatch) { text = (fenceMatch[1] ?? "").trim(); repairKind = `${repairKind}+fence_strip`; }
  let decision: Record<string, unknown> | undefined;
  try { decision = JSON.parse(text) as Record<string, unknown>; } catch { decision = undefined; }
  const completionTokens = body.usage?.completion_tokens ?? 0;
  return {
    kind: "chat",
    variantId,
    latencyMs,
    ok: response.ok && decision?.kind === "decision" && decision?.incidentClass === "log_full",
    detail: {
      httpStatus: response.status,
      repairKind,
      decision,
      completionTokens,
      tokensPerSecond: latencyMs > 0 ? Number((completionTokens / (latencyMs / 1_000)).toFixed(1)) : null,
      rawSha256: sha256(raw),
    },
  };
}

async function embeddingCanary(variantId: string, expectedDimensions: number): Promise<CanaryResult> {
  const startedAt = Date.now();
  const response = await fetch(`${baseUrl}/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: variantId,
      input: ["SQL Server log space warning 9002 in database LW_orders"],
    }),
  });
  const latencyMs = Date.now() - startedAt;
  const body = await response.json() as { data?: Array<{ embedding?: number[] }> };
  const dimensions = body.data?.[0]?.embedding?.length ?? 0;
  return {
    kind: "embedding",
    variantId,
    latencyMs,
    ok: response.ok && dimensions === expectedDimensions,
    detail: { httpStatus: response.status, dimensions, expectedDimensions },
  };
}

async function writeEvidence(profileKey: string, payload: Record<string, unknown>): Promise<void> {
  let runDirectory: string;
  try {
    runDirectory = resolveRunDirectory();
  } catch {
    console.log("No run selected; evidence not persisted (run npm run run:init first to record it).");
    return;
  }
  const path = `${runDirectory}/environment/mac-serving-${profileKey}.json`;
  await atomicWrite(path, `${JSON.stringify(payload, null, 2)}\n`);
  console.log(`Evidence written: ${path}`);
}

const registry = loadMacModelRegistry();

if (command === "status") {
  console.log(JSON.stringify(serverStatus(), null, 2));
  console.log(foundry(["cache", "list"], 60_000));
} else if (command === "up") {
  const profileKey = valueAfter("--profile") ?? registry.defaultProfile;
  const isEmbedding = profileKey in registry.embeddingProfiles;
  const profile = isEmbedding
    ? resolveMacEmbeddingProfile(profileKey, registry)
    : resolveMacProfile(profileKey, registry);
  const status = await ensureServer();
  foundry(["model", "download", profile.foundryAlias]);
  foundry(["model", "load", profile.foundryAlias]);
  const variantId = resolveVariantId(profile.foundryAlias, profile.foundryVariantId);
  const canary = isEmbedding
    ? await embeddingCanary(variantId, (profile as { expectedDimensions: number }).expectedDimensions)
    : await chatCanary(variantId);
  const evidence = {
    schemaVersion: 1,
    recordedAt: new Date().toISOString(),
    platform: registry.platform,
    foundryVersion: foundry(["--version"], 30_000).trim(),
    server: { ...status, baseUrl },
    profileKey,
    profile,
    canary,
  };
  console.log(JSON.stringify({ profileKey, variantId, baseUrl, canary }, null, 2));
  await writeEvidence(profileKey, evidence);
  if (!canary.ok) {
    console.error("Canary failed; the serving plane is not usable for this profile.");
    process.exitCode = 2;
  }
} else if (command === "down") {
  for (const alias of new Set([
    ...Object.values(registry.profiles).map((profile) => profile.foundryAlias),
    ...Object.values(registry.embeddingProfiles).map((profile) => profile.foundryAlias),
  ])) {
    try { foundry(["model", "unload", alias], 120_000); } catch { /* not loaded */ }
  }
  foundry(["server", "stop"], 120_000);
  console.log("Foundry daemon stopped.");
} else {
  console.error(`Unknown command ${JSON.stringify(command)}; use status | up | down.`);
  process.exitCode = 2;
}

// Keep LAB_ROOT referenced for tooling parity with sibling scripts.
void existsSync(LAB_ROOT);
