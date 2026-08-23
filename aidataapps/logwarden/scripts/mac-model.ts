import { sha256 } from "../src/hash.js";
import {
  loadMacModelRegistry,
  resolveMacEmbeddingProfile,
  resolveMacProfile,
} from "../src/models-mac.js";
import { atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";
import {
  ensureServer,
  foundry,
  openAiBaseUrl,
  resolveVariantId,
  serverStatus,
  unloadModel,
} from "./foundry-runtime.js";

// Drives Azure Foundry Local as the mac-profile serving plane (docs/MAC_PROFILE.md).
// One Foundry endpoint serves chat and embeddings; the lab pins it to CHAT_PORT
// so CHAT_BASE_URL and EMBEDDING_BASE_URL in .env agree with the daemon.
//
//   npm run mac:model -- status
//   npm run mac:model -- up --profile qwen3-4b-foundry
//   npm run mac:model -- up --profile qwen3-embedding-0.6b-foundry
//   npm run mac:model -- down

const command = process.argv[2] ?? "status";

interface CanaryResult {
  kind: "chat" | "embedding";
  variantId: string;
  latencyMs: number;
  ok: boolean;
  detail: Record<string, unknown>;
}

async function chatCanary(variantId: string): Promise<CanaryResult> {
  const startedAt = Date.now();
  const response = await fetch(`${openAiBaseUrl}/chat/completions`, {
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
  const response = await fetch(`${openAiBaseUrl}/embeddings`, {
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
    server: { ...status, baseUrl: openAiBaseUrl },
    profileKey,
    profile,
    canary,
  };
  console.log(JSON.stringify({ profileKey, variantId, baseUrl: openAiBaseUrl, canary }, null, 2));
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
    unloadModel(alias);
  }
  foundry(["server", "stop"], 120_000);
  console.log("Foundry daemon stopped.");
} else {
  console.error(`Unknown command ${JSON.stringify(command)}; use status | up | down.`);
  process.exitCode = 2;
}
