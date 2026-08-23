import { readFileSync } from "node:fs";
import { z } from "zod";
import { sha256 } from "../src/hash.js";
import { loadMacModelRegistry, resolveMacProfile } from "../src/models-mac.js";
import { LAB_ROOT, atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";
import {
  cacheLocation,
  ensureServer,
  foundry,
  loadModelTimed,
  openAiBaseUrl,
  resolveVariantId,
  restartServer,
  setCacheLocation,
  unloadModel,
} from "./foundry-runtime.js";

// Mac-plane model comparison over the frozen episode set in
// config/mac-eval-episodes.json. One profile per invocation:
//   npm run mac:eval -- --profile qwen3-4b-foundry
// Results are development evidence (docs/MAC_PROFILE.md), never campaign rows.

const decisionSchema = z.object({
  kind: z.literal("decision"),
  incidentClass: z.enum(["log_full", "blocking", "deadlock", "login_failure", "backup_failure", "io_corruption", "no_incident", "unknown"]),
  severity: z.enum(["low", "medium", "high"]),
  action: z.enum(["open_work_item", "escalate", "no_action"]),
  rationale: z.string(),
});

const toolRequestSchema = z.object({
  kind: z.literal("tool_request"),
  tool: z.string(),
  arguments: z.record(z.string(), z.unknown()),
  callId: z.string().optional(),
});

const replySchema = z.discriminatedUnion("kind", [decisionSchema, toolRequestSchema]);
type Reply = z.infer<typeof replySchema>;

const LEGAL_REPAIRS = new Set(["think_strip", "channel_strip", "fence_strip", "leading_text_strip", "trailing_text_strip"]);

interface ParsedReply {
  reply?: Reply;
  repairs: string[];
  contractLevel: "first_pass" | "legal_repair" | "illegal_repair" | "invalid";
  parseError?: string;
}

function parseModelReply(raw: string): ParsedReply {
  const repairs: string[] = [];
  let text = raw.trim();

  const thinkStripped = text.replace(/^<think>[\s\S]*?<\/think>\s*/u, "");
  if (thinkStripped !== text) { text = thinkStripped.trim(); repairs.push("think_strip"); }

  // Muse visible-variant channels: reasoning arrives as ` to=self<|message|>…`
  // and the answer as `<|start|>assistant to=user<|message|>…`; keep only the
  // last user-directed message. (Also accept the `<|channel|>final` framing.)
  if (text.includes("to=user<|message|>")) {
    const finals = [...text.matchAll(/to=user<\|message\|>([\s\S]*?)(?:<\|eom\|>|<\|end\|>|<\|return\|>|<\|start\|>|$)/gu)];
    const last = finals.at(-1)?.[1];
    if (last !== undefined) { text = last.trim(); repairs.push("channel_strip"); }
  } else if (text.includes("<|channel|>")) {
    const finals = [...text.matchAll(/<\|channel\|>final<\|message\|>([\s\S]*?)(?:<\|eom\|>|<\|end\|>|<\|return\|>|$)/gu)];
    const last = finals.at(-1)?.[1];
    if (last !== undefined) { text = last.trim(); repairs.push("channel_strip"); }
  }

  const fenceMatch = /^```(?:json)?\s*([\s\S]*?)```\s*$/u.exec(text);
  if (fenceMatch) { text = (fenceMatch[1] ?? "").trim(); repairs.push("fence_strip"); }

  const firstBrace = text.indexOf("{");
  if (firstBrace > 0) { text = text.slice(firstBrace); repairs.push("leading_text_strip"); }

  const balanced = extractBalancedObject(text, 0);
  if (balanced && balanced.length < text.trim().length) {
    text = balanced;
    repairs.push("trailing_text_strip");
  }

  const attempt = tryValidate(text);
  if (attempt.reply) return { reply: attempt.reply, repairs, contractLevel: levelFor(repairs) };

  // Last resort: scan for any balanced object that validates (contract failure,
  // but the decision is still scoreable — reported separately).
  for (let index = raw.indexOf("{"); index >= 0; index = raw.indexOf("{", index + 1)) {
    const candidate = extractBalancedObject(raw, index);
    if (!candidate) continue;
    const rescued = tryValidate(candidate);
    if (rescued.reply) {
      repairs.push("balanced_extract");
      return { reply: rescued.reply, repairs, contractLevel: "illegal_repair" };
    }
  }
  return { repairs, contractLevel: "invalid", ...(attempt.error === undefined ? {} : { parseError: attempt.error }) };
}

function tryValidate(text: string): { reply?: Reply; error?: string } {
  try {
    return { reply: replySchema.parse(JSON.parse(text)) };
  } catch (error) {
    return { error: error instanceof Error ? error.message.slice(0, 300) : String(error) };
  }
}

function extractBalancedObject(text: string, start: number): string | undefined {
  if (text[start] !== "{") return undefined;
  let depth = 0;
  let inString = false;
  for (let index = start; index < text.length; index += 1) {
    const character = text[index];
    if (inString) {
      if (character === "\\") index += 1;
      else if (character === '"') inString = false;
    } else if (character === '"') inString = true;
    else if (character === "{") depth += 1;
    else if (character === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(start, index + 1);
    }
  }
  return undefined;
}

function levelFor(repairs: string[]): ParsedReply["contractLevel"] {
  if (repairs.length === 0) return "first_pass";
  return repairs.every((repair) => LEGAL_REPAIRS.has(repair)) ? "legal_repair" : "illegal_repair";
}

const episodeFileSchema = z.object({
  schemaVersion: z.literal(1),
  name: z.string(),
  decodeConfig: z.object({ temperature: z.number(), maxTokens: z.number().int(), maxModelTurns: z.number().int() }),
  contract: z.string().min(200),
  episodes: z.array(z.object({
    episodeId: z.string(),
    packet: z.object({
      source: z.string(),
      evidence: z.string(),
      requiredDiagnostic: z.string().nullable(),
    }),
    toolResult: z.object({ tool: z.string(), result: z.record(z.string(), z.unknown()) }).optional(),
    expectedTool: z.object({ tool: z.string(), arguments: z.record(z.string(), z.unknown()) }).optional(),
    expected: z.object({ incidentClass: z.string(), severity: z.string(), action: z.string() }),
  })),
});

let endpointBase = openAiBaseUrl;

async function chat(variantId: string, messages: Array<{ role: string; content: string }>, maxTokens: number, temperature: number) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 600_000);
  const startedAt = Date.now();
  try {
    const response = await fetch(`${endpointBase}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: variantId, messages, max_tokens: maxTokens, temperature }),
      signal: controller.signal,
    });
    const latencyMs = Date.now() - startedAt;
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
    const body = await response.json() as {
      choices?: Array<{ message?: { content?: string } }>;
      usage?: { completion_tokens?: number; prompt_tokens?: number };
    };
    return {
      raw: body.choices?.[0]?.message?.content ?? "",
      latencyMs,
      completionTokens: body.usage?.completion_tokens ?? 0,
      promptTokens: body.usage?.prompt_tokens ?? 0,
    };
  } finally {
    clearTimeout(timer);
  }
}

const profileKey = valueAfter("--profile");
if (!profileKey) throw new Error("Pass --profile <mac profile key>");
const registry = loadMacModelRegistry();
const profile = resolveMacProfile(profileKey, registry);
const episodeFile = episodeFileSchema.parse(
  JSON.parse(readFileSync(valueAfter("--episodes") ?? `${LAB_ROOT}/config/mac-eval-episodes.json`, "utf8")),
);
const runDirectory = resolveRunDirectory();
const { temperature, maxModelTurns } = episodeFile.decodeConfig;
const maxTokens = profile.maxTokensOverride ?? episodeFile.decodeConfig.maxTokens;
const promptPrefix = profile.promptPrefix ?? "";

const externalEndpoint = profile.baseUrl !== undefined;
const defaultCache = externalEndpoint ? "" : cacheLocation();
let cacheSwitched = false;
if (!externalEndpoint && profile.cacheDir && profile.cacheDir !== defaultCache) {
  console.log(`Switching Foundry cache to ${profile.cacheDir}`);
  setCacheLocation(profile.cacheDir);
  cacheSwitched = true;
}

try {
  let loadTimeMs = 0;
  let variantId: string;
  if (externalEndpoint) {
    // e.g. mlx_lm.server for models the Foundry catalog lacks. The server is
    // managed outside this script; verify it answers before spending episodes.
    endpointBase = profile.baseUrl!.replace(/\/$/, "");
    variantId = profile.servedModelId ?? profile.foundryAlias;
    const probe = await fetch(`${endpointBase}/models`).catch(() => null);
    if (!probe?.ok) throw new Error(`No OpenAI-compatible server at ${endpointBase}; start it first.`);
    console.log(`Using external endpoint ${endpointBase} for ${variantId}; running ${episodeFile.episodes.length} episodes`);
  } else {
    if (cacheSwitched) await restartServer();
    else await ensureServer();

    // One resident chat model at a time, mirroring the lab's residency rule.
    for (const [key, other] of Object.entries(registry.profiles)) {
      if (key !== profileKey && !other.cacheDir && !other.baseUrl) unloadModel(other.foundryAlias);
    }
    if (!profile.cacheDir) foundry(["model", "download", profile.foundryAlias]);
    loadTimeMs = loadModelTimed(profile.foundryAlias);
    variantId = resolveVariantId(profile.foundryAlias, profile.foundryVariantId);
    console.log(`Loaded ${variantId} in ${(loadTimeMs / 1000).toFixed(1)}s; running ${episodeFile.episodes.length} episodes`);
  }

  const rows: Array<Record<string, unknown>> = [];
  for (const episode of episodeFile.episodes) {
    const messages: Array<{ role: string; content: string }> = [{
      role: "user",
      content: `${promptPrefix}${episodeFile.contract}\n\nEvidence packet:\n${JSON.stringify(episode.packet, null, 2)}`,
    }];
    const turns: Array<Record<string, unknown>> = [];
    let decision: z.infer<typeof decisionSchema> | undefined;
    let decisionLevel: ParsedReply["contractLevel"] | "missing" = "missing";
    let firstReply: Reply | undefined;
    let snapshotMisses = 0;
    let toolRequests: Array<z.infer<typeof toolRequestSchema>> = [];

    for (let turn = 0; turn < maxModelTurns; turn += 1) {
      const completion = await chat(variantId, messages, maxTokens, temperature);
      const parsed = parseModelReply(completion.raw);
      turns.push({
        latencyMs: completion.latencyMs,
        completionTokens: completion.completionTokens,
        promptTokens: completion.promptTokens,
        repairs: parsed.repairs,
        contractLevel: parsed.contractLevel,
        rawSha256: sha256(completion.raw),
        raw: completion.raw.length > 4_000 ? `${completion.raw.slice(0, 4_000)}…[truncated]` : completion.raw,
        ...(parsed.parseError === undefined ? {} : { parseError: parsed.parseError }),
      });
      if (!parsed.reply) break;
      firstReply = firstReply ?? parsed.reply;
      if (parsed.reply.kind === "decision") {
        decision = parsed.reply;
        decisionLevel = parsed.contractLevel;
        break;
      }
      toolRequests.push(parsed.reply);
      messages.push({ role: "assistant", content: completion.raw });
      const frozen = episode.toolResult && episode.toolResult.tool === parsed.reply.tool
        ? episode.toolResult.result
        : { status: "no_data", reason: "no snapshot for arguments" };
      if (!episode.toolResult || episode.toolResult.tool !== parsed.reply.tool) snapshotMisses += 1;
      // promptPrefix rides every user turn: it emulates a per-request chat
      // template kwarg (e.g. Qwen3 /no_think), which must not lapse after a
      // tool result or the model re-enters its reasoning channel.
      messages.push({
        role: "user",
        content: `${promptPrefix}${JSON.stringify({ kind: "tool_result", callId: parsed.reply.callId ?? null, tool: parsed.reply.tool, result: frozen })}`,
      });
    }

    const expected = episode.expected;
    const toolExpected = episode.expectedTool;
    const requiredToolFirst = toolExpected
      ? firstReply?.kind === "tool_request" && firstReply.tool === toolExpected.tool
      : undefined;
    const toolArgsCorrect = toolExpected && firstReply?.kind === "tool_request"
      ? Object.entries(toolExpected.arguments).every(([argKey, argValue]) =>
          String((firstReply as z.infer<typeof toolRequestSchema>).arguments[argKey] ?? "").toLowerCase() === String(argValue).toLowerCase())
      : undefined;
    rows.push({
      episodeId: episode.episodeId,
      turns,
      decision: decision ?? null,
      decisionContractLevel: decisionLevel,
      scores: {
        classCorrect: decision?.incidentClass === expected.incidentClass,
        severityCorrect: decision?.severity === expected.severity,
        actionCorrect: decision?.action === expected.action,
        ...(requiredToolFirst === undefined ? {} : { requiredToolFirst }),
        ...(toolArgsCorrect === undefined ? {} : { toolArgsCorrect }),
        spuriousToolRequest: !toolExpected && toolRequests.length > 0,
        snapshotMisses,
      },
    });
    const score = rows.at(-1)!.scores as Record<string, unknown>;
    console.log(`${episode.episodeId}: class=${String(score.classCorrect)} severity=${String(score.severityCorrect)} action=${String(score.actionCorrect)} level=${decisionLevel}`);
  }

  const scored = rows.map((row) => row.scores as Record<string, unknown>);
  const decisions = rows.filter((row) => row.decision !== null);
  const allTurns = rows.flatMap((row) => row.turns as Array<Record<string, unknown>>);
  const totalCompletionTokens = allTurns.reduce((sum, turn) => sum + Number(turn.completionTokens), 0);
  const totalLatencyMs = allTurns.reduce((sum, turn) => sum + Number(turn.latencyMs), 0);
  const rate = (predicate: (score: Record<string, unknown>) => boolean, pool = scored) =>
    pool.length === 0 ? null : Number((pool.filter(predicate).length / pool.length).toFixed(3));
  const toolScored = scored.filter((score) => "requiredToolFirst" in score);
  const summary = {
    schemaVersion: 1,
    evaluatedAt: new Date().toISOString(),
    episodeSet: { name: episodeFile.name, sha256: sha256(JSON.stringify(episodeFile)) },
    profileKey,
    variantId,
    promptPrefix,
    decode: { temperature, maxTokens, maxTokensOverridden: profile.maxTokensOverride !== undefined },
    executionProvider: profile.executionProvider,
    fileSizeMb: profile.fileSizeMb,
    loadTimeMs,
    episodes: rows.length,
    decisionsProduced: decisions.length,
    metrics: {
      classAccuracy: rate((score) => score.classCorrect === true),
      severityAccuracy: rate((score) => score.severityCorrect === true),
      actionAccuracy: rate((score) => score.actionCorrect === true),
      tripleAccuracy: rate((score) => score.classCorrect === true && score.severityCorrect === true && score.actionCorrect === true),
      firstPassContractRate: rate((turn) => turn.contractLevel === "first_pass", allTurns),
      legalContractRate: rate((turn) => turn.contractLevel === "first_pass" || turn.contractLevel === "legal_repair", allTurns),
      requiredToolFirstRate: rate((score) => score.requiredToolFirst === true, toolScored),
      toolArgsCorrectRate: rate((score) => score.toolArgsCorrect === true, toolScored),
      spuriousToolRate: rate((score) => score.spuriousToolRequest === true),
      meanCompletionTokensPerEpisode: Number((totalCompletionTokens / rows.length).toFixed(1)),
      meanEpisodeLatencyMs: Number((totalLatencyMs / rows.length).toFixed(0)),
      decodeTokensPerSecond: totalLatencyMs > 0 ? Number((totalCompletionTokens / (totalLatencyMs / 1_000)).toFixed(1)) : null,
    },
  };

  await atomicWrite(`${runDirectory}/tables/mac-eval/${profileKey}.jsonl`, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
  await atomicWrite(`${runDirectory}/metrics/mac-eval-${profileKey}.json`, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(JSON.stringify(summary.metrics, null, 2));
  console.log(`Rows: tables/mac-eval/${profileKey}.jsonl; summary: metrics/mac-eval-${profileKey}.json`);

  if (!externalEndpoint) unloadModel(profile.foundryAlias);
} finally {
  if (cacheSwitched) {
    setCacheLocation(defaultCache);
    await restartServer();
    console.log(`Restored Foundry cache to ${defaultCache}`);
  }
}
