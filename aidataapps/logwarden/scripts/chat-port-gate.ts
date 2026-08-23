import { execFile as execFileCallback } from "node:child_process";
import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import { promisify } from "node:util";
import sql from "mssql";
import { buildInitialAgentMessages } from "../src/agent-loop.js";
import { summarizeChatMetrics, validateChatMetricDelta } from "../src/chat-metrics.js";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { callOpenAiCompatibleModel, type GatewayCallResult } from "../src/model-gateway.js";
import { resolveModelProfile } from "../src/models.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal } from "../src/telemetry.js";

const execFile = promisify(execFileCallback);
const profileKey = argument("--profile") ?? process.env.MODEL_PROFILE ?? "qwen-smoke";
const maxTokens = Number(argument("--max-tokens") ?? "900");
if (!Number.isSafeInteger(maxTokens) || maxTokens < 256 || maxTokens > 4_096) throw new Error("--max-tokens must be an integer from 256 through 4096");
const profile = resolveModelProfile(profileKey);
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const gateId = randomUUID();
const startedAtUtc = new Date().toISOString();
const rawDirectory = `${runDirectory}/raw/chat-port-gate/${profileKey}/${gateId}`;
const receiptPath = `${runDirectory}/metrics/chat-port-gate-${profileKey}.json`;
const containerName = process.env.CHAT_CONTAINER_NAME ?? `${process.env.COMPOSE_PROJECT_NAME ?? "aidataapps-logwarden"}-chat`;
const created = await createComponentTelemetryJournal(runDirectory, run.runId, `chat-port-gate-${profileKey}`);
const gateSpan = await created.journal.startSpan("chat.port_gate", {}, {
  gateId,
  profileKey,
  modelId: profile.modelId,
  revision: profile.revision,
  image: profile.vllmImage,
  maxTokens,
});

try {
  await requireFreezeForTarget();
  await waitReady();
  const service = await inspectService();
  const health = await retainEndpoint("health", new URL("/health", config.inference.chatBaseUrl));
  if (health.status < 200 || health.status >= 300) throw new Error(`Chat health returned HTTP ${health.status}`);
  const models = await retainEndpoint("models", new URL("/v1/models", config.inference.chatBaseUrl));
  const modelEnvelope = JSON.parse(models.body) as { data?: Array<{ id?: string }> };
  if (!modelEnvelope.data?.some((row) => row.id === profile.modelId))
    throw new Error(`Served model list does not contain ${profile.modelId}`);

  const gpuBefore = await retainGpu("before");
  const metricsBefore = await retainMetrics("before");
  const decode = {
    temperature: 0,
    top_p: 1,
    top_k: 0,
    min_p: 0,
    repetition_penalty: 1,
    presence_penalty: 0,
    frequency_penalty: 0,
    seed: 0,
    max_tokens: maxTokens,
    n: 1,
    stop: [],
    chat_template_kwargs: profile.chatTemplateKwargs,
  };
  const packets = canaryPackets();
  const calls: GatewayCallResult[] = [];
  for (const packet of packets) calls.push(await callCanary(packet, decode));
  const repeatedPacket = packets[0]!;
  calls.push(await callCanary(repeatedPacket, decode));
  calls.push(...await Promise.all(Array.from({ length: 4 }, () => callCanary(repeatedPacket, decode))));
  for (const result of calls) validateCall(result);
  const gpuDuring = await retainGpu("during");
  const metricsAfter = await retainMetrics("after");
  const metricDelta = validateChatMetricDelta(metricsBefore.summary, metricsAfter.summary, calls.length);
  if (gpuDuring.processes.length === 0 || gpuDuring.memoryUsedMiB < 1_000)
    throw new Error("GPU snapshot did not prove a resident chat model process");

  const logs = await command("docker", ["logs", "--timestamps", "--tail", "2000", containerName]);
  await atomicWrite(`${rawDirectory}/service.log`, logs, 0o600);
  if (/CUDA out of memory|Engine core initialization failed|Traceback \(most recent call last\)/i.test(logs))
    throw new Error("Chat service logs contain a fatal runtime signature");
  const samplingLines = logs.split(/\r?\n/).filter((line) => /SamplingParams\(/.test(line));
  const effectiveSamplingParamsLine = samplingLines.at(-1) ?? null;
  if (effectiveSamplingParamsLine === null) throw new Error("vLLM did not log effective SamplingParams");
  for (const expected of ["n=1", "presence_penalty=0", "frequency_penalty=0", "repetition_penalty=1", "temperature=0", "top_p=1", "top_k=0", "min_p=0", "seed=0", `max_tokens=${maxTokens}`]) {
    if (!effectiveSamplingParamsLine.includes(expected)) throw new Error(`Effective SamplingParams does not contain ${expected}`);
  }

  const contentHashes = calls.slice(packets.length - 1).map((result) => sha256(result.attempts.at(-1)!.content!));
  const exactMatches = contentHashes.filter((hash) => hash === contentHashes[0]).length;
  const exactMatchRate = exactMatches / contentHashes.length;
  await created.journal.record("point", "state.chat_port_gate_passed", {
    traceId: gateSpan.traceId,
    parentSpanId: gateSpan.spanId,
  }, { gateId, profileKey, metricDelta, exactMatchRate }, { status: "success" });
  await created.journal.endSpan(gateSpan, "success", { gateId, profileKey, calls: calls.length, metricDelta, exactMatchRate });
  await created.journal.flush();
  const ingestion = await ingestJournal();
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    gateId,
    profileKey,
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    model: { modelId: profile.modelId, revision: profile.revision, image: profile.vllmImage, profileHash: hashJson(profile) },
    maxTokens,
    decode,
    service,
    health: endpointReceipt(health),
    models: endpointReceipt(models),
    gpu: { before: gpuBefore, during: gpuDuring },
    metrics: { before: metricsBefore, after: metricsAfter },
    metricDelta,
    effectiveSamplingParamsLine,
    calls: calls.map(summarizeCall),
    determinism: { repeatedCalls: contentHashes.length, exactMatches, exactMatchRate, contentHashes },
    telemetryJournalPath: created.path,
    ingestion,
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const serialized = `${JSON.stringify(receipt, null, 2)}\n`;
  await atomicWrite(`${rawDirectory}/receipt.json`, serialized, 0o600);
  await atomicWrite(receiptPath, serialized, 0o600);
  await persistEvidence("PASS", receipt);
  console.log(JSON.stringify({
    runId: run.runId,
    profileKey,
    calls: calls.length,
    metricDelta,
    exactMatchRate,
    gpuMemoryUsedMiB: gpuDuring.memoryUsedMiB,
    disposition: "PASS",
    receiptPath,
    receiptSha256: receipt.receiptSha256,
  }, null, 2));
} catch (error) {
  const errorDetail = safeError(error);
  await created.journal.record("point", "state.chat_port_gate_failed", {
    traceId: gateSpan.traceId,
    parentSpanId: gateSpan.spanId,
  }, { gateId, profileKey, errorDetail }, { status: "failed" }).catch(() => undefined);
  await created.journal.endSpan(gateSpan, "failed", { gateId, profileKey, errorDetail }).catch(() => undefined);
  await created.journal.flush().catch(() => undefined);
  const ingestion = await ingestJournal().catch((ingestionError) => ({ status: "failed", errorDetail: safeError(ingestionError) }));
  const failureBody = {
    schemaVersion: 1,
    runId: run.runId,
    gateId,
    profileKey,
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    maxTokens,
    errorDetail,
    telemetryJournalPath: created.path,
    ingestion,
    disposition: "STOP_PORT",
  };
  const failure = { ...failureBody, receiptSha256: hashJson(failureBody) };
  await atomicWrite(`${rawDirectory}/failure-receipt.json`, `${JSON.stringify(failure, null, 2)}\n`, 0o600).catch(() => undefined);
  await persistEvidence("STOP_PORT", failure).catch(() => undefined);
  console.error(JSON.stringify(failure, null, 2));
  process.exitCode = 1;
}

async function callCanary(packet: Record<string, unknown>, decode: Record<string, unknown>): Promise<GatewayCallResult> {
  const messages = buildInitialAgentMessages({ arm: "A-direct", packet });
  const result = await callOpenAiCompatibleModel({
    endpoint: `${config.inference.chatBaseUrl}/chat/completions`,
    model: profile.modelId,
    messages,
    decode,
    toolRegistry: [],
    journal: created.journal,
    runDirectory,
    context: { episodeId: String(packet.episodeId), traceId: gateSpan.traceId, parentSpanId: gateSpan.spanId },
    timeoutMs: 300_000,
    maxResponseBytes: 4 * 1024 * 1024,
    maxRetries: 0,
  });
  const request = JSON.parse(result.attempts[0]!.requestBody.toString("utf8")) as Record<string, unknown>;
  if ("tools" in request || "response_format" in request) throw new Error("Primary port gate sent a constrained/native-tool transport field");
  const sent = request.messages as Array<{ role?: string }>;
  if (!Array.isArray(sent) || sent.length !== 1 || sent[0]?.role !== "user") throw new Error("Primary port gate did not use exactly one initial user turn");
  return result;
}

function validateCall(result: GatewayCallResult): void {
  if (result.status !== "success" || result.value?.kind !== "decision")
    throw new Error(`Canary did not produce a structured decision: ${result.errorClass}: ${result.errorDetail}`);
  const attempt = result.attempts.at(-1)!;
  if (attempt.httpStatus === null || attempt.httpStatus < 200 || attempt.httpStatus >= 300) throw new Error("Canary HTTP status was not 2xx");
  if (attempt.promptTokens === null || attempt.completionTokens === null || attempt.totalTokens === null) throw new Error("Canary omitted server token usage");
  if (attempt.finishReason === "length") throw new Error("Canary exhausted the governed completion budget");
}

async function waitReady(): Promise<void> {
  const deadline = Date.now() + 3_600_000;
  let lastError = "service not ready";
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${config.inference.chatBaseUrl}/models`, { signal: AbortSignal.timeout(5_000) });
      if (response.ok) return;
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = safeError(error);
    }
    await delay(5_000);
  }
  throw new Error(`Timed out waiting for chat service: ${lastError}`);
}

async function inspectService(): Promise<Record<string, unknown>> {
  const raw = await command("docker", ["inspect", containerName]);
  const value = JSON.parse(raw)[0] as {
    Id: string;
    Image: string;
    Config: { Image: string; Cmd: string[]; Labels: Record<string, string> };
    State: { Status: string; StartedAt: string };
  };
  const expectedImageId = (await command("docker", ["image", "inspect", profile.vllmImage, "--format", "{{.Id}}"])) .trim();
  if (value.State.Status !== "running") throw new Error("Chat container is not running");
  if (value.Config.Image !== profile.vllmImage || value.Image !== expectedImageId) throw new Error("Chat container image identity does not match the registry digest");
  if (value.Config.Labels["ai.labs.model-profile"] !== profileKey || value.Config.Labels["ai.labs.model-profile-hash"] !== hashJson(profile))
    throw new Error("Chat container profile labels do not match the registry");
  assertCommand(value.Config.Cmd, "--model", profile.modelId);
  assertCommand(value.Config.Cmd, "--revision", profile.revision);
  assertCommand(value.Config.Cmd, "--served-model-name", profile.modelId);
  assertCommand(value.Config.Cmd, "--max-num-seqs", process.env.LOGWARDEN_CHAT_MAX_NUM_SEQS ?? "64");
  if (!value.Config.Cmd.includes("--generation-config") || !value.Config.Cmd.includes("vllm")) throw new Error("Chat service omitted --generation-config vllm");
  return {
    containerId: value.Id,
    imageId: value.Image,
    configuredImage: value.Config.Image,
    command: value.Config.Cmd,
    labels: value.Config.Labels,
    state: value.State.Status,
    startedAtUtc: value.State.StartedAt,
  };
}

async function retainEndpoint(name: string, endpoint: URL): Promise<{ status: number; contentType: string | null; body: string; path: string; sha256: string }> {
  const response = await fetch(endpoint, { signal: AbortSignal.timeout(30_000) });
  const body = await response.text();
  const path = `${rawDirectory}/${name}.bin`;
  await atomicWrite(path, body, 0o600);
  return { status: response.status, contentType: response.headers.get("content-type"), body, path, sha256: sha256(body) };
}

async function retainMetrics(label: string) {
  const snapshot = await retainEndpoint(`metrics-${label}`, new URL("/metrics", config.inference.chatBaseUrl));
  if (snapshot.status < 200 || snapshot.status >= 300) throw new Error(`Chat metrics returned HTTP ${snapshot.status}`);
  return { ...endpointReceipt(snapshot), summary: summarizeChatMetrics(snapshot.body, profile.modelId) };
}

async function retainGpu(label: string): Promise<{ label: string; atUtc: string; memoryUsedMiB: number; utilizationGpuPct: number; processes: string[]; rawPath: string; rawSha256: string }> {
  const summary = await command("nvidia-smi", ["--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"]);
  const processes = await command("nvidia-smi", ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]).catch(() => "");
  const [memory, utilization] = summary.trim().split(",").map((value) => Number(value.trim()));
  const raw = `${summary.trim()}\n${processes.trim()}\n`;
  const rawPath = `${rawDirectory}/gpu-${label}.txt`;
  await atomicWrite(rawPath, raw, 0o600);
  return { label, atUtc: new Date().toISOString(), memoryUsedMiB: memory!, utilizationGpuPct: utilization!, processes: processes.split("\n").filter(Boolean), rawPath, rawSha256: sha256(raw) };
}

async function requireFreezeForTarget(): Promise<void> {
  if (profileKey === "qwen-smoke") return;
  try {
    const freeze = JSON.parse(await readFile(`${runDirectory}/manifests/freeze.json`, "utf8")) as { disposition?: string; targetProfiles?: string[] };
    if (freeze.disposition !== "FROZEN" || !freeze.targetProfiles?.includes(profileKey)) throw new Error("freeze does not authorize profile");
  } catch (error) {
    throw new Error(`STOP_DATA: target profile cannot load before the governed freeze: ${safeError(error)}`);
  }
}

async function ingestJournal() {
  const pool = await connect(config.databases.lab, config.databases.controlName, 300_000);
  try { return await ingestTelemetryJournal(pool, run.runId, created.path, { batchSize: 100 }); }
  finally { await pool.close(); }
}

async function persistEvidence(disposition: "PASS" | "STOP_PORT", detail: unknown): Promise<void> {
  const pool = await connect(config.databases.lab, config.databases.controlName);
  const detailJson = canonicalJson(detail);
  try {
    await pool.request()
      .input("key", sql.VarChar(120), `chat-port-${profileKey}-${gateId}`)
      .input("run", sql.VarChar(120), run.runId)
      .input("disposition", sql.VarChar(40), disposition)
      .input("detail", sql.NVarChar(sql.MAX), detailJson)
      .input("hash", sql.Char(64), sha256(detailJson))
      .query(`INSERT control.evidence_events(event_key,run_id,stage,scientific_tier,disposition,detail_json,detail_sha256)
        VALUES(@key,@run,'chat_port_gate','tier1',@disposition,@detail,@hash);`);
  } finally { await pool.close(); }
}

function canaryPackets(): Array<Record<string, unknown>> {
  const fixtures = [
    { eventName: "rpc_completed", errorNumber: null, severity: null, message: "A bounded statement completed successfully under the alert threshold." },
    { eventName: "error_reported", errorNumber: 1205, severity: 13, message: "Transaction was deadlocked on lock resources and was chosen as the deadlock victim." },
    { eventName: "errorlog", errorNumber: 18456, severity: 14, message: "Login failed for user UNKNOWN_LOGIN; reason: password validation failed." },
    { eventName: "error_reported", errorNumber: 50000, severity: 11, message: "Evidence is incomplete and classification is uncertain." },
  ];
  return fixtures.map((event, index) => ({
    packetVersion: "port-gate-v1",
    episodeId: `port-gate-${profileKey}-${index}`,
    anchorTimeUtc: "2026-01-01T00:00:00.000Z",
    sourceEvents: [{ canonicalEventId: String(index + 1), sourceKind: "synthetic_port_gate", occurredAtUtc: "2026-01-01T00:00:00.000Z", state: null, clientAppName: "LogWarden-Port-Gate", fingerprint: sha256(canonicalJson(event)), ...event }],
    recentHistory: { sameFingerprint5m: 0, sameClass1h: 0, openRelatedIncidents: 0 },
    availableTools: [],
    frozenToolSnapshots: {},
    sourceDiagnostics: { linkedEventCount: 1, sources: ["synthetic_port_gate"] },
    redactions: [],
    hashes: { sourceSetSha256: sha256(canonicalJson(event)) },
  }));
}

function summarizeCall(result: GatewayCallResult): Record<string, unknown> {
  return {
    operationId: result.operationId,
    traceId: result.traceId,
    status: result.status,
    terminalState: result.terminalState,
    elapsedMs: result.elapsedMs,
    valueSha256: result.value === null ? null : hashJson(result.value),
    attempts: result.attempts.map((attempt) => ({
      clientRequestId: attempt.clientRequestId,
      requestBodySha256: attempt.requestBodySha256,
      responseBodySha256: attempt.responseBodySha256,
      requestPath: attempt.requestPath,
      responsePath: attempt.responsePath,
      metadataPath: attempt.metadataPath,
      httpStatus: attempt.httpStatus,
      finishReason: attempt.finishReason,
      promptTokens: attempt.promptTokens,
      completionTokens: attempt.completionTokens,
      totalTokens: attempt.totalTokens,
      clientElapsedMs: attempt.clientElapsedMs,
      headersWaitMs: attempt.headersWaitMs,
      bodyReadMs: attempt.bodyReadMs,
      parseMs: attempt.parseMs,
      repairKind: attempt.repairKind,
      reasoningSha256: attempt.reasoningContent === null ? null : sha256(attempt.reasoningContent),
      reasoningBytes: attempt.reasoningContent === null ? null : Buffer.byteLength(attempt.reasoningContent),
      errorClass: attempt.errorClass,
    })),
  };
}

function endpointReceipt(value: { status: number; contentType: string | null; body: string; path: string; sha256: string }) {
  return { status: value.status, contentType: value.contentType, bytes: Buffer.byteLength(value.body), path: value.path, sha256: value.sha256 };
}

function assertCommand(command: string[], flag: string, expected: string): void {
  const index = command.indexOf(flag);
  if (index < 0 || command[index + 1] !== expected) throw new Error(`Chat service ${flag} does not equal ${expected}`);
}

async function command(executable: string, args: string[]): Promise<string> {
  const result = await execFile(executable, args, { encoding: "utf8", maxBuffer: 32 * 1024 * 1024 });
  return `${result.stdout}${result.stderr}`;
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}

function delay(milliseconds: number): Promise<void> { return new Promise((resolve) => setTimeout(resolve, milliseconds)); }
function safeError(error: unknown): string { return (error as { message?: string }).message ?? String(error); }
