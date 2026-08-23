import { execFile as execFileCallback } from "node:child_process";
import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import { promisify } from "node:util";
import { loadConfig } from "../src/config.js";
import {
  summarizeEmbeddingMetrics,
  validateEmbeddingMetricDelta,
  type EmbeddingMetricSummary,
} from "../src/embedding-metrics.js";
import { callOpenAiCompatibleEmbeddings, type EmbeddingCallEvidence } from "../src/embeddings.js";
import { hashJson, sha256 } from "../src/hash.js";
import { resolveEmbeddingProfile } from "../src/models.js";
import { connect } from "../src/repository.js";
import { atomicWrite, LAB_ROOT, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal, type TelemetryIngestionResult } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal, type FileTelemetryJournal, type StartedSpan } from "../src/telemetry.js";

const execFile = promisify(execFileCallback);
const profileKey = argument("--profile") ?? "qwen3-embedding-0.6b";
const config = loadConfig();
const profile = resolveEmbeddingProfile(profileKey);
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const gateId = randomUUID();
const startedAtUtc = new Date().toISOString();
const rawDirectory = `${runDirectory}/raw/embedding-port-gate/${gateId}`;
const receiptPath = `${runDirectory}/metrics/embedding-port-gate-${profileKey}.json`;
const created = await createComponentTelemetryJournal(runDirectory, run.runId, `embedding-port-gate-${profileKey}`);
const gateSpan = await created.journal.startSpan("embedding.port_gate", {}, {
  gateId,
  profileKey,
  modelId: profile.modelId,
  revision: profile.revision,
  dimensions: profile.dimensions,
  vllmImage: profile.vllmImage,
});

try {
  const result = await runGate(created.journal, gateSpan);
  await created.journal.record("point", "state.embedding_port_gate_passed", {
    traceId: gateSpan.traceId,
    parentSpanId: gateSpan.spanId,
  }, { gateId, profileKey, metricDelta: result.metricDelta }, { status: "success" });
  await created.journal.endSpan(gateSpan, "success", {
    gateId,
    profileKey,
    callCount: result.calls.length,
    inputCount: result.calls.reduce((sum, call) => sum + call.inputCount, 0),
  });
  await created.journal.flush();
  const ingestion = await ingestGateJournal(created.path);
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    gateId,
    profileKey,
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    model: {
      modelId: profile.modelId,
      revision: profile.revision,
      dimensions: profile.dimensions,
      vllmImage: profile.vllmImage,
    },
    endpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`,
    rawDirectory,
    telemetryJournalPath: created.path,
    service: result.service,
    health: result.health,
    models: result.models,
    gpu: result.gpu,
    metrics: result.metrics,
    metricDelta: result.metricDelta,
    calls: result.calls,
    determinism: result.determinism,
    ingestion,
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
  console.log(JSON.stringify({
    runId: run.runId,
    gateId,
    profileKey,
    modelId: profile.modelId,
    callCount: result.calls.length,
    inputCount: result.calls.reduce((sum, call) => sum + call.inputCount, 0),
    metricDelta: result.metricDelta,
    gpuMemoryUsedMiB: result.gpu.during.memoryUsedMiB,
    disposition: "PASS",
    receiptPath,
    receiptSha256: receipt.receiptSha256,
  }, null, 2));
} catch (error) {
  const errorDetail = safeError(error);
  await created.journal.record("point", "state.embedding_port_gate_failed", {
    traceId: gateSpan.traceId,
    parentSpanId: gateSpan.spanId,
  }, { gateId, profileKey, errorDetail }, { status: "failed" }).catch(() => undefined);
  await created.journal.endSpan(gateSpan, "failed", { gateId, profileKey, errorDetail }).catch(() => undefined);
  await created.journal.flush().catch(() => undefined);
  const ingestion = await ingestGateJournal(created.path).catch((ingestionError) => ({
    status: "failed",
    errorDetail: safeError(ingestionError),
  }));
  const failedBody = {
    schemaVersion: 1,
    runId: run.runId,
    gateId,
    profileKey,
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    rawDirectory,
    telemetryJournalPath: created.path,
    errorDetail,
    ingestion,
    disposition: "FAIL",
  };
  const failed = { ...failedBody, receiptSha256: hashJson(failedBody) };
  await atomicWrite(`${rawDirectory}/failure-receipt.json`, `${JSON.stringify(failed, null, 2)}\n`).catch(() => undefined);
  console.error(JSON.stringify(failed, null, 2));
  process.exitCode = 1;
}

async function runGate(journal: FileTelemetryJournal, gateSpan: StartedSpan) {
  const service = await inspectEmbeddingService();
  const logs = await command("docker", ["logs", "--timestamps", "--tail", "300", service.containerId]);
  const logPath = `${rawDirectory}/service.log`;
  await atomicWrite(logPath, logs);
  service.logPath = logPath;
  service.logSha256 = sha256(logs);
  if (/CUDA out of memory|Engine core initialization failed|Traceback \(most recent call last\)/i.test(logs)) {
    throw new Error("Embedding service logs contain a fatal startup signature");
  }
  await journal.record("point", "embedding.service.identity_validated", {
    traceId: gateSpan.traceId,
    parentSpanId: gateSpan.spanId,
  }, service);

  const health = await retainEndpointSnapshot("health", new URL("/health", config.inference.qwenEmbeddingBaseUrl), journal, gateSpan);
  const models = await retainEndpointSnapshot("models", new URL("/v1/models", config.inference.qwenEmbeddingBaseUrl), journal, gateSpan);
  validateHealthAndModels(health, models);

  const gpuBefore = await retainGpuSnapshot("before", journal, gateSpan);
  const metricsBefore = await retainMetricSnapshot("before", journal, gateSpan);
  const callContext = { traceId: gateSpan.traceId, parentSpanId: gateSpan.spanId };
  const deterministicInput = "A SQL Server session is blocked behind an open transaction; retrieve the safest diagnostic runbook.";
  const coldResult = await requireEmbeddingSuccess(await callOpenAiCompatibleEmbeddings({
    endpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`,
    model: profile.modelId,
    inputs: [deterministicInput],
    dimensions: profile.dimensions,
    journal,
    runDirectory,
    context: callContext,
  }));
  const warmResult = await requireEmbeddingSuccess(await callOpenAiCompatibleEmbeddings({
    endpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`,
    model: profile.modelId,
    inputs: [deterministicInput],
    dimensions: profile.dimensions,
    journal,
    runDirectory,
    context: callContext,
  }));
  const metricsDuring = await retainMetricSnapshot("during", journal, gateSpan);
  const gpuDuring = await retainGpuSnapshot("during", journal, gateSpan);
  const warmRepeatResult = await requireEmbeddingSuccess(await callOpenAiCompatibleEmbeddings({
    endpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`,
    model: profile.modelId,
    inputs: [deterministicInput],
    dimensions: profile.dimensions,
    journal,
    runDirectory,
    context: callContext,
  }));
  const batchResult = await requireEmbeddingSuccess(await callOpenAiCompatibleEmbeddings({
    endpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`,
    model: profile.modelId,
    inputs: [
      "Find a runbook for a transaction-log-full incident caused by an unavailable log backup path.",
      "Find a runbook for a deadlock graph involving two sessions and reversed update order.",
      "Find a runbook for repeated SQL login failures without exposing supplied credentials.",
    ],
    dimensions: profile.dimensions,
    journal,
    runDirectory,
    context: callContext,
  }));
  const metricsAfter = await retainMetricSnapshot("after", journal, gateSpan);
  const gpuAfter = await retainGpuSnapshot("after", journal, gateSpan);
  const calls = [coldResult.evidence, warmResult.evidence, warmRepeatResult.evidence, batchResult.evidence];
  validateNormalized(calls);
  const coldHash = coldResult.evidence.vectorSha256[0]!;
  const warmHash = warmResult.evidence.vectorSha256[0]!;
  const warmRepeatHash = warmRepeatResult.evidence.vectorSha256[0]!;
  const coldWarmDrift = vectorDrift(coldResult.vectors[0]!, warmResult.vectors[0]!);
  const warmRepeatDrift = vectorDrift(warmResult.vectors[0]!, warmRepeatResult.vectors[0]!);
  if (coldWarmDrift.cosineSimilarity < 0.9999 || coldWarmDrift.maxAbsoluteDifference > 0.001) {
    throw new Error(`Cold-to-warm embedding drift exceeded tolerance: ${JSON.stringify(coldWarmDrift)}`);
  }
  if (warmHash !== warmRepeatHash || warmRepeatDrift.maxAbsoluteDifference !== 0) {
    throw new Error(`Two warmed embeddings for the exact same input were not byte-deterministic: ${JSON.stringify(warmRepeatDrift)}`);
  }
  const inputCount = calls.reduce((sum, call) => sum + call.inputCount, 0);
  const metricDelta = validateEmbeddingMetricDelta(metricsBefore.summary, metricsAfter.summary, calls.length, inputCount);
  if (gpuDuring.memoryUsedMiB < 100 || gpuDuring.processes.length === 0) {
    throw new Error("GPU snapshot did not prove a resident embedding process");
  }
  return {
    service,
    health,
    models,
    gpu: { before: gpuBefore, during: gpuDuring, after: gpuAfter },
    metrics: { before: metricsBefore, during: metricsDuring, after: metricsAfter },
    metricDelta,
    calls,
    determinism: {
      inputSha256: coldResult.evidence.inputSha256[0],
      coldVectorSha256: coldHash,
      warmVectorSha256: warmHash,
      warmRepeatVectorSha256: warmRepeatHash,
      coldWarmExactMatch: coldHash === warmHash,
      warmRepeatExactMatch: true,
      coldWarmDrift,
      warmRepeatDrift,
      coldWarmTolerance: { minimumCosineSimilarity: 0.9999, maximumAbsoluteDifference: 0.001 },
    },
  };
}

async function inspectEmbeddingService(): Promise<{
  containerId: string;
  imageId: string;
  configuredImage: string;
  command: string[];
  state: string;
  startedAtUtc: string;
  vllmSourceRevision: string;
  vllmImageTag: string;
  logPath?: string;
  logSha256?: string;
}> {
  const containerId = (await command("docker", ["compose", "--profile", "embedding", "ps", "-q", "embedding-qwen"])).trim();
  if (!/^[0-9a-f]{12,64}$/.test(containerId)) throw new Error("Could not resolve the embedding-qwen container ID");
  const [imageId, configuredImage, rawCommand, state, startedAtUtc, vllmSourceRevision, vllmImageTag] = await Promise.all([
    inspect(containerId, "{{.Image}}"),
    inspect(containerId, "{{.Config.Image}}"),
    inspect(containerId, "{{json .Config.Cmd}}"),
    inspect(containerId, "{{.State.Status}}"),
    inspect(containerId, "{{.State.StartedAt}}"),
    inspect(containerId, '{{index .Config.Labels "org.opencontainers.image.revision"}}'),
    inspect(containerId, '{{index .Config.Labels "ai.vllm.image.tag"}}'),
  ]);
  const service = {
    containerId,
    imageId: imageId.trim(),
    configuredImage: configuredImage.trim(),
    command: JSON.parse(rawCommand) as string[],
    state: state.trim(),
    startedAtUtc: startedAtUtc.trim(),
    vllmSourceRevision: vllmSourceRevision.trim(),
    vllmImageTag: vllmImageTag.trim(),
  };
  if (service.state !== "running") throw new Error(`Embedding container state is ${service.state}`);
  if (service.configuredImage !== profile.vllmImage) throw new Error("Embedding container image does not match the pinned model registry image");
  if (commandArgument(service.command, "--model") !== profile.modelId) throw new Error("Embedding container model ID does not match the registry");
  if (commandArgument(service.command, "--served-model-name") !== profile.modelId) throw new Error("Embedding served model name does not match the registry");
  if (commandArgument(service.command, "--revision") !== profile.revision) throw new Error("Embedding container revision does not match the registry");
  if (commandArgument(service.command, "--runner") !== "pooling") throw new Error("Embedding container is not using the vLLM pooling runner");
  return service;
}

async function retainEndpointSnapshot(
  name: string,
  endpoint: URL,
  journal: FileTelemetryJournal,
  gateSpan: StartedSpan,
): Promise<{ endpoint: string; path: string; payloadSha256: string; payloadBytes: number; httpStatus: number; contentType: string | null; body: Buffer }> {
  const started = performance.now();
  const response = await fetch(endpoint, { signal: AbortSignal.timeout(10_000) });
  const body = Buffer.from(await response.arrayBuffer());
  const path = `${rawDirectory}/${name}.bin`;
  await atomicWrite(path, body);
  const snapshot = {
    endpoint: endpoint.toString(),
    path,
    payloadSha256: sha256(body),
    payloadBytes: body.length,
    httpStatus: response.status,
    contentType: response.headers.get("content-type"),
    latencyMs: round(performance.now() - started),
  };
  await journal.record("raw_snapshot", `embedding.${name}.snapshot`, {
    traceId: gateSpan.traceId,
    parentSpanId: gateSpan.spanId,
  }, snapshot, { status: response.ok ? "available" : "unavailable" });
  return { ...snapshot, body };
}

function validateHealthAndModels(
  health: { httpStatus: number },
  models: { httpStatus: number; body: Buffer },
): void {
  if (health.httpStatus < 200 || health.httpStatus >= 300) throw new Error(`Embedding health endpoint returned HTTP ${health.httpStatus}`);
  if (models.httpStatus < 200 || models.httpStatus >= 300) throw new Error(`Embedding models endpoint returned HTTP ${models.httpStatus}`);
  let parsed: unknown;
  try {
    parsed = JSON.parse(models.body.toString("utf8"));
  } catch (error) {
    throw new Error("Embedding models endpoint did not return JSON", { cause: error });
  }
  const data = (parsed as { data?: unknown }).data;
  if (!Array.isArray(data)) throw new Error("Embedding models endpoint is missing its data array");
  const ids = data.map((item) => item !== null && typeof item === "object" ? (item as { id?: unknown }).id : undefined);
  if (ids.length !== 1 || ids[0] !== profile.modelId) {
    throw new Error(`Embedding models endpoint did not expose exactly ${profile.modelId}`);
  }
}

async function retainMetricSnapshot(
  phase: "before" | "during" | "after",
  journal: FileTelemetryJournal,
  gateSpan: StartedSpan,
): Promise<{
  phase: string;
  observedAtUtc: string;
  endpoint: string;
  path: string;
  payloadSha256: string;
  payloadBytes: number;
  latencyMs: number;
  summary: EmbeddingMetricSummary;
}> {
  const endpoint = new URL("/metrics", config.inference.qwenEmbeddingBaseUrl);
  const started = performance.now();
  const response = await fetch(endpoint, { signal: AbortSignal.timeout(10_000) });
  const body = Buffer.from(await response.arrayBuffer());
  const observedAtUtc = new Date().toISOString();
  const path = `${rawDirectory}/metrics-${phase}.prom`;
  await atomicWrite(path, body);
  if (!response.ok) throw new Error(`Embedding metrics endpoint returned HTTP ${response.status}`);
  const summary = summarizeEmbeddingMetrics(body.toString("utf8"), profile.modelId);
  const snapshot = {
    phase,
    observedAtUtc,
    endpoint: endpoint.toString(),
    path,
    payloadSha256: sha256(body),
    payloadBytes: body.length,
    latencyMs: round(performance.now() - started),
    summary,
  };
  await journal.record("raw_snapshot", "embedding.metrics.snapshot", {
    traceId: gateSpan.traceId,
    parentSpanId: gateSpan.spanId,
  }, snapshot, { status: "available" });
  return snapshot;
}

async function retainGpuSnapshot(
  phase: "before" | "during" | "after",
  journal: FileTelemetryJournal,
  gateSpan: StartedSpan,
): Promise<{
  phase: string;
  observedAtUtc: string;
  devicePath: string;
  processPath: string;
  deviceSha256: string;
  processSha256: string;
  gpuName: string;
  gpuUuid: string;
  driverVersion: string;
  memoryUsedMiB: number;
  memoryTotalMiB: number;
  utilizationGpuPct: number;
  powerDrawW: number | null;
  processes: string[];
}> {
  const observedAtUtc = new Date().toISOString();
  const deviceRaw = await command("nvidia-smi", [
    "--query-gpu=name,uuid,driver_version,memory.used,memory.total,utilization.gpu,power.draw",
    "--format=csv,noheader,nounits",
  ]);
  const processRaw = await command("nvidia-smi", [
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
  ]).catch(() => "");
  const devicePath = `${rawDirectory}/gpu-${phase}.csv`;
  const processPath = `${rawDirectory}/gpu-processes-${phase}.csv`;
  await Promise.all([atomicWrite(devicePath, deviceRaw), atomicWrite(processPath, processRaw)]);
  const fields = deviceRaw.trim().split(",").map((value) => value.trim());
  if (fields.length < 7) throw new Error("nvidia-smi device snapshot was incomplete");
  const snapshot = {
    phase,
    observedAtUtc,
    devicePath,
    processPath,
    deviceSha256: sha256(deviceRaw),
    processSha256: sha256(processRaw),
    gpuName: fields[0]!,
    gpuUuid: fields[1]!,
    driverVersion: fields[2]!,
    memoryUsedMiB: finiteNumber(fields[3]!, "GPU memory used"),
    memoryTotalMiB: finiteNumber(fields[4]!, "GPU memory total"),
    utilizationGpuPct: finiteNumber(fields[5]!, "GPU utilization"),
    powerDrawW: fields[6] === "[N/A]" || fields[6] === "N/A" ? null : finiteNumber(fields[6]!, "GPU power draw"),
    processes: processRaw.trim().split("\n").map((line) => line.trim()).filter(Boolean),
  };
  await journal.record("raw_snapshot", "embedding.gpu.snapshot", {
    traceId: gateSpan.traceId,
    parentSpanId: gateSpan.spanId,
  }, snapshot, { status: "available" });
  return snapshot;
}

async function requireEmbeddingSuccess(result: Awaited<ReturnType<typeof callOpenAiCompatibleEmbeddings>>): Promise<{
  evidence: EmbeddingCallEvidence;
  vectors: number[][];
}> {
  if (result.status !== "success" || result.vectors === null) {
    throw new Error(`Embedding request failed closed: ${result.evidence.errorClass}: ${result.evidence.errorDetail}`);
  }
  return { evidence: result.evidence, vectors: result.vectors };
}

function validateNormalized(calls: EmbeddingCallEvidence[]): void {
  for (const call of calls) {
    for (const [index, norm] of call.vectorNorms.entries()) {
      if (Math.abs(norm - 1) > 0.0001) {
        throw new Error(`Embedding ${index} in operation ${call.operationId} has L2 norm ${norm}; expected unit normalization`);
      }
    }
  }
}

function vectorDrift(left: number[], right: number[]): {
  dimensions: number;
  changedDimensions: number;
  maxAbsoluteDifference: number;
  meanAbsoluteDifference: number;
  rootMeanSquareError: number;
  cosineSimilarity: number;
} {
  if (left.length !== right.length || left.length === 0) throw new Error("Cannot compare embedding vectors with different or zero dimensions");
  let changedDimensions = 0;
  let maxAbsoluteDifference = 0;
  let sumAbsoluteDifference = 0;
  let sumSquaredDifference = 0;
  let dot = 0;
  let leftNormSquared = 0;
  let rightNormSquared = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index]!;
    const rightValue = right[index]!;
    const difference = Math.abs(leftValue - rightValue);
    if (difference !== 0) changedDimensions += 1;
    maxAbsoluteDifference = Math.max(maxAbsoluteDifference, difference);
    sumAbsoluteDifference += difference;
    sumSquaredDifference += difference * difference;
    dot += leftValue * rightValue;
    leftNormSquared += leftValue * leftValue;
    rightNormSquared += rightValue * rightValue;
  }
  return {
    dimensions: left.length,
    changedDimensions,
    maxAbsoluteDifference,
    meanAbsoluteDifference: sumAbsoluteDifference / left.length,
    rootMeanSquareError: Math.sqrt(sumSquaredDifference / left.length),
    cosineSimilarity: dot / Math.sqrt(leftNormSquared * rightNormSquared),
  };
}

async function ingestGateJournal(path: string): Promise<TelemetryIngestionResult> {
  const pool = await connect(config.databases.lab, config.databases.controlName);
  try {
    return await ingestTelemetryJournal(pool, run.runId, path, { batchSize: 25 });
  } finally {
    await pool.close();
  }
}

async function inspect(containerId: string, format: string): Promise<string> {
  return command("docker", ["inspect", "--format", format, containerId]);
}

async function command(executable: string, args: string[]): Promise<string> {
  const result = await execFile(executable, args, { cwd: LAB_ROOT, maxBuffer: 64 * 1024 * 1024 });
  return result.stdout;
}

function commandArgument(command: string[], name: string): string | undefined {
  const index = command.indexOf(name);
  return index < 0 ? undefined : command[index + 1];
}

function finiteNumber(value: string, name: string): number {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`${name} is not finite: ${value}`);
  return number;
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}

function safeError(error: unknown): string {
  return (error as { message?: string }).message ?? String(error);
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
