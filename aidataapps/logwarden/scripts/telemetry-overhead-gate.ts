import { createServer, type ServerResponse } from "node:http";
import { performance } from "node:perf_hooks";
import { readFile } from "node:fs/promises";
import { parseAgentResponse } from "../src/contracts.js";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson } from "../src/hash.js";
import { callOpenAiCompatibleModel } from "../src/model-gateway.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal } from "../src/telemetry.js";

const measuredPerArm = 20;
const warmupsPerArm = 3;
const maxP95AddedMs = 100;
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const config = loadConfig();
const server = createServer((request, response) => {
  request.resume();
  sendOpenAi(response);
});
await new Promise<void>((resolve, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", () => { server.off("error", reject); resolve(); });
});
const address = server.address();
if (address === null || typeof address === "string") throw new Error("Overhead fake gateway failed to bind");
const endpoint = `http://127.0.0.1:${address.port}/v1/chat/completions`;
const messages = [{ role: "user" as const, content: "Synthetic overhead canary." }];
const decode = { temperature: 0, max_tokens: 64, seed: 0 };
const created = await createComponentTelemetryJournal(runDirectory, run.runId, "telemetry-overhead-gate");
const baseline: number[] = [];
const instrumented: number[] = [];

try {
  for (let index = 0; index < warmupsPerArm; index += 1) {
    await bareCall(endpoint, messages, decode);
    await fullCall();
  }
  for (let block = 0; block < measuredPerArm / 2; block += 1) {
    baseline.push(await timed(() => bareCall(endpoint, messages, decode)));
    instrumented.push(await timed(fullCall));
    instrumented.push(await timed(fullCall));
    baseline.push(await timed(() => bareCall(endpoint, messages, decode)));
  }
  const baselineStats = stats(baseline);
  const instrumentedStats = stats(instrumented);
  const overhead = {
    meanAddedMs: round(instrumentedStats.meanMs - baselineStats.meanMs),
    p50AddedMs: round(instrumentedStats.p50Ms - baselineStats.p50Ms),
    p95AddedMs: round(instrumentedStats.p95Ms - baselineStats.p95Ms),
    meanRatio: round(instrumentedStats.meanMs / baselineStats.meanMs),
  };
  const passed = overhead.p95AddedMs <= maxP95AddedMs;
  await created.journal.record("metric", "gate.telemetry_overhead.summary", {}, {
    measuredPerArm,
    warmupsPerArm,
    baseline: baselineStats,
    instrumented: instrumentedStats,
    overhead,
    maxP95AddedMs,
    disposition: passed ? "PASS" : "FAIL",
  }, { status: passed ? "success" : "failed" });
  await created.journal.flush();
  const pool = await connect(config.databases.lab, config.databases.controlName);
  try {
    const ingestion = await ingestTelemetryJournal(pool, run.runId, created.path, { batchSize: 50 });
    const receiptBody = {
      schemaVersion: 1,
      runId: run.runId,
      design: {
        endpointKind: "in-process OpenAI-compatible fake gateway",
        order: "ABBA blocks",
        warmupsPerArm,
        measuredPerArm,
        baseline: "HTTP body read + OpenAI envelope JSON parse + LogWarden contract parse",
        instrumented: "same semantic parse plus hash-chained journal, exact raw request/response files, metadata, spans, and state event",
        durabilityQualification: "filesystem write completion; database ingestion follows the bounded sample",
      },
      baseline: baselineStats,
      instrumented: instrumentedStats,
      overhead,
      acceptance: { maxP95AddedMs, passed },
      journalPath: created.path,
      ingestion,
      disposition: passed ? "PASS" : "FAIL",
    };
    const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
    await atomicWrite(`${runDirectory}/telemetry/overhead-gate.json`, `${JSON.stringify(receipt, null, 2)}\n`);
    console.log(JSON.stringify({ runId: run.runId, baseline: baselineStats, instrumented: instrumentedStats, overhead, disposition: receipt.disposition, receiptSha256: receipt.receiptSha256 }, null, 2));
    if (!passed) throw new Error(`Telemetry p95 added latency exceeded ${maxP95AddedMs} ms`);
  } finally {
    await pool.close();
  }
} finally {
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

async function fullCall(): Promise<void> {
  const result = await callOpenAiCompatibleModel({
    endpoint,
    model: "logwarden-fake-model",
    messages,
    decode,
    journal: created.journal,
    runDirectory,
    timeoutMs: 1_000,
    maxResponseBytes: 4096,
  });
  if (result.status !== "success") throw new Error(`Instrumented overhead canary failed: ${result.errorClass}`);
}

async function bareCall(endpointUrl: string, requestMessages: typeof messages, requestDecode: typeof decode): Promise<void> {
  const body = canonicalJson({ model: "logwarden-fake-model", messages: requestMessages, stream: false, response_format: { type: "json_object" }, ...requestDecode });
  const response = await fetch(endpointUrl, { method: "POST", headers: { "content-type": "application/json" }, body });
  if (!response.ok) throw new Error(`Bare overhead canary HTTP ${response.status}`);
  const envelope = JSON.parse(await response.text()) as { choices?: Array<{ message?: { content?: string } }> };
  const content = envelope.choices?.[0]?.message?.content;
  if (typeof content !== "string") throw new Error("Bare overhead canary lacks content");
  parseAgentResponse(content);
}

async function timed(operation: () => Promise<void>): Promise<number> {
  const started = performance.now();
  await operation();
  return round(performance.now() - started);
}

function stats(values: number[]): { count: number; meanMs: number; p50Ms: number; p95Ms: number; minMs: number; maxMs: number; standardDeviationMs: number } {
  const sorted = [...values].sort((left, right) => left - right);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return {
    count: values.length,
    meanMs: round(mean),
    p50Ms: quantile(sorted, 0.5),
    p95Ms: quantile(sorted, 0.95),
    minMs: sorted[0]!,
    maxMs: sorted.at(-1)!,
    standardDeviationMs: round(Math.sqrt(variance)),
  };
}

function quantile(sorted: number[], probability: number): number {
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(probability * sorted.length) - 1));
  return sorted[index]!;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function sendOpenAi(response: ServerResponse): void {
  response.writeHead(200, { "content-type": "application/json" });
  response.end(JSON.stringify({
    id: "overhead-canary",
    choices: [{ finish_reason: "stop", message: { role: "assistant", content: JSON.stringify({
      kind: "decision",
      incidentClass: "benign_noise",
      severity: "info",
      action: "no_action",
      actionArguments: {},
      citedChunkIds: [],
      confidence: 1,
      abstain: false,
      correlationKey: "overhead-canary",
      summary: "Synthetic canary.",
      rationale: "This response measures instrumentation overhead only.",
    }) } }],
    usage: { prompt_tokens: 4, completion_tokens: 8, total_tokens: 12 },
  }));
}
