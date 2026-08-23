import { randomUUID } from "node:crypto";
import { performance } from "node:perf_hooks";
import { canonicalJson, hashJson, sha256 } from "./hash.js";
import { atomicWrite } from "./run.js";
import type { FileTelemetryJournal } from "./telemetry.js";

export type EmbeddingFailureClass =
  | "http_error"
  | "timeout"
  | "network_error"
  | "output_limit"
  | "empty_output"
  | "invalid_json"
  | "response_envelope_invalid"
  | "count_mismatch"
  | "index_mismatch"
  | "dimension_mismatch"
  | "non_finite_vector"
  | "telemetry_failure";

export interface EmbeddingCallOptions {
  endpoint: string;
  model: string;
  inputs: string[];
  dimensions: number;
  journal: FileTelemetryJournal;
  runDirectory: string;
  context?: {
    jobId?: number;
    episodeId?: string;
    attemptId?: number;
    traceId?: string;
    parentSpanId?: string;
  };
  timeoutMs?: number;
  maxResponseBytes?: number;
  fetchImpl?: typeof fetch;
}

export interface EmbeddingUsage {
  promptTokens: number;
  totalTokens: number;
}

export interface EmbeddingCallEvidence {
  operationId: string;
  clientRequestId: string;
  traceId: string;
  spanId: string;
  endpoint: string;
  model: string;
  inputCount: number;
  inputSha256: string[];
  requestBodySha256: string;
  requestPath: string;
  responseBodySha256: string;
  responsePath: string;
  metadataPath: string;
  startedAtUtc: string;
  requestDurableAtUtc: string;
  responseHeadersAtUtc: string | null;
  firstContentAtUtc: string | null;
  bodyFinishedAtUtc: string | null;
  parseStartedAtUtc: string | null;
  parseFinishedAtUtc: string | null;
  headersWaitMs: number | null;
  bodyReadMs: number | null;
  parseMs: number | null;
  clientElapsedMs: number;
  httpStatus: number | null;
  serviceRequestId: string | null;
  responseBytes: number;
  vectorCount: number | null;
  vectorDimensions: number[];
  vectorSha256: string[];
  vectorNorms: number[];
  usage: EmbeddingUsage | null;
  truncated: boolean;
  status: "success" | "failed";
  errorClass: EmbeddingFailureClass | null;
  errorDetail: string | null;
}

export interface EmbeddingCallResult {
  status: "success" | "failed";
  vectors: number[][] | null;
  evidence: EmbeddingCallEvidence;
}

interface EmbeddingEnvelope {
  model?: unknown;
  data?: unknown;
  usage?: unknown;
}

interface ParsedEmbeddingEnvelope {
  vectors: number[][];
  usage: EmbeddingUsage;
}

/**
 * Calls an OpenAI-compatible embeddings endpoint while preserving exact request
 * and response bytes before network send and response parsing, respectively.
 */
export async function callOpenAiCompatibleEmbeddings(options: EmbeddingCallOptions): Promise<EmbeddingCallResult> {
  if (options.inputs.length === 0) throw new Error("An embedding request must contain at least one input");
  if (!Number.isSafeInteger(options.dimensions) || options.dimensions <= 0) throw new Error("Embedding dimensions must be a positive integer");

  const operationId = randomUUID();
  const clientRequestId = operationId;
  const timeoutMs = options.timeoutMs ?? 180_000;
  const maxResponseBytes = options.maxResponseBytes ?? 64 * 1024 * 1024;
  const fetchImpl = options.fetchImpl ?? fetch;
  const context = options.context ?? {};
  const rawBase = `${options.runDirectory}/raw/embeddings/${operationId}`;
  const requestPath = `${rawBase}/request.json`;
  const responsePath = `${rawBase}/response.bin`;
  const metadataPath = `${rawBase}/metadata.json`;
  const requestBody = Buffer.from(canonicalJson({
    model: options.model,
    input: options.inputs,
    encoding_format: "float",
  }));
  const requestBodySha256 = sha256(requestBody);
  const inputSha256 = options.inputs.map((input) => sha256(input));
  const root = await options.journal.startSpan("embedding.request", context, {
    operationId,
    clientRequestId,
    endpointIdentitySha256: sha256(new URL(options.endpoint).origin),
    model: options.model,
    dimensions: options.dimensions,
    inputCount: options.inputs.length,
    inputSetSha256: hashJson(inputSha256),
    requestBodySha256,
    timeoutMs,
    maxResponseBytes,
  });
  const startedAtUtc = new Date().toISOString();
  const started = performance.now();
  let requestDurableAtUtc = startedAtUtc;
  let responseHeadersAtUtc: string | null = null;
  let firstContentAtUtc: string | null = null;
  let bodyFinishedAtUtc: string | null = null;
  let parseStartedAtUtc: string | null = null;
  let parseFinishedAtUtc: string | null = null;
  let headersAtMono: number | null = null;
  let bodyFinishedAtMono: number | null = null;
  let parseMs: number | null = null;
  let httpStatus: number | null = null;
  let serviceRequestId: string | null = null;
  let responseBody: Buffer<ArrayBufferLike> = Buffer.alloc(0);
  let truncated = false;
  let vectors: number[][] | null = null;
  let usage: EmbeddingUsage | null = null;
  let errorClass: EmbeddingFailureClass | null = null;
  let errorDetail: string | null = null;
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    await atomicWrite(requestPath, requestBody);
    requestDurableAtUtc = new Date().toISOString();
    await options.journal.record("point", "embedding.request.durable", {
      ...context,
      traceId: root.traceId,
      parentSpanId: root.spanId,
    }, {
      operationId,
      clientRequestId,
      path: requestPath,
      bytes: requestBody.length,
      sha256: requestBodySha256,
      inputCount: options.inputs.length,
      inputSha256,
    });

    try {
      const response = await fetchImpl(options.endpoint, {
        method: "POST",
        headers: { "content-type": "application/json", "x-request-id": clientRequestId },
        body: requestBody.toString("utf8"),
        signal: controller.signal,
      });
      httpStatus = response.status;
      serviceRequestId = response.headers.get("x-request-id") ?? response.headers.get("x-correlation-id");
      headersAtMono = performance.now();
      responseHeadersAtUtc = new Date().toISOString();
      const limited = await readLimitedBody(response, maxResponseBytes, () => {
        if (firstContentAtUtc === null) firstContentAtUtc = new Date().toISOString();
      });
      responseBody = limited.body;
      truncated = limited.truncated;
      bodyFinishedAtMono = performance.now();
      bodyFinishedAtUtc = new Date().toISOString();

      // The raw response is made durable before any JSON or vector parsing.
      await atomicWrite(responsePath, responseBody);
      await options.journal.record("point", "embedding.response.durable", {
        ...context,
        traceId: root.traceId,
        parentSpanId: root.spanId,
      }, {
        operationId,
        clientRequestId,
        path: responsePath,
        bytes: responseBody.length,
        sha256: sha256(responseBody),
        httpStatus,
        truncated,
      });

      if (truncated) {
        errorClass = "output_limit";
        errorDetail = `Embedding response exceeded ${maxResponseBytes} byte limit`;
      } else if (!response.ok) {
        errorClass = "http_error";
        errorDetail = `HTTP ${response.status}`;
      } else if (responseBody.length === 0) {
        errorClass = "empty_output";
        errorDetail = "Embedding service returned an empty body";
      } else {
        const parseStarted = performance.now();
        parseStartedAtUtc = new Date().toISOString();
        try {
          const parsed = parseEmbeddingResponse(responseBody, options.inputs.length, options.dimensions);
          vectors = parsed.vectors;
          usage = parsed.usage;
        } catch (error) {
          const classified = classifyParseError(error);
          errorClass = classified.errorClass;
          errorDetail = classified.errorDetail;
        } finally {
          parseMs = round(performance.now() - parseStarted);
          parseFinishedAtUtc = new Date().toISOString();
        }
      }
    } catch (error) {
      errorClass = timedOut ? "timeout" : "network_error";
      errorDetail = timedOut ? `Embedding request exceeded ${timeoutMs} ms` : safeError(error);
      bodyFinishedAtMono = performance.now();
      bodyFinishedAtUtc = new Date().toISOString();
      await atomicWrite(responsePath, responseBody);
      await options.journal.record("point", "embedding.response.durable", {
        ...context,
        traceId: root.traceId,
        parentSpanId: root.spanId,
      }, {
        operationId,
        clientRequestId,
        path: responsePath,
        bytes: 0,
        sha256: sha256(responseBody),
        httpStatus,
        truncated: false,
      }, { status: "failed" });
    }
  } catch (error) {
    errorClass = "telemetry_failure";
    errorDetail = safeError(error);
  } finally {
    clearTimeout(timer);
  }

  const finished = performance.now();
  const vectorDimensions = vectors?.map((vector) => vector.length) ?? [];
  const vectorSha256 = vectors?.map((vector) => hashJson(vector)) ?? [];
  const vectorNorms = vectors?.map(l2Norm) ?? [];
  const status = errorClass === null ? "success" : "failed";
  const evidence: EmbeddingCallEvidence = {
    operationId,
    clientRequestId,
    traceId: root.traceId,
    spanId: root.spanId,
    endpoint: options.endpoint,
    model: options.model,
    inputCount: options.inputs.length,
    inputSha256,
    requestBodySha256,
    requestPath,
    responseBodySha256: sha256(responseBody),
    responsePath,
    metadataPath,
    startedAtUtc,
    requestDurableAtUtc,
    responseHeadersAtUtc,
    firstContentAtUtc,
    bodyFinishedAtUtc,
    parseStartedAtUtc,
    parseFinishedAtUtc,
    headersWaitMs: headersAtMono === null ? null : round(headersAtMono - started),
    bodyReadMs: headersAtMono === null || bodyFinishedAtMono === null ? null : round(bodyFinishedAtMono - headersAtMono),
    parseMs,
    clientElapsedMs: round(finished - started),
    httpStatus,
    serviceRequestId,
    responseBytes: responseBody.length,
    vectorCount: vectors?.length ?? null,
    vectorDimensions,
    vectorSha256,
    vectorNorms: vectorNorms.map(round9),
    usage,
    truncated,
    status,
    errorClass,
    errorDetail,
  };

  try {
    await atomicWrite(metadataPath, `${JSON.stringify({ schemaVersion: 1, ...evidence }, null, 2)}\n`);
    await options.journal.record("point", status === "success" ? "state.embedding_validated" : "state.embedding_rejected", {
      ...context,
      traceId: root.traceId,
      parentSpanId: root.spanId,
    }, {
      operationId,
      clientRequestId,
      status,
      errorClass,
      errorDetail,
      vectorCount: evidence.vectorCount,
      vectorDimensions,
      vectorSha256,
      vectorNorms: evidence.vectorNorms,
      usage,
      metadataPath,
    }, { status });
    await options.journal.endSpan(root, status, {
      operationId,
      clientRequestId,
      errorClass,
      clientElapsedMs: evidence.clientElapsedMs,
      headersWaitMs: evidence.headersWaitMs,
      bodyReadMs: evidence.bodyReadMs,
      parseMs,
      inputCount: options.inputs.length,
      vectorCount: evidence.vectorCount,
      promptTokens: usage?.promptTokens ?? null,
      tokenCountProvenance: usage === null ? null : "server_reported",
    });
    await options.journal.flush();
  } catch (error) {
    throw new Error(`Embedding telemetry finalization failed: ${safeError(error)}`, { cause: error });
  }
  return { status, vectors, evidence };
}

export function parseEmbeddingResponse(
  responseBody: Buffer | string,
  expectedCount: number,
  expectedDimensions: number,
): ParsedEmbeddingEnvelope {
  let envelope: EmbeddingEnvelope;
  try {
    envelope = JSON.parse(typeof responseBody === "string" ? responseBody : responseBody.toString("utf8")) as EmbeddingEnvelope;
  } catch (error) {
    throw new EmbeddingResponseError("invalid_json", "OpenAI-compatible embedding response is not JSON", error);
  }
  if (!Array.isArray(envelope.data)) {
    throw new EmbeddingResponseError("response_envelope_invalid", "Embedding response is missing a data array");
  }
  if (envelope.data.length !== expectedCount) {
    throw new EmbeddingResponseError("count_mismatch", `Expected ${expectedCount} embeddings, received ${envelope.data.length}`);
  }
  const indexed = envelope.data.map((item, ordinal) => {
    if (item === null || typeof item !== "object" || Array.isArray(item)) {
      throw new EmbeddingResponseError("response_envelope_invalid", `Embedding data[${ordinal}] is not an object`);
    }
    const candidate = item as { index?: unknown; embedding?: unknown };
    if (!Number.isSafeInteger(candidate.index) || (candidate.index as number) < 0) {
      throw new EmbeddingResponseError("index_mismatch", `Embedding data[${ordinal}] has an invalid index`);
    }
    if (!Array.isArray(candidate.embedding)) {
      throw new EmbeddingResponseError("response_envelope_invalid", `Embedding data[${ordinal}] is missing its vector`);
    }
    return { index: candidate.index as number, vector: candidate.embedding };
  });
  const seen = new Set(indexed.map((item) => item.index));
  if (seen.size !== expectedCount || indexed.some((item, ordinal) => item.index !== ordinal)) {
    throw new EmbeddingResponseError("index_mismatch", "Embedding response indexes are not the exact ordered sequence 0..N-1");
  }
  const vectors = indexed.map(({ vector }, ordinal) => {
    if (vector.length !== expectedDimensions) {
      throw new EmbeddingResponseError("dimension_mismatch", `Embedding ${ordinal} has ${vector.length} dimensions; expected ${expectedDimensions}`);
    }
    if (!vector.every((value) => typeof value === "number" && Number.isFinite(value))) {
      throw new EmbeddingResponseError("non_finite_vector", `Embedding ${ordinal} contains a non-finite or non-numeric value`);
    }
    return vector as number[];
  });
  if (envelope.usage === null || typeof envelope.usage !== "object" || Array.isArray(envelope.usage)) {
    throw new EmbeddingResponseError("response_envelope_invalid", "Embedding response is missing usage counters");
  }
  const rawUsage = envelope.usage as { prompt_tokens?: unknown; total_tokens?: unknown };
  if (!isNonnegativeInteger(rawUsage.prompt_tokens) || !isNonnegativeInteger(rawUsage.total_tokens)) {
    throw new EmbeddingResponseError("response_envelope_invalid", "Embedding usage counters are not non-negative integers");
  }
  if (rawUsage.total_tokens < rawUsage.prompt_tokens) {
    throw new EmbeddingResponseError("response_envelope_invalid", "Embedding total_tokens is smaller than prompt_tokens");
  }
  return {
    vectors,
    usage: { promptTokens: rawUsage.prompt_tokens, totalTokens: rawUsage.total_tokens },
  };
}

export function l2Norm(vector: number[]): number {
  return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
}

class EmbeddingResponseError extends Error {
  constructor(readonly errorClass: Exclude<EmbeddingFailureClass, "timeout" | "network_error" | "http_error" | "output_limit" | "empty_output" | "telemetry_failure">, message: string, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = "EmbeddingResponseError";
  }
}

function classifyParseError(error: unknown): { errorClass: EmbeddingFailureClass; errorDetail: string } {
  if (error instanceof EmbeddingResponseError) return { errorClass: error.errorClass, errorDetail: error.message };
  return { errorClass: "response_envelope_invalid", errorDetail: safeError(error) };
}

async function readLimitedBody(
  response: Response,
  limit: number,
  onFirstChunk: () => void,
): Promise<{ body: Buffer<ArrayBufferLike>; truncated: boolean }> {
  if (response.body === null) return { body: Buffer.alloc(0), truncated: false };
  const reader = response.body.getReader();
  const chunks: Buffer[] = [];
  let bytes = 0;
  while (true) {
    const next = await reader.read();
    if (next.done) break;
    if (next.value.length > 0) onFirstChunk();
    const remaining = limit - bytes;
    if (next.value.length > remaining) {
      if (remaining > 0) chunks.push(Buffer.from(next.value.subarray(0, remaining)));
      await reader.cancel("response byte limit exceeded").catch(() => undefined);
      return { body: Buffer.concat(chunks), truncated: true };
    }
    chunks.push(Buffer.from(next.value));
    bytes += next.value.length;
  }
  return { body: Buffer.concat(chunks), truncated: false };
}

function isNonnegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function safeError(error: unknown): string {
  return (error as { message?: string }).message ?? String(error);
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function round9(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}
