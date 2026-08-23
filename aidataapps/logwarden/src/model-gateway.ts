import { randomUUID } from "node:crypto";
import { performance } from "node:perf_hooks";
import { AgentResponseParseError, parseAgentResponse, type AgentResponse, type RepairKind } from "./contracts.js";
import { canonicalJson, hashJson, sha256 } from "./hash.js";
import { atomicWrite } from "./run.js";
import type { FileTelemetryJournal } from "./telemetry.js";

export type ModelFailureClass =
  | "http_error"
  | "timeout"
  | "network_error"
  | "output_limit"
  | "empty_output"
  | "invalid_json"
  | "response_envelope_invalid"
  | "contract_schema"
  | "telemetry_failure";

export interface GatewayMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  name?: string;
}

export interface ModelCallContext {
  jobId?: number;
  episodeId?: string;
  attemptId?: number;
  traceId?: string;
  parentSpanId?: string;
}

export interface GatewayCallOptions {
  endpoint: string;
  model: string;
  messages: GatewayMessage[];
  decode: Record<string, unknown>;
  toolRegistry?: unknown;
  responseSchema?: unknown;
  responseFormat?: unknown;
  journal: FileTelemetryJournal;
  runDirectory: string;
  context?: ModelCallContext;
  timeoutMs?: number;
  maxResponseBytes?: number;
  maxRetries?: number;
  retryBackoffMs?: number;
  fetchImpl?: typeof fetch;
}

export interface GatewayAttemptEvidence {
  retryOrdinal: number;
  clientRequestId: string;
  spanId: string;
  requestBody: Buffer;
  requestBodySha256: string;
  requestPath: string;
  responseBody: Buffer;
  responseBodySha256: string;
  responsePath: string;
  metadataPath: string;
  endpoint: string;
  startedAtUtc: string;
  requestWriteFinishedAtUtc: string | null;
  responseHeadersAtUtc: string | null;
  firstContentAtUtc: string | null;
  bodyFinishedAtUtc: string | null;
  clientElapsedMs: number;
  connectWriteMs: null;
  headersWaitMs: number | null;
  bodyReadMs: number | null;
  httpStatus: number | null;
  serviceRequestId: string | null;
  finishReason: string | null;
  promptTokens: number | null;
  completionTokens: number | null;
  totalTokens: number | null;
  content: string | null;
  repairKind: RepairKind | "rejected";
  parsedValue: AgentResponse | null;
  parseMs: number | null;
  parseStartedAtUtc: string | null;
  parseFinishedAtUtc: string | null;
  status: "success" | "retry" | "failed";
  errorClass: ModelFailureClass | null;
  errorDetail: string | null;
  truncated: boolean;
}

export interface GatewayCallResult {
  operationId: string;
  traceId: string;
  rootSpanId: string;
  status: "success" | "failed";
  terminalState: "decision_received" | "tool_requested" | "contract_rejected" | "model_timeout" | "retryable_failure";
  value: AgentResponse | null;
  errorClass: ModelFailureClass | null;
  errorDetail: string | null;
  attempts: GatewayAttemptEvidence[];
  elapsedMs: number;
  identities: {
    messagesSha256: string;
    promptSha256: string;
    toolRegistrySha256: string;
    responseSchemaSha256: string;
    decodeConfigSha256: string;
    endpointIdentitySha256: string;
  };
}

interface OpenAiEnvelope {
  id?: string;
  choices?: Array<{ finish_reason?: string | null; message?: { content?: string | null } }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
}

export async function callOpenAiCompatibleModel(options: GatewayCallOptions): Promise<GatewayCallResult> {
  const operationId = randomUUID();
  const timeoutMs = options.timeoutMs ?? 180_000;
  const maxResponseBytes = options.maxResponseBytes ?? 2 * 1024 * 1024;
  const maxRetries = options.maxRetries ?? 0;
  const retryBackoffMs = options.retryBackoffMs ?? 250;
  const fetchImpl = options.fetchImpl ?? fetch;
  const context = options.context ?? {};
  const identities = {
    messagesSha256: hashJson(options.messages),
    promptSha256: sha256(options.messages.map((message) => `${message.role}\n${message.content}`).join("\n\n")),
    toolRegistrySha256: hashJson(options.toolRegistry ?? []),
    responseSchemaSha256: hashJson(options.responseSchema ?? { contract: "logwarden-json-v1" }),
    decodeConfigSha256: hashJson(options.decode),
    endpointIdentitySha256: sha256(new URL(options.endpoint).origin),
  };
  const root = await options.journal.startSpan("model.operation", context, {
    operationId,
    model: options.model,
    timeoutMs,
    maxResponseBytes,
    maxRetries,
    ...identities,
  });
  const operationStarted = performance.now();
  const attempts: GatewayAttemptEvidence[] = [];
  let finalErrorClass: ModelFailureClass | null = null;
  let finalErrorDetail: string | null = null;

  try {
    for (let retryOrdinal = 0; retryOrdinal <= maxRetries; retryOrdinal += 1) {
      const clientRequestId = `${operationId}:${retryOrdinal}`;
      const requestObject = {
        model: options.model,
        messages: options.messages,
        stream: false,
        ...(options.responseFormat === undefined ? {} : { response_format: options.responseFormat }),
        ...options.decode,
      };
      const requestBody = Buffer.from(canonicalJson(requestObject));
      const rawBase = `${options.runDirectory}/raw/model/${operationId}/attempt-${retryOrdinal}`;
      const requestPath = `${rawBase}/request.json`;
      const responsePath = `${rawBase}/response.bin`;
      const metadataPath = `${rawBase}/metadata.json`;
      await atomicWrite(requestPath, requestBody);
      const requestWriteFinishedAtUtc = new Date().toISOString();
      await options.journal.record("point", "model.request.durable", {
        ...context, traceId: root.traceId, parentSpanId: root.spanId,
      }, { operationId, clientRequestId, retryOrdinal, path: requestPath, bytes: requestBody.length, sha256: sha256(requestBody) });

      const attemptSpan = await options.journal.startSpan("model.request", {
        ...context, traceId: root.traceId, parentSpanId: root.spanId,
      }, { operationId, clientRequestId, retryOrdinal, requestBodySha256: sha256(requestBody) });
      const attempt = await executeAttempt({
        endpoint: options.endpoint,
        model: options.model,
        requestBody,
        requestPath,
        responsePath,
        metadataPath,
        retryOrdinal,
        clientRequestId,
        spanId: attemptSpan.spanId,
        requestWriteFinishedAtUtc,
        timeoutMs,
        maxResponseBytes,
        fetchImpl,
      });
      const retry = attempt.errorClass !== null && retryOrdinal < maxRetries && isRetryable(attempt.errorClass, attempt.httpStatus);
      attempt.status = attempt.errorClass === null ? "success" : retry ? "retry" : "failed";
      attempts.push(attempt);
      await atomicWrite(metadataPath, `${JSON.stringify(attemptMetadata(attempt), null, 2)}\n`);
      await options.journal.record("point", "model.response.durable", {
        ...context, traceId: root.traceId, parentSpanId: attemptSpan.spanId,
      }, {
        operationId, clientRequestId, retryOrdinal, path: responsePath,
        bytes: attempt.responseBody.length, sha256: attempt.responseBodySha256,
        httpStatus: attempt.httpStatus, truncated: attempt.truncated,
      }, { status: attempt.status });
      await options.journal.endSpan(attemptSpan, attempt.status, {
        clientRequestId,
        httpStatus: attempt.httpStatus,
        errorClass: attempt.errorClass,
        clientElapsedMs: attempt.clientElapsedMs,
        headersWaitMs: attempt.headersWaitMs,
        bodyReadMs: attempt.bodyReadMs,
        parseMs: attempt.parseMs,
        promptTokens: attempt.promptTokens,
        completionTokens: attempt.completionTokens,
        totalTokens: attempt.totalTokens,
        tokenCountProvenance: attempt.totalTokens === null ? null : "server_reported",
      });

      if (attempt.errorClass === null && attempt.parsedValue !== null) {
        const terminalState = attempt.parsedValue.kind === "decision" ? "decision_received" : "tool_requested";
        await options.journal.record("point", `state.${terminalState}`, {
          ...context, traceId: root.traceId, parentSpanId: root.spanId,
        }, { operationId, clientRequestId, retryOrdinal });
        const elapsedMs = round(performance.now() - operationStarted);
        await options.journal.endSpan(root, "success", { operationId, terminalState, attempts: attempts.length });
        await options.journal.flush();
        return {
          operationId, traceId: root.traceId, rootSpanId: root.spanId, status: "success", terminalState,
          value: attempt.parsedValue, errorClass: null, errorDetail: null, attempts, elapsedMs, identities,
        };
      }

      finalErrorClass = attempt.errorClass;
      finalErrorDetail = attempt.errorDetail;
      if (!retry) break;
      const backoff = retryBackoffMs * 2 ** retryOrdinal;
      await options.journal.record("point", "model.retry.backoff", {
        ...context, traceId: root.traceId, parentSpanId: root.spanId,
      }, { operationId, clientRequestId, retryOrdinal, backoffMs: backoff, errorClass: attempt.errorClass });
      await delay(backoff);
    }

    const terminalState = terminalStateFor(finalErrorClass);
    await options.journal.record("point", `state.${terminalState}`, {
      ...context, traceId: root.traceId, parentSpanId: root.spanId,
    }, { operationId, errorClass: finalErrorClass, errorDetail: finalErrorDetail, attempts: attempts.length }, { status: "failed" });
    const elapsedMs = round(performance.now() - operationStarted);
    await options.journal.endSpan(root, "failed", { operationId, terminalState, errorClass: finalErrorClass, attempts: attempts.length });
    await options.journal.flush();
    return {
      operationId, traceId: root.traceId, rootSpanId: root.spanId, status: "failed", terminalState,
      value: null, errorClass: finalErrorClass, errorDetail: finalErrorDetail, attempts, elapsedMs, identities,
    };
  } catch (error) {
    const detail = safeError(error);
    await options.journal.record("point", "state.retryable_failure", {
      ...context, traceId: root.traceId, parentSpanId: root.spanId,
    }, { operationId, errorClass: "telemetry_failure", errorDetail: detail }, { status: "failed" }).catch(() => undefined);
    await options.journal.endSpan(root, "failed", { operationId, errorClass: "telemetry_failure" }).catch(() => undefined);
    await options.journal.flush().catch(() => undefined);
    throw error;
  }
}

async function executeAttempt(input: {
  endpoint: string;
  model: string;
  requestBody: Buffer;
  requestPath: string;
  responsePath: string;
  metadataPath: string;
  retryOrdinal: number;
  clientRequestId: string;
  spanId: string;
  requestWriteFinishedAtUtc: string;
  timeoutMs: number;
  maxResponseBytes: number;
  fetchImpl: typeof fetch;
}): Promise<GatewayAttemptEvidence> {
  const startedAtUtc = new Date().toISOString();
  const started = performance.now();
  let headersAtUtc: string | null = null;
  let firstContentAtUtc: string | null = null;
  let bodyFinishedAtUtc: string | null = null;
  let headersAtMono: number | null = null;
  let firstContentAtMono: number | null = null;
  let responseBody: Buffer<ArrayBufferLike> = Buffer.alloc(0);
  let httpStatus: number | null = null;
  let serviceRequestId: string | null = null;
  let finishReason: string | null = null;
  let promptTokens: number | null = null;
  let completionTokens: number | null = null;
  let totalTokens: number | null = null;
  let content: string | null = null;
  let repairKind: RepairKind | "rejected" = "rejected";
  let parsedValue: AgentResponse | null = null;
  let parseMs: number | null = null;
  let parseStartedAtUtc: string | null = null;
  let parseFinishedAtUtc: string | null = null;
  let errorClass: ModelFailureClass | null = null;
  let errorDetail: string | null = null;
  let truncated = false;
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, input.timeoutMs);
  try {
    const response = await input.fetchImpl(input.endpoint, {
      method: "POST",
      headers: { "content-type": "application/json", "x-request-id": input.clientRequestId },
      body: input.requestBody.toString("utf8"),
      signal: controller.signal,
    });
    httpStatus = response.status;
    serviceRequestId = response.headers.get("x-request-id") ?? response.headers.get("x-correlation-id");
    headersAtMono = performance.now();
    headersAtUtc = new Date().toISOString();
    const body = await readLimitedBody(response, input.maxResponseBytes, () => {
      if (firstContentAtUtc === null) {
        firstContentAtUtc = new Date().toISOString();
        firstContentAtMono = performance.now();
      }
    });
    responseBody = body.body;
    truncated = body.truncated;
    bodyFinishedAtUtc = new Date().toISOString();
    await atomicWrite(input.responsePath, responseBody);
    if (truncated) {
      errorClass = "output_limit";
      errorDetail = `Response exceeded ${input.maxResponseBytes} byte limit`;
    } else if (!response.ok) {
      errorClass = "http_error";
      errorDetail = `HTTP ${response.status}`;
    } else if (responseBody.length === 0) {
      errorClass = "empty_output";
      errorDetail = "Model service returned an empty body";
    } else {
      const parseStarted = performance.now();
      parseStartedAtUtc = new Date().toISOString();
      try {
        let envelope: OpenAiEnvelope;
        try {
          envelope = JSON.parse(responseBody.toString("utf8")) as OpenAiEnvelope;
        } catch (error) {
          throw new AgentResponseParseError("invalid_json", "OpenAI-compatible response envelope is not JSON", { cause: error });
        }
        const choice = envelope.choices?.[0];
        if (choice?.message === undefined || typeof choice.message.content !== "string") {
          errorClass = "response_envelope_invalid";
          errorDetail = "Response is missing choices[0].message.content";
        } else {
          content = choice.message.content;
          finishReason = choice.finish_reason ?? null;
          promptTokens = finiteInteger(envelope.usage?.prompt_tokens);
          completionTokens = finiteInteger(envelope.usage?.completion_tokens);
          totalTokens = finiteInteger(envelope.usage?.total_tokens);
          const parsed = parseAgentResponse(content);
          repairKind = parsed.repairKind;
          parsedValue = parsed.value;
        }
      } catch (error) {
        if (error instanceof AgentResponseParseError) {
          errorClass = error.errorClass;
          errorDetail = error.message;
        } else {
          errorClass = "invalid_json";
          errorDetail = safeError(error);
        }
      } finally {
        parseMs = round(performance.now() - parseStarted);
        parseFinishedAtUtc = new Date().toISOString();
      }
    }
  } catch (error) {
    errorClass = timedOut ? "timeout" : "network_error";
    errorDetail = timedOut ? `Model request exceeded ${input.timeoutMs} ms` : safeError(error);
    bodyFinishedAtUtc = new Date().toISOString();
    await atomicWrite(input.responsePath, responseBody);
  } finally {
    clearTimeout(timer);
  }
  const finished = performance.now();
  return {
    retryOrdinal: input.retryOrdinal,
    clientRequestId: input.clientRequestId,
    spanId: input.spanId,
    requestBody: input.requestBody,
    requestBodySha256: sha256(input.requestBody),
    requestPath: input.requestPath,
    responseBody,
    responseBodySha256: sha256(responseBody),
    responsePath: input.responsePath,
    metadataPath: input.metadataPath,
    endpoint: input.endpoint,
    startedAtUtc,
    requestWriteFinishedAtUtc: input.requestWriteFinishedAtUtc,
    responseHeadersAtUtc: headersAtUtc,
    firstContentAtUtc,
    bodyFinishedAtUtc,
    clientElapsedMs: round(finished - started),
    connectWriteMs: null,
    headersWaitMs: headersAtMono === null ? null : round(headersAtMono - started),
    bodyReadMs: headersAtMono === null ? null : round(finished - headersAtMono),
    httpStatus,
    serviceRequestId,
    finishReason,
    promptTokens,
    completionTokens,
    totalTokens,
    content,
    repairKind,
    parsedValue,
    parseMs,
    parseStartedAtUtc,
    parseFinishedAtUtc,
    status: "failed",
    errorClass,
    errorDetail,
    truncated,
  };
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

function attemptMetadata(attempt: GatewayAttemptEvidence): Record<string, unknown> {
  const { requestBody: _requestBody, responseBody: _responseBody, parsedValue, content, ...metadata } = attempt;
  return {
    schemaVersion: 1,
    ...metadata,
    parsedValueSha256: parsedValue === null ? null : hashJson(parsedValue),
    contentSha256: content === null ? null : sha256(content),
  };
}

function terminalStateFor(errorClass: ModelFailureClass | null): GatewayCallResult["terminalState"] {
  if (errorClass === "timeout") return "model_timeout";
  if (errorClass === "empty_output" || errorClass === "invalid_json" || errorClass === "response_envelope_invalid" || errorClass === "contract_schema" || errorClass === "output_limit") return "contract_rejected";
  return "retryable_failure";
}

function isRetryable(errorClass: ModelFailureClass, httpStatus: number | null): boolean {
  return errorClass === "timeout" || errorClass === "network_error" || (errorClass === "http_error" && httpStatus !== null && [429, 500, 502, 503, 504].includes(httpStatus));
}

function finiteInteger(value: number | undefined): number | null {
  return Number.isSafeInteger(value) && (value ?? -1) >= 0 ? value! : null;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function safeError(error: unknown): string {
  return (error as { message?: string }).message ?? String(error);
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
