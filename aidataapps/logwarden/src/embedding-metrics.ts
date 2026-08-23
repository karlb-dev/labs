import { parsePrometheusExposition, type PrometheusSample } from "./prometheus.js";

export const REQUIRED_EMBEDDING_METRICS = [
  "http_requests_total",
  "vllm:num_requests_running",
  "vllm:num_requests_waiting",
  "vllm:num_preemptions_total",
  "vllm:prompt_tokens_total",
  "vllm:request_success_total",
  "vllm:e2e_request_latency_seconds_count",
] as const;

export interface EmbeddingMetricSummary {
  discoveredMetricNames: string[];
  embeddingHttpRequests: number;
  successfulRequests: number;
  erroredRequests: number;
  promptTokens: number;
  latencyObservations: number;
  preemptions: number;
  runningRequests: number;
  waitingRequests: number;
}

export interface EmbeddingMetricDelta {
  embeddingHttpRequests: number;
  successfulRequests: number;
  erroredRequests: number;
  promptTokens: number;
  latencyObservations: number;
  preemptions: number;
}

export function summarizeEmbeddingMetrics(source: string, modelId: string): EmbeddingMetricSummary {
  const parsed = parsePrometheusExposition(source);
  const missing = REQUIRED_EMBEDDING_METRICS.filter((name) => !parsed.metricNames.includes(name));
  if (missing.length > 0) throw new Error(`Embedding service is missing required metrics: ${missing.join(", ")}`);
  const model = (sample: PrometheusSample) => sample.labels.model_name === modelId;
  const success = (sample: PrometheusSample) => model(sample) && sample.labels.finished_reason === "stop";
  const error = (sample: PrometheusSample) => model(sample) && sample.labels.finished_reason === "error";
  const embeddingHttp = (sample: PrometheusSample) =>
    sample.labels.handler === "/v1/embeddings" && sample.labels.method === "POST" && sample.labels.status === "2xx";
  return {
    discoveredMetricNames: parsed.metricNames,
    // Prometheus does not expose a labeled route series until its first request.
    // The metric family is still required above; an absent embedding-route series
    // is therefore the exact zero baseline, not missing instrumentation.
    embeddingHttpRequests: sumOrZero(parsed.samples, "http_requests_total", embeddingHttp),
    successfulRequests: requiredSum(parsed.samples, "vllm:request_success_total", success),
    erroredRequests: requiredSum(parsed.samples, "vllm:request_success_total", error),
    promptTokens: requiredSum(parsed.samples, "vllm:prompt_tokens_total", model),
    latencyObservations: requiredSum(parsed.samples, "vllm:e2e_request_latency_seconds_count", model),
    preemptions: requiredSum(parsed.samples, "vllm:num_preemptions_total", model),
    runningRequests: requiredSum(parsed.samples, "vllm:num_requests_running", model),
    waitingRequests: requiredSum(parsed.samples, "vllm:num_requests_waiting", model),
  };
}

export function validateEmbeddingMetricDelta(
  before: EmbeddingMetricSummary,
  after: EmbeddingMetricSummary,
  expectedHttpRequests: number,
  expectedInputCount: number,
): EmbeddingMetricDelta {
  const delta = {
    embeddingHttpRequests: counterDelta("embedding HTTP requests", before.embeddingHttpRequests, after.embeddingHttpRequests),
    successfulRequests: counterDelta("successful requests", before.successfulRequests, after.successfulRequests),
    erroredRequests: counterDelta("errored requests", before.erroredRequests, after.erroredRequests),
    promptTokens: counterDelta("prompt tokens", before.promptTokens, after.promptTokens),
    latencyObservations: counterDelta("latency observations", before.latencyObservations, after.latencyObservations),
    preemptions: counterDelta("preemptions", before.preemptions, after.preemptions),
  };
  if (delta.embeddingHttpRequests < expectedHttpRequests) {
    throw new Error(`Embedding HTTP request counter advanced by ${delta.embeddingHttpRequests}; expected at least ${expectedHttpRequests}`);
  }
  if (delta.successfulRequests < expectedInputCount) {
    throw new Error(`vLLM successful-request counter advanced by ${delta.successfulRequests}; expected at least ${expectedInputCount}`);
  }
  if (delta.latencyObservations < expectedInputCount) {
    throw new Error(`vLLM latency histogram advanced by ${delta.latencyObservations}; expected at least ${expectedInputCount}`);
  }
  if (delta.promptTokens <= 0) throw new Error("vLLM prompt-token counter did not advance");
  if (delta.erroredRequests !== 0) throw new Error(`vLLM reported ${delta.erroredRequests} errored embedding requests`);
  if (delta.preemptions !== 0) throw new Error(`vLLM reported ${delta.preemptions} embedding preemptions`);
  return delta;
}

function requiredSum(samples: PrometheusSample[], name: string, predicate: (sample: PrometheusSample) => boolean): number {
  const matches = samples.filter((sample) => sample.name === name && predicate(sample));
  if (matches.length === 0) throw new Error(`Required metric ${name} has no matching labeled series`);
  const values = matches.map((sample) => sample.value);
  if (!values.every(Number.isFinite)) throw new Error(`Required metric ${name} contains a non-finite value`);
  return values.reduce((sum, value) => sum + value, 0);
}

function sumOrZero(samples: PrometheusSample[], name: string, predicate: (sample: PrometheusSample) => boolean): number {
  const values = samples.filter((sample) => sample.name === name && predicate(sample)).map((sample) => sample.value);
  if (!values.every(Number.isFinite)) throw new Error(`Required metric ${name} contains a non-finite value`);
  return values.reduce((sum, value) => sum + value, 0);
}

function counterDelta(name: string, before: number, after: number): number {
  const delta = after - before;
  if (!Number.isFinite(delta) || delta < 0) throw new Error(`The ${name} counter reset or became invalid during the gate`);
  return delta;
}
