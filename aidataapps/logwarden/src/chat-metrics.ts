import { parsePrometheusExposition, type PrometheusSample } from "./prometheus.js";

export const REQUIRED_CHAT_METRICS = [
  "http_requests_total",
  "vllm:num_requests_running",
  "vllm:num_requests_waiting",
  "vllm:num_preemptions_total",
  "vllm:prompt_tokens_total",
  "vllm:generation_tokens_total",
  "vllm:request_success_total",
  "vllm:e2e_request_latency_seconds_count",
] as const;

export interface ChatMetricSummary {
  discoveredMetricNames: string[];
  chatHttpRequests: number;
  successfulRequests: number;
  stopRequests: number;
  lengthRequests: number;
  erroredRequests: number;
  promptTokens: number;
  generationTokens: number;
  latencyObservations: number;
  preemptions: number;
  runningRequests: number;
  waitingRequests: number;
}

export interface ChatMetricDelta {
  chatHttpRequests: number;
  successfulRequests: number;
  stopRequests: number;
  lengthRequests: number;
  erroredRequests: number;
  promptTokens: number;
  generationTokens: number;
  latencyObservations: number;
  preemptions: number;
}

export function summarizeChatMetrics(source: string, modelId: string): ChatMetricSummary {
  const parsed = parsePrometheusExposition(source);
  const missing = REQUIRED_CHAT_METRICS.filter((name) => !parsed.metricNames.includes(name));
  if (missing.length > 0) throw new Error(`Chat service is missing required metrics: ${missing.join(", ")}`);
  const model = (sample: PrometheusSample) => sample.labels.model_name === modelId;
  const finished = (reason: string) => (sample: PrometheusSample) => model(sample) && sample.labels.finished_reason === reason;
  const route = (sample: PrometheusSample) =>
    sample.labels.handler === "/v1/chat/completions" && sample.labels.method === "POST" && sample.labels.status === "2xx";
  const stopRequests = sumOrZero(parsed.samples, "vllm:request_success_total", finished("stop"));
  const lengthRequests = sumOrZero(parsed.samples, "vllm:request_success_total", finished("length"));
  const erroredRequests = sumOrZero(parsed.samples, "vllm:request_success_total", finished("error"));
  return {
    discoveredMetricNames: parsed.metricNames,
    chatHttpRequests: sumOrZero(parsed.samples, "http_requests_total", route),
    successfulRequests: sumOrZero(parsed.samples, "vllm:request_success_total", model) - erroredRequests,
    stopRequests,
    lengthRequests,
    erroredRequests,
    promptTokens: requiredSum(parsed.samples, "vllm:prompt_tokens_total", model),
    generationTokens: requiredSum(parsed.samples, "vllm:generation_tokens_total", model),
    latencyObservations: requiredSum(parsed.samples, "vllm:e2e_request_latency_seconds_count", model),
    preemptions: requiredSum(parsed.samples, "vllm:num_preemptions_total", model),
    runningRequests: requiredSum(parsed.samples, "vllm:num_requests_running", model),
    waitingRequests: requiredSum(parsed.samples, "vllm:num_requests_waiting", model),
  };
}

export function validateChatMetricDelta(
  before: ChatMetricSummary,
  after: ChatMetricSummary,
  expectedRequests: number,
): ChatMetricDelta {
  const delta = {
    chatHttpRequests: counterDelta("chat HTTP requests", before.chatHttpRequests, after.chatHttpRequests),
    successfulRequests: counterDelta("successful requests", before.successfulRequests, after.successfulRequests),
    stopRequests: counterDelta("stop-finished requests", before.stopRequests, after.stopRequests),
    lengthRequests: counterDelta("length-finished requests", before.lengthRequests, after.lengthRequests),
    erroredRequests: counterDelta("errored requests", before.erroredRequests, after.erroredRequests),
    promptTokens: counterDelta("prompt tokens", before.promptTokens, after.promptTokens),
    generationTokens: counterDelta("generation tokens", before.generationTokens, after.generationTokens),
    latencyObservations: counterDelta("latency observations", before.latencyObservations, after.latencyObservations),
    preemptions: counterDelta("preemptions", before.preemptions, after.preemptions),
  };
  if (delta.chatHttpRequests < expectedRequests) throw new Error(`Chat HTTP counter advanced by ${delta.chatHttpRequests}; expected at least ${expectedRequests}`);
  if (delta.successfulRequests < expectedRequests) throw new Error(`Successful-request counter advanced by ${delta.successfulRequests}; expected at least ${expectedRequests}`);
  if (delta.latencyObservations < expectedRequests) throw new Error(`Latency histogram advanced by ${delta.latencyObservations}; expected at least ${expectedRequests}`);
  if (delta.promptTokens <= 0 || delta.generationTokens <= 0) throw new Error("Chat token counters did not advance");
  if (delta.erroredRequests !== 0) throw new Error(`vLLM reported ${delta.erroredRequests} errored requests`);
  if (delta.preemptions !== 0) throw new Error(`vLLM reported ${delta.preemptions} preemptions`);
  return delta;
}

function requiredSum(samples: PrometheusSample[], name: string, predicate: (sample: PrometheusSample) => boolean): number {
  const values = samples.filter((sample) => sample.name === name && predicate(sample)).map((sample) => sample.value);
  if (values.length === 0) throw new Error(`Required metric ${name} has no matching labeled series`);
  if (!values.every(Number.isFinite)) throw new Error(`Required metric ${name} contains a non-finite value`);
  return values.reduce((sum, value) => sum + value, 0);
}

function sumOrZero(samples: PrometheusSample[], name: string, predicate: (sample: PrometheusSample) => boolean): number {
  const values = samples.filter((sample) => sample.name === name && predicate(sample)).map((sample) => sample.value);
  if (!values.every(Number.isFinite)) throw new Error(`Required metric ${name} contains a non-finite value`);
  return values.reduce((sum, value) => sum + value, 0);
}

function counterDelta(name: string, before: number, after: number): number {
  const value = after - before;
  if (!Number.isFinite(value) || value < 0) throw new Error(`The ${name} counter reset or became invalid during the gate`);
  return value;
}
