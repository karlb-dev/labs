import { summarizeChatMetrics, validateChatMetricDelta } from "../src/chat-metrics.js";

function fixture(model: string, requests: number, stop: number, length: number, prompt: number, generation: number): string {
  return [
    "# TYPE http_requests_total counter",
    `http_requests_total{handler="/v1/chat/completions",method="POST",status="2xx"} ${requests}`,
    `vllm:num_requests_running{model_name="${model}"} 0`,
    `vllm:num_requests_waiting{model_name="${model}"} 0`,
    `vllm:num_preemptions_total{model_name="${model}"} 0`,
    `vllm:prompt_tokens_total{model_name="${model}"} ${prompt}`,
    `vllm:generation_tokens_total{model_name="${model}"} ${generation}`,
    `vllm:request_success_total{model_name="${model}",finished_reason="stop"} ${stop}`,
    `vllm:request_success_total{model_name="${model}",finished_reason="length"} ${length}`,
    `vllm:request_success_total{model_name="${model}",finished_reason="error"} 0`,
    `vllm:e2e_request_latency_seconds_count{model_name="${model}"} ${stop + length}`,
  ].join("\n");
}

describe("chat service metric gate", () => {
  it("reconciles exact route, token, finish, latency, error, and preemption deltas", () => {
    const before = summarizeChatMetrics(fixture("served", 2, 2, 0, 100, 20), "served");
    const after = summarizeChatMetrics(fixture("served", 8, 8, 0, 700, 140), "served");
    expect(validateChatMetricDelta(before, after, 6)).toEqual({
      chatHttpRequests: 6,
      successfulRequests: 6,
      stopRequests: 6,
      lengthRequests: 0,
      erroredRequests: 0,
      promptTokens: 600,
      generationTokens: 120,
      latencyObservations: 6,
      preemptions: 0,
    });
  });
});
