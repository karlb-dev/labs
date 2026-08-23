import { parsePrometheusExposition, sumPrometheusMetric } from "../src/prometheus.js";

describe("Prometheus exposition parser", () => {
  it("retains labels, histogram shapes, and metric discovery", () => {
    const parsed = parsePrometheusExposition([
      "# TYPE vllm:num_requests_running gauge",
      "vllm:num_requests_running{model_name=\"qwen\"} 2",
      "request_latency_seconds_bucket{le=\"0.5\"} 4",
      "request_latency_seconds_sum 1.25",
      "request_latency_seconds_count 4",
    ].join("\n"));
    expect(parsed.metricNames).toContain("vllm:num_requests_running");
    expect(parsed.samples[1]!.type).toBe("histogram_bucket");
    expect(sumPrometheusMetric(parsed.samples, ["vllm:num_requests_running"])).toBe(2);
  });
});
