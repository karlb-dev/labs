import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { summarizeEmbeddingMetrics, validateEmbeddingMetricDelta } from "../src/embedding-metrics.js";
import { callOpenAiCompatibleEmbeddings, parseEmbeddingResponse } from "../src/embeddings.js";
import { createComponentTelemetryJournal, readAndValidateTelemetryJournal } from "../src/telemetry.js";

describe("instrumented OpenAI-compatible embedding client", () => {
  let baseUrl = "";
  let close: () => Promise<void>;

  beforeAll(async () => {
    const server = createServer(route);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    if (address === null || typeof address === "string") throw new Error("Fake embedding server did not bind TCP");
    baseUrl = `http://127.0.0.1:${address.port}`;
    close = () => new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  });

  afterAll(async () => close());

  it("durably records and validates an ordered embedding batch", async () => {
    const fixture = await fixtureJournal("success");
    const result = await callOpenAiCompatibleEmbeddings({
      endpoint: `${baseUrl}/success`,
      model: "fake-embedding",
      inputs: ["alpha", "beta"],
      dimensions: 3,
      journal: fixture.journal,
      runDirectory: fixture.directory,
      timeoutMs: 100,
    });
    expect(result.status).toBe("success");
    expect(result.vectors).toEqual([[1, 0, 0], [0, 1, 0]]);
    expect(result.evidence.vectorDimensions).toEqual([3, 3]);
    expect(result.evidence.usage).toEqual({ promptTokens: 4, totalTokens: 4 });
    expect((await readFile(result.evidence.requestPath)).length).toBeGreaterThan(0);
    expect((await readFile(result.evidence.responsePath)).length).toBeGreaterThan(0);
    const records = await readAndValidateTelemetryJournal(fixture.path, fixture.runId);
    expect(records.some(({ record }) => record.name === "embedding.request.durable")).toBe(true);
    expect(records.some(({ record }) => record.name === "embedding.response.durable")).toBe(true);
    expect(records.some(({ record }) => record.name === "state.embedding_validated")).toBe(true);
  });

  for (const testCase of [
    ["invalid-json", "invalid_json"],
    ["wrong-count", "count_mismatch"],
    ["wrong-index", "index_mismatch"],
    ["wrong-dimension", "dimension_mismatch"],
    ["http-error", "http_error"],
    ["empty", "empty_output"],
    ["timeout", "timeout"],
    ["truncated", "output_limit"],
  ] as const) {
    it(`fails closed on ${testCase[0]}`, async () => {
      const fixture = await fixtureJournal(testCase[0]);
      const result = await callOpenAiCompatibleEmbeddings({
        endpoint: `${baseUrl}/${testCase[0]}`,
        model: "fake-embedding",
        inputs: ["alpha", "beta"],
        dimensions: 3,
        journal: fixture.journal,
        runDirectory: fixture.directory,
        timeoutMs: 30,
        maxResponseBytes: testCase[0] === "truncated" ? 64 : 4096,
      });
      expect(result.status).toBe("failed");
      expect(result.evidence.errorClass).toBe(testCase[1]);
      expect(result.vectors).toBeNull();
      expect(await readFile(result.evidence.responsePath)).toBeInstanceOf(Buffer);
      const records = await readAndValidateTelemetryJournal(fixture.path, fixture.runId);
      expect(records.some(({ record }) => record.name === "state.embedding_rejected")).toBe(true);
    });
  }

  it("rejects invalid usage and non-finite vector payloads", () => {
    expect(() => parseEmbeddingResponse(JSON.stringify({
      data: [{ index: 0, embedding: [1, 0, 0] }],
      usage: { prompt_tokens: 2 },
    }), 1, 3)).toThrow(/usage counters/);
    expect(() => parseEmbeddingResponse(JSON.stringify({
      data: [{ index: 0, embedding: [1, "NaN", 0] }],
      usage: { prompt_tokens: 2, total_tokens: 2 },
    }), 1, 3)).toThrow(/non-finite/);
    expect(() => parseEmbeddingResponse(JSON.stringify({
      model: "unexpected-model",
      data: [{ index: 0, embedding: [1, 0, 0] }],
      usage: { prompt_tokens: 2, total_tokens: 2 },
    }), 1, 3, "expected-model")).toThrow(/response model/);
  });
});

describe("embedding metric gate", () => {
  it("requires request, token, success, and latency deltas without errors or preemptions", () => {
    const before = summarizeEmbeddingMetrics(metrics(10, 20, 100, 20, 0, 0), "fake-embedding");
    const after = summarizeEmbeddingMetrics(metrics(13, 25, 135, 25, 0, 0), "fake-embedding");
    expect(validateEmbeddingMetricDelta(before, after, 3, 5)).toMatchObject({
      embeddingHttpRequests: 3,
      successfulRequests: 5,
      promptTokens: 35,
      latencyObservations: 5,
    });
  });

  it("fails when required service telemetry is absent", () => {
    expect(() => summarizeEmbeddingMetrics("# no metrics\n", "fake-embedding")).toThrow(/missing required metrics/);
  });

  it("treats an as-yet-uncreated embedding route series as an exact zero baseline", () => {
    const baseline = summarizeEmbeddingMetrics(metrics(0, 0, 0, 0, 0, 0).replace(
      'http_requests_total{handler="/v1/embeddings",method="POST",status="2xx"} 0',
      'http_requests_total{handler="/v1/models",method="GET",status="2xx"} 1',
    ), "fake-embedding");
    expect(baseline.embeddingHttpRequests).toBe(0);
  });
});

async function fixtureJournal(name: string) {
  const directory = await mkdtemp(join(tmpdir(), `logwarden-embedding-${name}-`));
  const runId = `run-${name}`;
  return { directory, runId, ...await createComponentTelemetryJournal(directory, runId, "embedding-test") };
}

function route(request: IncomingMessage, response: ServerResponse): void {
  const name = request.url?.slice(1) ?? "";
  request.resume();
  if (name === "timeout") {
    setTimeout(() => success(response), 100);
    return;
  }
  if (name === "http-error") {
    response.writeHead(503, { "content-type": "application/json" });
    response.end('{"error":"retry later"}');
    return;
  }
  if (name === "invalid-json") return void response.end("not-json");
  if (name === "empty") return void response.end();
  if (name === "truncated") return void response.end("x".repeat(1024));
  if (name === "wrong-count") return json(response, envelope([[1, 0, 0]]));
  if (name === "wrong-index") return json(response, {
    ...envelope([[1, 0, 0], [0, 1, 0]]),
    data: [{ index: 1, embedding: [1, 0, 0] }, { index: 0, embedding: [0, 1, 0] }],
  });
  if (name === "wrong-dimension") return json(response, envelope([[1, 0], [0, 1]]));
  success(response);
}

function success(response: ServerResponse): void {
  json(response, envelope([[1, 0, 0], [0, 1, 0]]));
}

function envelope(vectors: number[][]) {
  return {
    object: "list",
    model: "fake-embedding",
    data: vectors.map((embedding, index) => ({ object: "embedding", index, embedding })),
    usage: { prompt_tokens: 4, total_tokens: 4 },
  };
}

function json(response: ServerResponse, body: unknown): void {
  response.writeHead(200, { "content-type": "application/json", "x-request-id": "fake-service-request" });
  response.end(JSON.stringify(body));
}

function metrics(http: number, success: number, tokens: number, latency: number, errors: number, preemptions: number): string {
  return [
    "# TYPE http_requests_total counter",
    `http_requests_total{handler="/v1/embeddings",method="POST",status="2xx"} ${http}`,
    `vllm:num_requests_running{model_name="fake-embedding"} 0`,
    `vllm:num_requests_waiting{model_name="fake-embedding"} 0`,
    `vllm:num_preemptions_total{model_name="fake-embedding"} ${preemptions}`,
    `vllm:prompt_tokens_total{model_name="fake-embedding"} ${tokens}`,
    `vllm:request_success_total{model_name="fake-embedding",finished_reason="stop"} ${success}`,
    `vllm:request_success_total{model_name="fake-embedding",finished_reason="error"} ${errors}`,
    `vllm:e2e_request_latency_seconds_count{model_name="fake-embedding"} ${latency}`,
  ].join("\n");
}
