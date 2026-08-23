import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { callOpenAiCompatibleModel } from "../src/model-gateway.js";
import { createComponentTelemetryJournal, readAndValidateTelemetryJournal } from "../src/telemetry.js";

const decision = {
  kind: "decision",
  incidentClass: "blocking",
  severity: "medium",
  action: "open_work_item",
  actionArguments: {},
  citedChunkIds: [],
  confidence: 0.7,
  abstain: false,
  correlationKey: "block-1",
  summary: "Blocking requires review.",
  rationale: "The bounded snapshot shows an active blocker.",
};

describe("instrumented OpenAI-compatible gateway", () => {
  let baseUrl = "";
  let close: () => Promise<void>;
  const routeCounts = new Map<string, number>();

  beforeAll(async () => {
    const server = createServer((request, response) => route(request, response, routeCounts));
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    if (address === null || typeof address === "string") throw new Error("Fake gateway did not bind TCP");
    baseUrl = `http://127.0.0.1:${address.port}`;
    close = () => new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  });

  afterAll(async () => close());

  for (const testCase of [
    { route: "success", status: "success", terminal: "decision_received", error: null },
    { route: "invalid-json", status: "failed", terminal: "contract_rejected", error: "invalid_json" },
    { route: "empty", status: "failed", terminal: "contract_rejected", error: "empty_output" },
    { route: "schema", status: "failed", terminal: "contract_rejected", error: "contract_schema" },
    { route: "http-error", status: "failed", terminal: "retryable_failure", error: "http_error" },
    { route: "timeout", status: "failed", terminal: "model_timeout", error: "timeout" },
    { route: "truncated", status: "failed", terminal: "contract_rejected", error: "output_limit" },
  ] as const) {
    it(`durably closes the ${testCase.route} route`, async () => {
      const fixture = await fixtureJournal(testCase.route);
      const result = await callOpenAiCompatibleModel({
        endpoint: `${baseUrl}/${testCase.route}`,
        model: "fake",
        messages: [{ role: "user", content: "fixture" }],
        decode: { temperature: 0 },
        journal: fixture.journal,
        runDirectory: fixture.directory,
        timeoutMs: 40,
        maxResponseBytes: 512,
      });
      expect(result.status).toBe(testCase.status);
      expect(result.terminalState).toBe(testCase.terminal);
      expect(result.errorClass).toBe(testCase.error);
      const lines = await readAndValidateTelemetryJournal(fixture.path, fixture.runId);
      const rootEnds = lines.filter(({ record }) => record.eventKind === "span_end" && record.spanId === result.rootSpanId);
      expect(rootEnds).toHaveLength(1);
      expect(lines.some(({ record }) => record.name === `state.${testCase.terminal}`)).toBe(true);
      expect(await readFile(result.attempts[0]!.requestPath)).toEqual(result.attempts[0]!.requestBody);
      expect(await readFile(result.attempts[0]!.responsePath)).toEqual(result.attempts[0]!.responseBody);
    });
  }

  it("records a retry attempt separately before succeeding", async () => {
    routeCounts.delete("retry");
    const fixture = await fixtureJournal("retry");
    const result = await callOpenAiCompatibleModel({
      endpoint: `${baseUrl}/retry`, model: "fake", messages: [{ role: "user", content: "fixture" }],
      decode: { temperature: 0 }, journal: fixture.journal, runDirectory: fixture.directory,
      timeoutMs: 100, maxRetries: 1, retryBackoffMs: 1,
    });
    expect(result.status).toBe("success");
    expect(result.attempts.map((attempt) => attempt.status)).toEqual(["retry", "success"]);
    expect(new Set(result.attempts.map((attempt) => attempt.clientRequestId)).size).toBe(2);
  });
});

async function fixtureJournal(name: string) {
  const directory = await mkdtemp(join(tmpdir(), `logwarden-gateway-${name}-`));
  const runId = `run-${name}`;
  return { directory, runId, ...await createComponentTelemetryJournal(directory, runId, "gateway-test") };
}

function route(request: IncomingMessage, response: ServerResponse, counts: Map<string, number>): void {
  const name = request.url?.slice(1) ?? "";
  counts.set(name, (counts.get(name) ?? 0) + 1);
  request.resume();
  if (name === "timeout") {
    setTimeout(() => openAi(response, decision), 150);
    return;
  }
  if (name === "http-error" || (name === "retry" && counts.get(name) === 1)) {
    response.writeHead(503, { "content-type": "application/json" });
    response.end('{"error":"retry later"}');
    return;
  }
  if (name === "invalid-json") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end("not-json");
    return;
  }
  if (name === "empty") return openAi(response, "");
  if (name === "schema") return openAi(response, { ...decision, incidentClass: "not-a-class" });
  if (name === "truncated") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({ choices: [{ message: { content: "x".repeat(2048) } }] }));
    return;
  }
  openAi(response, decision);
}

function openAi(response: ServerResponse, value: unknown): void {
  const content = typeof value === "string" ? value : JSON.stringify(value);
  response.writeHead(200, { "content-type": "application/json", "x-request-id": "fake-service-request" });
  response.end(JSON.stringify({
    id: "fake-completion",
    choices: [{ finish_reason: "stop", message: { role: "assistant", content } }],
    usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
  }));
}
