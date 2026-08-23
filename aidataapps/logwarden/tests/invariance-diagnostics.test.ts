import { describe, expect, it } from "vitest";
import { normalizeModelRequest, normalizeModelResponse } from "../src/invariance-diagnostics.js";

describe("batching invariance diagnostics", () => {
  it("removes response-envelope identities without hiding generated-output drift", () => {
    const left = JSON.stringify({ id: "one", created: 1, choices: [{ message: { content: "same" } }], usage: { total_tokens: 3 } });
    const right = JSON.stringify({ id: "two", created: 2, choices: [{ message: { content: "same" } }], usage: { total_tokens: 3 } });
    expect(normalizeModelResponse(left)).toEqual(normalizeModelResponse(right));
    expect(normalizeModelResponse(left)).not.toEqual(normalizeModelResponse(right.replace("same", "different")));
  });

  it("normalizes only execution-local identities in structured tool messages", () => {
    const request = (callId: string, retrievalRunId: number, status: string) => JSON.stringify({
      model: "m", messages: [{ role: "user", content: JSON.stringify({
        kind: "tool_result", callId, result: { retrievalRunId, rows: [{ status }] }, remainingBudget: { modelTurns: 2 },
      }) }],
    });
    expect(normalizeModelRequest(request("1:2:1", 10, "ok")))
      .toEqual(normalizeModelRequest(request("8:9:1", 99, "ok")));
    expect(normalizeModelRequest(request("1:2:1", 10, "ok")))
      .not.toEqual(normalizeModelRequest(request("8:9:1", 99, "changed")));
  });
});
