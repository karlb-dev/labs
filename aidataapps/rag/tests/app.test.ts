import { describe, expect, it, vi } from "vitest";
import { buildApp } from "../src/app.js";
import type { QueryAgent } from "../src/types.js";

function fakeAgent(): QueryAgent {
  return {
    query: vi.fn().mockResolvedValue({
      kind: "answer",
      answer: "Grounded answer",
      citations: [],
      modelProfile: "test",
    }),
    ready: vi.fn().mockResolvedValue(undefined),
  };
}

describe("HTTP API", () => {
  it("validates query requests", async () => {
    const app = buildApp({ agent: fakeAgent() });
    const response = await app.inject({
      method: "POST",
      url: "/api/query",
      payload: { query: "x" },
    });
    expect(response.statusCode).toBe(400);
    expect(response.json()).toMatchObject({ error: "invalid_request" });
    await app.close();
  });

  it("defaults actions to disabled", async () => {
    const agent = fakeAgent();
    const app = buildApp({ agent });
    const response = await app.inject({
      method: "POST",
      url: "/api/query",
      payload: { query: "How do I inspect J4?" },
    });
    expect(response.statusCode).toBe(200);
    expect(agent.query).toHaveBeenCalledWith({
      query: "How do I inspect J4?",
      allowActions: false,
    });
    await app.close();
  });

  it("reports dependency readiness failures", async () => {
    const agent = fakeAgent();
    vi.mocked(agent.ready).mockRejectedValue(new Error("SQL unavailable"));
    const app = buildApp({ agent });
    const response = await app.inject({ method: "GET", url: "/ready" });
    expect(response.statusCode).toBe(503);
    expect(response.json()).toMatchObject({
      status: "not_ready",
      error: "SQL unavailable",
    });
    await app.close();
  });
});
