import { afterEach, describe, expect, it, vi } from "vitest";
import { VllmGateway } from "../src/inference.js";
import type { ModelProfile } from "../src/models.js";

const profile: ModelProfile = {
  key: "gemma-test",
  description: "test",
  family: "gemma",
  modelId: "test/model",
  revision: "1234567",
  vllmImage: "test/image",
  maxModelLen: 1024,
  gpuMemoryUtilization: 0.5,
  systemRole: false,
  chatTemplateKwargs: {},
  vllmArgs: [],
};

afterEach(() => vi.unstubAllGlobals());

describe("VllmGateway", () => {
  it("folds instructions into the user turn for no-system-role models", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ choices: [{ message: { content: '{"kind":"answer"}' } }] }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const gateway = new VllmGateway({
      chatBaseUrl: "http://chat/v1",
      embeddingBaseUrl: "http://embed/v1",
      embeddingModel: "embed",
      embeddingDimensions: 2,
      model: profile,
    });

    await gateway.complete("system rules", "user question");

    const request = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const body = JSON.parse(String(request.body)) as {
      messages: Array<{ role: string; content: string }>;
    };
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0]?.role).toBe("user");
    expect(body.messages[0]?.content).toContain("system rules");
    expect(body.messages[0]?.content).toContain("user question");
  });

  it("checks embedding dimensions", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ data: [{ index: 0, embedding: [0.1] }] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    const gateway = new VllmGateway({
      chatBaseUrl: "http://chat/v1",
      embeddingBaseUrl: "http://embed/v1",
      embeddingModel: "embed",
      embeddingDimensions: 2,
      model: profile,
    });

    await expect(gateway.embed(["hello"])).rejects.toThrow("expected 2");
  });
});
