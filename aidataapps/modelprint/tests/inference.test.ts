import { afterEach, describe, expect, it, vi } from "vitest";
import { VllmGateway } from "../src/inference.js";
import { sliceAssistantLogprobs } from "../src/likelihood.js";
import { resolveModelProfile } from "../src/models.js";
import { decodeCell } from "../src/types.js";

afterEach(() => vi.unstubAllGlobals());

describe("likelihood span slicing", () => {
  it("counts the assistant span when its first token merges at the template junction", () => {
    const result = sliceAssistantLogprobs([1, 2, 3, 4], [
      { "1": { logprob: -0.1, decoded_token: "<assistant>" } },
      { "2": { logprob: -0.2, decoded_token: ":Hel" } },
      { "3": { logprob: -0.3, decoded_token: "lo" } },
      { "4": { logprob: -0.4, decoded_token: "<|im_end|>" } },
    ], "Hello");
    expect(result?.tokenCount).toBe(2);
    expect(result?.values).toEqual([-0.2, -0.3]);
    expect(result?.firstTokenIndex).toBe(1);
  });
});

describe("VllmGateway", () => {
  it("sends exactly one user message and retains raw/final/reasoning fields", async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const request = JSON.parse(String(init?.body)) as Record<string, unknown>;
      expect(request.messages).toEqual([{ role: "user", content: "test" }]);
      for (const key of ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty", "seed", "max_tokens", "n", "stop"]) {
        expect(request).toHaveProperty(key);
      }
      return new Response(JSON.stringify({ choices: [{ finish_reason: "stop", message: { content: "final", reasoning_content: "reason" } }], usage: { prompt_tokens: 3, completion_tokens: 2 } }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    const result = await new VllmGateway().completeMessages("http://localhost/v1", resolveModelProfile("qwen-smoke"), [{ role: "user", content: "test" }], decodeCell("det", 50));
    expect(result.finalText).toBe("final");
    expect(result.reasoningText).toBe("reason");
    expect(result.rawResponse).toHaveProperty("choices");
  });

  it("rejects a system-role campaign request before HTTP", async () => {
    await expect(new VllmGateway().completeMessages("http://localhost/v1", resolveModelProfile("qwen-smoke"), [
      { role: "system", content: "x" }, { role: "user", content: "y" },
    ], decodeCell("det", 50))).rejects.toThrow(/exactly one user message/);
  });
});
