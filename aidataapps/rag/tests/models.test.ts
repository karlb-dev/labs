import { describe, expect, it } from "vitest";
import { loadModelRegistry, resolveModelProfile } from "../src/models.js";

describe("model registry", () => {
  it("contains the complete large-model comparison matrix", () => {
    const registry = loadModelRegistry();
    expect(registry.benchmarkProfiles).toEqual([
      "muse-glimmer-30b",
      "gemma-4-31b",
      "olmo-3.1-32b-instruct",
      "qwen-3.8-27b",
    ]);
    for (const key of registry.benchmarkProfiles) {
      expect(resolveModelProfile(key, registry).revision).toMatch(/^[a-f0-9]{40}$/);
    }
  });

  it("retains the repo-pinned Qwen 3.6 profile for replay", () => {
    expect(resolveModelProfile("qwen-3.6-27b-pinned").modelId).toBe(
      "Qwen/Qwen3.6-27B",
    );
  });
});
