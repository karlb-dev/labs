import { loadMacModelRegistry, resolveMacEmbeddingProfile, resolveMacProfile } from "../src/models-mac.js";

describe("mac serving registry", () => {
  it("loads and resolves the default and smoke profiles", () => {
    const registry = loadMacModelRegistry();
    const agent = resolveMacProfile(registry.defaultProfile, registry);
    const smoke = resolveMacProfile(registry.smokeProfile, registry);
    expect(agent.role).toBe("agent");
    expect(smoke.role).toBe("plumbing");
  });

  it("keeps mac profiles distinct from frozen campaign identities", () => {
    const registry = loadMacModelRegistry();
    for (const [key, profile] of Object.entries(registry.profiles)) {
      expect(key.endsWith("-foundry") || key.endsWith("-mlx")).toBe(true);
      expect(profile.foundryAlias.length).toBeGreaterThan(0);
      if (key.endsWith("-mlx")) {
        expect(profile.baseUrl).toBeDefined();
        expect(profile.servedModelId).toBeDefined();
      }
    }
    expect(registry.comparability).toMatch(/never comparable/);
  });

  it("pins the embedding profile to the lab's embedding family", () => {
    const embedding = resolveMacEmbeddingProfile("qwen3-embedding-0.6b-foundry");
    expect(embedding.baseModelId).toBe("Qwen/Qwen3-Embedding-0.6B");
    expect(embedding.expectedDimensions).toBe(1024);
  });
});
