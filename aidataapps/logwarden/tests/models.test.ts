import { loadModelRegistry, resolveModelProfile } from "../src/models.js";

describe("frozen model registry", () => {
  it("contains four distinct digest-resolved targets", () => {
    const registry = loadModelRegistry();
    expect(new Set(registry.targetProfiles).size).toBe(4);
    for (const key of registry.targetProfiles) {
      const profile = resolveModelProfile(key, registry);
      expect(profile.revision).toMatch(/^[0-9a-f]{40}$/);
      expect(profile.vllmImage).toMatch(/@sha256:[0-9a-f]{64}$/);
    }
  });
});
