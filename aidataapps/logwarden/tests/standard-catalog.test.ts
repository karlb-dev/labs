import { generateStandardScenarioCatalog } from "../src/standard-catalog.js";

describe("standard scenario catalog", () => {
  const catalog = generateStandardScenarioCatalog();
  const variants = catalog.scenarios.flatMap((scenario) => scenario.variants?.map((variant) => ({ scenario, variant })) ?? []);

  it("has the governed 60-template and 600-episode role allocation", () => {
    expect(catalog.scenarios).toHaveLength(60);
    expect(variants).toHaveLength(600);
    const roleCounts = new Map<string, number>();
    for (const { variant } of variants) roleCounts.set(variant.splitRole, (roleCounts.get(variant.splitRole) ?? 0) + 1);
    expect(Object.fromEntries([...roleCounts.entries()].sort())).toEqual({
      calibration: 60,
      dev: 60,
      test_id: 300,
      test_unknown: 60,
      test_variant_holdout: 120,
    });
  });

  it("keeps template groups wholly within one split role", () => {
    for (const scenario of catalog.scenarios)
      expect(new Set(scenario.variants?.map((variant) => variant.splitRole)).size).toBe(1);
    expect(new Set(catalog.scenarios.map((scenario) => scenario.groupId)).size).toBe(60);
  });

  it("covers every family and regime with at least 40 held-out test episodes per family", () => {
    expect(new Set(catalog.scenarios.map((scenario) => scenario.family)).size).toBe(10);
    expect([...new Set(catalog.scenarios.map((scenario) => scenario.regime))].sort()).toEqual(["C", "K", "M", "N", "U"]);
    const testCounts = new Map<string, number>();
    for (const { scenario, variant } of variants.filter(({ variant }) => ["test_id", "test_variant_holdout"].includes(variant.splitRole)))
      testCounts.set(scenario.family, (testCounts.get(scenario.family) ?? 0) + 1);
    expect([...testCounts.values()].every((count) => count >= 40)).toBe(true);
  });

  it("uses only bounded ground-truth tool policies", () => {
    for (const scenario of catalog.scenarios) {
      expect(scenario.groundTruth?.requiredTools.length ?? 0).toBeLessThanOrEqual(4);
      const tools = [
        ...(scenario.groundTruth?.requiredTools ?? []),
        ...(scenario.groundTruth?.optionalTools ?? []),
        ...(scenario.groundTruth?.forbiddenTools ?? []),
      ];
      expect(new Set(tools).size).toBe(tools.length);
    }
  });

  it("anchors successful workload evidence to the RPC carrying the correlation marker", () => {
    const successful = catalog.scenarios.filter((scenario) =>
      scenario.injector === "query_pressure" || scenario.injector === "benign_noise");
    expect(successful.length).toBeGreaterThan(0);
    for (const scenario of successful)
      expect(scenario.expectedEvidence).toEqual([
        expect.objectContaining({ source: "xe", event: "rpc_completed" }),
      ]);
  });
});
