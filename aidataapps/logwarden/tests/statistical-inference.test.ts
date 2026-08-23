import {
  holmAdjustedPValues,
  pairedGroupedBootstrap,
  pairedGroupedSignSwap,
  stableStatisticsSeed,
  withinStratumGroupLabelPermutation,
} from "../src/statistics.js";

const rows = [
  { groupId: "g1", stratum: "f1|test", left: 1, right: 0 },
  { groupId: "g1", stratum: "f1|test", left: 1, right: 1 },
  { groupId: "g2", stratum: "f1|test", left: 0, right: 0 },
  { groupId: "g2", stratum: "f1|test", left: 1, right: 0 },
  { groupId: "g3", stratum: "f2|test", left: 1, right: 0 },
  { groupId: "g3", stratum: "f2|test", left: 0, right: 0 },
];

describe("paired grouped statistical inference", () => {
  it("is byte-deterministic under a frozen seed and resamples whole groups", () => {
    const first = pairedGroupedBootstrap(rows, { seed: 590003, replicates: 1000, confidenceLevel: 0.95 });
    const second = pairedGroupedBootstrap(rows, { seed: 590003, replicates: 1000, confidenceLevel: 0.95 });
    expect(first).toEqual(second);
    expect(first.observedDifference).toBe(0.5);
    expect(first.groupCount).toBe(3);
    expect(first.replicatesValues).toHaveLength(1000);
    expect(first.ciLow).toBeGreaterThanOrEqual(0);
    expect(first.ciHigh).toBeLessThanOrEqual(1);
  });

  it("uses finite-sample corrected paired group swaps for a null", () => {
    const result = pairedGroupedSignSwap(rows, { seed: 20260823, replicates: 1000 });
    expect(result.observedDifference).toBe(0.5);
    expect(result.nullValues).toHaveLength(1000);
    expect(result.pValueTwoSided).toBeGreaterThan(0);
    expect(result.pValueTwoSided).toBeLessThanOrEqual(1);
  });

  it("applies monotone Holm adjustment and stable identity-specific seeds", () => {
    expect(holmAdjustedPValues([
      { id: "a", pValue: 0.01 }, { id: "b", pValue: 0.04 }, { id: "c", pValue: 0.03 },
    ])).toEqual({ a: 0.03, c: 0.06, b: 0.06 });
    expect(stableStatisticsSeed(20260823, "A")).toBe(stableStatisticsSeed(20260823, "A"));
    expect(stableStatisticsSeed(20260823, "A")).not.toBe(stableStatisticsSeed(20260823, "B"));
  });

  it("permutes complete group labels only within frozen strata", () => {
    const labels = [
      { rowId: "a1", groupId: "a", stratum: "f1|test", label: "x" },
      { rowId: "a2", groupId: "a", stratum: "f1|test", label: "x" },
      { rowId: "b1", groupId: "b", stratum: "f1|test", label: "y" },
      { rowId: "b2", groupId: "b", stratum: "f1|test", label: "y" },
      { rowId: "c1", groupId: "c", stratum: "f2|test", label: "z" },
      { rowId: "c2", groupId: "c", stratum: "f2|test", label: "z" },
    ];
    const result = withinStratumGroupLabelPermutation(labels, (permuted) =>
      [...permuted].filter(([rowId, label]) => rowId.startsWith("a") && label === "x").length / 2,
    { seed: 20260823, replicates: 1000, observedValue: 1 });
    expect(result.algorithm).toBe("within-stratum-group-label-permutation-v1");
    expect(result.stratumCount).toBe(2);
    expect(result.nullValues).toContain(0);
    expect(result.nullValues).toContain(1);
  });
});
