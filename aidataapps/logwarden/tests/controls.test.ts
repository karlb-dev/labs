import { loadTier1ControlPolicy, maskErrorNumbersAndSignatures, parseInferenceControlId } from "../src/controls.js";

describe("frozen Tier 1 controls", () => {
  it("loads every inference control and rejects undeclared controls", () => {
    const policy = loadTier1ControlPolicy();
    expect(Object.keys(policy.controls)).toHaveLength(4);
    expect(parseInferenceControlId("shuffled-runbooks-v1")).toBe("shuffled-runbooks-v1");
    expect(() => parseInferenceControlId("invented-control")).toThrow();
  });

  it("recursively masks numeric fields, numeric text, exact signatures, and tool results without mutation", () => {
    const input = {
      sourceEvents: [{ errorNumber: 1205, message: "Error 1205: deadlock victim selected" }],
      tool: { rows: [{ error_number: 9002, detail: "Transaction log for database LW_1 is full (9002)." }] },
    };
    const result = maskErrorNumbersAndSignatures(input);
    expect(input.sourceEvents[0]!.errorNumber).toBe(1205);
    expect(result.replacementCount).toBeGreaterThanOrEqual(6);
    expect(result.value).toEqual({
      sourceEvents: [{ errorNumber: null, message: "Error [ERROR_NUMBER]: [EXACT_SIGNATURE] selected" }],
      tool: { rows: [{ error_number: null, detail: "[EXACT_SIGNATURE] ([ERROR_NUMBER])." }] },
    });
    expect(result.transformedSha256).not.toBe(result.originalSha256);
  });
});
