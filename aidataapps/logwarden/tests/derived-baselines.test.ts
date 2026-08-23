import { loadDerivedBaselineConfig, majorityPrediction, oraclePrediction, retrievalOnlyPrediction, selectRouterCandidate, verifyMajorityFit } from "../src/derived-baselines.js";

describe("frozen derived baselines", () => {
  it("recomputes B0 exclusively from the hashed dev split", () => {
    expect(() => verifyMajorityFit()).not.toThrow();
    expect(majorityPrediction()).toEqual({
      incidentClass: "benign_noise", severity: "info", action: "no_action", confidence: 0.5, abstain: false,
    });
  });

  it("maps top runbook metadata and fails safely when retrieval is empty", () => {
    expect(retrievalOnlyPrediction({ runbookId: "TSG-DLK-01", incidentClass: "deadlock", severityFloor: "medium" })).toEqual({
      incidentClass: "deadlock", severity: "medium", action: "run_tsg", confidence: 0.5, abstain: false,
    });
    expect(retrievalOnlyPrediction(null)).toEqual({
      incidentClass: "unknown_ambiguous", severity: "low", action: "escalate_to_human", confidence: 0, abstain: true,
    });
  });

  it("keeps the oracle evaluator-only and routes only unresolved B1 rows", () => {
    expect(loadDerivedBaselineConfig().oracle.evaluatorOnly).toBe(true);
    expect(oraclePrediction({ expectedClass: "integrity_signal", expectedSeverity: "critical", expectedAction: "escalate_to_human", shouldAbstain: false })).toEqual({
      incidentClass: "integrity_signal", severity: "critical", action: "escalate_to_human", confidence: 1, abstain: false,
    });
    const b1 = { predictionId: 1, outcome: "decision" as const, predictionJson: { resolved: true } };
    const tools = { predictionId: 2, outcome: "failure" as const, predictionJson: {} };
    expect(selectRouterCandidate(b1, tools)).toEqual({ sourceArmId: "B1-rules-v1", candidate: b1 });
    expect(selectRouterCandidate({ ...b1, predictionJson: { resolved: false } }, tools)).toEqual({ sourceArmId: "A-tools", candidate: tools });
  });
});
