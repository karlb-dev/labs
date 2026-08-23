import { describe, expect, it } from "vitest";
import { scorePrediction, severityCost, toolScoreRows, type ScoreTruth } from "../src/scoring.js";

const truth: ScoreTruth = {
  expectedClass: "blocking", expectedSeverity: "high", expectedAction: "run_tsg",
  acceptableActions: ["run_tsg", "open_work_item"], shouldAbstain: false,
  expectedRunbooks: ["TSG-BLK-01"], requiredTools: ["get_blocking_snapshot", "runbook_search"],
  optionalTools: ["get_recent_incident_counts"], forbiddenTools: ["get_log_space"],
  costWeights: { miss: 10, falseAlarm: 2, unnecessaryTool: 0.25 },
};

describe("decision scoring", () => {
  it("requires the whole agent path for end-to-end credit", () => {
    const score = scorePrediction({
      truth,
      prediction: { outcome: "decision", predictedClass: "blocking", predictedSeverity: "high", predictedAction: "open_work_item", abstained: false, contractValid: true, policyValid: true },
      armId: "A-tools",
      tools: [
        { toolName: "get_blocking_snapshot", status: "success", policyStatus: "allow", snapshotMiss: false },
        { toolName: "runbook_search", status: "success", policyStatus: "allow", snapshotMiss: false },
      ],
      citedRunbooks: ["TSG-BLK-01"],
      returnedRunbooks: ["TSG-BLK-01"],
    });
    expect(score.endToEndSuccess).toBe(true);
    expect(score.actionCorrect).toBe(true);
    expect(score.costWeightedLoss).toBe(0);
  });

  it("counts a correct guess without required diagnostics as incomplete", () => {
    const score = scorePrediction({
      truth,
      prediction: { outcome: "decision", predictedClass: "blocking", predictedSeverity: "high", predictedAction: "run_tsg", abstained: false, contractValid: true, policyValid: true },
      armId: "A-direct", tools: [], citedRunbooks: [], returnedRunbooks: [],
    });
    expect(score.actionCorrect).toBe(true);
    expect(score.requiredToolsSatisfied).toBe(false);
    expect(score.endToEndSuccess).toBe(false);
  });

  it("uses asymmetric severity costs and makes failures costly", () => {
    expect(severityCost("high", "medium")).toBe(1);
    expect(severityCost("medium", "high")).toBe(0.5);
    expect(severityCost("critical", null)).toBe(10);
    const score = scorePrediction({
      truth,
      prediction: { outcome: "failure", predictedClass: null, predictedSeverity: null, predictedAction: null, abstained: null, contractValid: false, policyValid: false },
      armId: "A-tools", tools: [], citedRunbooks: [], returnedRunbooks: [],
    });
    expect(score.endToEndSuccess).toBe(false);
    expect(score.costWeightedLoss).toBeGreaterThanOrEqual(20);
  });

  it("scores required, forbidden, and observed unspecified tools", () => {
    const rows = toolScoreRows(truth, [
      { toolName: "get_blocking_snapshot", status: "success", policyStatus: "allow", snapshotMiss: false },
      { toolName: "get_log_space", status: "success", policyStatus: "allow", snapshotMiss: false },
      { toolName: "invented", status: "failed", policyStatus: "deny", snapshotMiss: false },
    ]);
    expect(rows.find((row) => row.toolName === "get_blocking_snapshot")?.correctUse).toBe(true);
    expect(rows.find((row) => row.toolName === "get_log_space")?.correctUse).toBe(false);
    expect(rows.find((row) => row.toolName === "invented")?.expectation).toBe("unspecified");
  });
});
