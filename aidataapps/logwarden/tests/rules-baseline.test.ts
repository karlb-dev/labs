import { applyRulesBaseline, loadRulesBaseline } from "../src/rules-baseline.js";

const rules = loadRulesBaseline();

describe("strong deterministic B1 baseline", () => {
  it("resolves documented engine signatures", () => {
    expect(applyRulesBaseline(packet({ eventName: "error_reported", errorNumber: 1205 }), rules))
      .toMatchObject({ resolved: true, matchedRuleId: "deadlock-victim-1205", prediction: { incidentClass: "deadlock", action: "run_tsg" } });
    expect(applyRulesBaseline(packet({ eventName: "error_reported", errorNumber: 2627 }), rules))
      .toMatchObject({ resolved: true, matchedRuleId: "schema-data-high", prediction: { severity: "high" } });
  });

  it("separates recognizable lab integrity and ambiguity messages", () => {
    expect(applyRulesBaseline(packet({ errorNumber: 50000, severity: 16, state: 42, message: "Consistency check requires review" }), rules))
      .toMatchObject({ resolved: true, matchedRuleId: "integrity-lab-message", prediction: { incidentClass: "integrity_signal", abstain: false } });
    expect(applyRulesBaseline(packet({ errorNumber: 50000, severity: 11, state: 53, message: "Signals disagree" }), rules))
      .toMatchObject({ resolved: true, matchedRuleId: "ambiguous-lab-message", prediction: { incidentClass: "unknown_ambiguous", abstain: true } });
  });

  it("treats a successful completion without an engine error as no action", () => {
    expect(applyRulesBaseline(packet({ eventName: "rpc_completed", message: "completed" }), rules))
      .toMatchObject({ resolved: true, matchedRuleId: "successful-completion-no-engine-error", prediction: { incidentClass: "benign_noise", action: "no_action" } });
  });

  it("routes unresolved evidence to explicit abstention", () => {
    expect(applyRulesBaseline(packet({ eventName: "error_reported", errorNumber: 50000, severity: 14, message: "unrecognized" }), rules))
      .toMatchObject({ resolved: false, matchedRuleId: null, prediction: { incidentClass: "unknown_ambiguous", action: "escalate_to_human", confidence: 0, abstain: true } });
  });
});

function packet(event: Partial<{ eventName: string; errorNumber: number | null; severity: number | null; state: number | null; message: string | null }>) {
  return {
    packetVersion: "incident-packet-v1",
    sourceEvents: [{ eventName: "error_reported", errorNumber: null, severity: null, state: null, message: null, ...event }],
    recentHistory: { sameFingerprint5m: 0, sameClass1h: 0, openRelatedIncidents: 0 },
  };
}
