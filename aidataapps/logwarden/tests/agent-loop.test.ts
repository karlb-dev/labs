import {
  allowedToolsForArm,
  boundToolResult,
  buildInitialAgentMessages,
  decisionPolicyFindings,
} from "../src/agent-loop.js";

const packet = {
  packetVersion: "incident-packet-v1",
  availableTools: ["runbook_search", "get_log_space", "not_registered"],
  recentHistory: { sameClass1h: 0 },
  sourceEvents: [],
};

const decision = {
  kind: "decision" as const,
  incidentClass: "blocking" as const,
  severity: "medium" as const,
  action: "open_work_item" as const,
  actionArguments: {},
  citedChunkIds: ["chunk-1"],
  confidence: 0.7,
  abstain: false,
  correlationKey: "episode:test",
  summary: "Blocking requires review.",
  rationale: "The bounded snapshot shows an active blocker.",
};

describe("bounded agent loop policy", () => {
  it("derives the exact per-arm tool surface from packet and registry", () => {
    expect(allowedToolsForArm("A-direct", packet)).toEqual([]);
    expect(allowedToolsForArm("A-rag", packet)).toEqual(["runbook_search"]);
    expect(allowedToolsForArm("A-tools", packet)).toEqual(["get_log_space", "runbook_search"]);
  });

  it("places the contract in one first user turn with no system message", () => {
    const messages = buildInitialAgentMessages({ arm: "A-tools", packet });
    expect(messages).toHaveLength(1);
    expect(messages[0]!.role).toBe("user");
    expect(messages[0]!.content).toContain("Return exactly one JSON object");
    expect(messages[0]!.content).toContain("Incident packet:");
    expect(messages[0]!.content.split("Incident packet:")[0]).not.toContain("not_registered");
  });

  it("retains exact short results and makes long results valid bounded JSON", () => {
    const short = boundToolResult({ status: "ok", rows: [1] }, 256);
    expect(short.truncated).toBe(false);
    expect(JSON.parse(short.transmittedJson)).toEqual({ rows: [1], status: "ok" });
    const long = boundToolResult({ status: "ok", rows: [{ content: "x".repeat(4_000) }] }, 512);
    expect(long.truncated).toBe(true);
    expect(long.transmittedJson.length).toBeLessThanOrEqual(512);
    expect(JSON.parse(long.transmittedJson)).toMatchObject({ status: "truncated", originalSha256: long.rawSha256 });
  });

  it("rejects unreturned citations, direct-arm citations, and non-abstaining unknowns", () => {
    expect(decisionPolicyFindings(decision, "A-tools", new Set(["chunk-1"]))).toEqual([]);
    expect(decisionPolicyFindings(decision, "A-direct", new Set())).toEqual([
      "direct_arm_citation",
      "unreturned_citation:chunk-1",
    ]);
    expect(decisionPolicyFindings({ ...decision, incidentClass: "unknown_ambiguous", citedChunkIds: [] }, "A-tools", new Set()))
      .toContain("unknown_without_abstention");
  });
});
