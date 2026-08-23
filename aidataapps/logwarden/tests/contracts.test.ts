import { operatingContract, parseAgentResponse } from "../src/contracts.js";

const decision = {
  kind: "decision",
  incidentClass: "blocking",
  severity: "medium",
  action: "open_work_item",
  actionArguments: {},
  citedChunkIds: [],
  confidence: 0.7,
  abstain: false,
  correlationKey: "block-1",
  summary: "Blocking requires review.",
  rationale: "The bounded snapshot shows an active blocker.",
};

describe("structured JSON transport", () => {
  it("accepts exact JSON without repair", () => {
    expect(parseAgentResponse(JSON.stringify(decision)).repairKind).toBe("none");
  });

  it("allows only the frozen fence and leading-text repairs", () => {
    expect(parseAgentResponse(`\`\`\`json\n${JSON.stringify(decision)}\n\`\`\``).repairKind).toBe("fence_strip");
    expect(parseAgentResponse(`Here is the result: ${JSON.stringify(decision)}`).repairKind).toBe("leading_text_strip");
    expect(() => parseAgentResponse(`${JSON.stringify(decision)} trailing`)).toThrow(/valid JSON/);
  });

  it("makes decision field placement explicit in the operating contract", () => {
    const contract = operatingContract([]);
    expect(contract).toContain("required top-level fields beside actionArguments");
    expect(contract).toContain("never put them inside actionArguments");
  });

  it("rejects schema drift and unknown fields", () => {
    expect(() => parseAgentResponse(JSON.stringify({ ...decision, secret: true }))).toThrow();
    expect(() => parseAgentResponse('{"kind":"tool_request","tool":"x","arguments":{},"extra":1}')).toThrow();
  });

  it("builds a deterministic user-turn operating contract", () => {
    const tools = [{ id: "get_log_space", args: { databaseName: "string" } }];
    expect(operatingContract(tools)).toBe(operatingContract(tools));
    expect(operatingContract(tools)).toContain("Return exactly one JSON object");
  });
});
