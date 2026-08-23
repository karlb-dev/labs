import { describe, expect, it } from "vitest";
import { normalizeV1, scoreCandidate, type EvalContract } from "../../src/scoring.js";
import { recomposeForParse } from "../../src/recompose.js";

const contract = (overrides: Partial<EvalContract> = {}): EvalContract => ({
  expect_empty: false, forbid_markdown: true,
  must_contain: [], must_contain_any: [], must_not_contain: [],
  required_objects: [], scoring: [], ...overrides,
});

describe("normalize-v1", () => {
  it("collapses whitespace but stays case-sensitive", () => {
    expect(normalizeV1("SELECT  *\n FROM   t")).toBe("SELECT * FROM t");
    expect(normalizeV1("code")).not.toBe(normalizeV1("Code"));
  });
  it("applies NFC", () => {
    expect(normalizeV1("Strasse")).toBe("Strasse");
    expect(normalizeV1("é")).toBe("é");
  });
});

describe("scoreCandidate", () => {
  it("scores an exact gold as success", () => {
    const score = scoreCandidate({
      candidate: "WHEN 1  THEN 'Active' ELSE 'Inactive' END",
      contract: contract(), acceptedInsertions: ["WHEN 1 THEN 'Active' ELSE 'Inactive' END"],
      docPrefix: "SELECT CASE IsActive ", docSuffix: "", parseIntroduced: 0,
    });
    expect(score.normalizedExact).toBe(true);
    expect(score.outcome).toBe("success");
  });

  it("accepts any of the accepted insertions, not only the canonical", () => {
    const score = scoreCandidate({
      candidate: "B", contract: contract(), acceptedInsertions: ["A", "B"],
      docPrefix: "", docSuffix: "", parseIntroduced: null,
    });
    expect(score.normalizedExact).toBe(true);
  });

  it("scores expect_empty rows: empty is abstain_correct, text is fail", () => {
    const empty = scoreCandidate({
      candidate: "", contract: contract({ expect_empty: true }), acceptedInsertions: [],
      docPrefix: "", docSuffix: "", parseIntroduced: null,
    });
    expect(empty.outcome).toBe("abstain_correct");
    const text = scoreCandidate({
      candidate: "SELECT 1", contract: contract({ expect_empty: true }), acceptedInsertions: [],
      docPrefix: "", docSuffix: "", parseIntroduced: null,
    });
    expect(text.outcome).toBe("fail");
  });

  it("scores empty output on a non-empty row as abstain_wrong", () => {
    const score = scoreCandidate({
      candidate: "", contract: contract(), acceptedInsertions: ["X"],
      docPrefix: "", docSuffix: "", parseIntroduced: null,
    });
    expect(score.outcome).toBe("abstain_wrong");
  });

  it("fails constraints via must_not_contain", () => {
    const score = scoreCandidate({
      candidate: "```sql```", contract: contract({ must_not_contain: ["```"] }),
      acceptedInsertions: ["x"], docPrefix: "", docSuffix: "", parseIntroduced: null,
    });
    expect(score.constraintPass).toBe(false);
    expect(score.outcome).toBe("format_fail");
  });

  it("checks grounding against the recomposed statement, not the candidate alone", () => {
    const score = scoreCandidate({
      candidate: "Name", contract: contract({ required_objects: ["dbo.Customers"] }),
      acceptedInsertions: ["Name"], docPrefix: "SELECT c.", docSuffix: "\nFROM dbo.Customers AS c;",
      parseIntroduced: null,
    });
    expect(score.groundingPass).toBe(true);
  });

  it("gates success on the parse oracle", () => {
    const score = scoreCandidate({
      candidate: "X", contract: contract(), acceptedInsertions: ["X"],
      docPrefix: "", docSuffix: "", parseIntroduced: 2,
    });
    expect(score.parsePass).toBe(false);
    expect(score.outcome).not.toBe("success");
  });
});

describe("recompose-v1", () => {
  it("joins with a newline when the cursor line is an open -- comment", () => {
    const result = recomposeForParse("-- top customers by revenue", "SELECT 1;", "");
    expect(result.joiner).toBe("\n");
    expect(result.text).toBe("-- top customers by revenue\nSELECT 1;");
  });
  it("uses plain concatenation mid-statement", () => {
    expect(recomposeForParse("SELECT c.", "Name", "").joiner).toBe("");
  });
  it("does not double a newline the candidate already starts with", () => {
    expect(recomposeForParse("-- intent", "\nSELECT 1;", "").joiner).toBe("");
  });
  it("ignores comments on earlier lines", () => {
    expect(recomposeForParse("-- intent\nSELECT c.", "Name", "").joiner).toBe("");
  });
});
