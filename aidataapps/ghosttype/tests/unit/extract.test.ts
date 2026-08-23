import { describe, expect, it } from "vitest";
import { extractCandidate } from "../../src/extract.js";

// candidate-extract-v1 contract (addendum A-1): whitespace strip and
// suffix-overlap trim are the ONLY repairs; fences/prose/mode leaks are
// flagged, never repaired; empty extracted text is an abstention.
describe("candidate-extract-v1", () => {
  it("strips a single leading/trailing whitespace run", () => {
    const result = extractCandidate("\n  SELECT 1;  \n", "");
    expect(result.extracted).toBe("SELECT 1;");
    expect(result.strippedLeading).toBe(3);
    expect(result.formatFailure).toBeNull();
  });

  it("flags markdown fences without repairing them", () => {
    const result = extractCandidate("```sql\nSELECT 1;\n```", "");
    expect(result.formatFailure).toBe("markdown_fence");
    expect(result.extracted).toContain("```");
  });

  it("flags prose leads", () => {
    expect(extractCandidate("Here is the query: SELECT 1;", "").formatFailure).toBe("prose");
    expect(extractCandidate("Sure, SELECT 1;", "").formatFailure).toBe("prose");
  });

  it("flags prompt-scaffold mode leaks", () => {
    expect(extractCandidate("<mode>intent</mode> SELECT 1;", "").formatFailure).toBe("mode_leak");
  });

  it("does not flag SQL that merely starts with a keyword resembling prose", () => {
    expect(extractCandidate("THEN 'Large' ELSE 'Standard' END", "").formatFailure).toBeNull();
  });

  it("trims the classic FIM suffix duplication and records the length", () => {
    const result = extractCandidate("Name, RegionId\nFROM dbo.Customers;", "\nFROM dbo.Customers;");
    expect(result.extracted).toBe("Name, RegionId");
    expect(result.suffixOverlapTrimmed).toBe("\nFROM dbo.Customers;".length);
  });

  it("leaves candidates alone when no tail matches the suffix", () => {
    const result = extractCandidate("Name", "\nFROM dbo.Customers;");
    expect(result.extracted).toBe("Name");
    expect(result.suffixOverlapTrimmed).toBe(0);
  });

  it("treats empty output as abstention", () => {
    expect(extractCandidate("", "x").isAbstention).toBe(true);
    expect(extractCandidate("  \n ", "x").isAbstention).toBe(true);
  });
});
