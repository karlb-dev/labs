import { canonicalRetrievalQuery, validateRetrievalRows } from "../src/retrieval.js";

describe("retrieval contract", () => {
  it("canonicalizes bounded user queries", () => {
    expect(canonicalRetrievalQuery("  transaction\n log   full ")).toBe("transaction log full");
    expect(() => canonicalRetrievalQuery("   ")).toThrow(/1\.\.1000/);
  });

  it("validates lexical, vector, and hybrid component shapes", () => {
    expect(validateRetrievalRows([row({ lexical_rank: 1, lexical_score: 900 })], "lexical_fulltext", 5)[0])
      .toMatchObject({ lexicalRank: 1, vectorRank: null });
    expect(validateRetrievalRows([row({ execution_mode: "vector_exact", vector_rank: 1, vector_distance: 0.1 })], "vector_exact", 5)[0])
      .toMatchObject({ vectorRank: 1, lexicalRank: null });
    expect(validateRetrievalRows([row({ execution_mode: "hybrid_rrf", lexical_rank: 2, lexical_score: 700, vector_rank: 1, vector_distance: 0.05 })], "hybrid_rrf", 5)[0])
      .toMatchObject({ lexicalRank: 2, vectorRank: 1 });
  });

  it("rejects rank, mode, and component-shape drift", () => {
    expect(() => validateRetrievalRows([row({ rank_ordinal: 2, lexical_rank: 1, lexical_score: 900 })], "lexical_fulltext", 5)).toThrow(/ordered sequence/);
    expect(() => validateRetrievalRows([row({ lexical_rank: 1, lexical_score: 900, vector_rank: 1 })], "lexical_fulltext", 5)).toThrow(/component shape/);
    expect(() => validateRetrievalRows([row({ execution_mode: "hybrid_rrf" })], "hybrid_rrf", 5)).toThrow(/component rank/);
  });
});

function row(overrides: Record<string, unknown>) {
  return {
    rank_ordinal: 1,
    chunk_id: "TSG-LOG-01-c01",
    runbook_id: "TSG-LOG-01",
    lexical_rank: null,
    vector_rank: null,
    lexical_score: null,
    vector_distance: null,
    fused_score: 0.02,
    execution_mode: "lexical_fulltext",
    retrieval_run_id: 1,
    heading_path: "Transaction log > Scope",
    content: "Inspect transaction log utilization.",
    ...overrides,
  };
}
