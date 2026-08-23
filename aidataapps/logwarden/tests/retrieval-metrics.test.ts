import { mean, parseRunbookRanking, percentile, scoreRunbookRetrieval } from "../src/retrieval-metrics.js";

describe("retrieval metrics", () => {
  it("scores multi-guide retrieval without rewarding duplicate chunks", () => {
    const score = scoreRunbookRetrieval(
      ["TSG-LOG-01", "TSG-LOG-02"],
      ["TSG-LOG-01", "TSG-LOG-01", "TSG-LOG-02", "TSG-QRY-01"],
      4,
    );
    expect(score.recallAtK).toBe(1);
    expect(score.reciprocalRank).toBe(1);
    expect(score.ndcgAtK).toBeCloseTo((1 + 1 / Math.log2(4)) / (1 + 1 / Math.log2(3)));
    expect(score.relevantRunbooksReturned).toEqual(["TSG-LOG-01", "TSG-LOG-02"]);
    expect(score.noAnswerCorrect).toBeNull();
  });

  it("preserves duplicate rank positions when parsing retained results", () => {
    expect(parseRunbookRanking('["TSG-LOG-01","TSG-LOG-01","TSG-LOG-02"]')).toEqual([
      "TSG-LOG-01", "TSG-LOG-01", "TSG-LOG-02",
    ]);
  });

  it("separates no-answer accuracy from relevance metrics", () => {
    expect(scoreRunbookRetrieval([], [], 5)).toMatchObject({
      answerable: false, recallAtK: 0, reciprocalRank: 0, ndcgAtK: 0, noAnswerCorrect: true,
    });
    expect(scoreRunbookRetrieval([], ["TSG-LOG-01"], 5).noAnswerCorrect).toBe(false);
  });

  it("uses the requested cutoff and validates summary helpers", () => {
    expect(scoreRunbookRetrieval(["TSG-LOG-01"], ["TSG-QRY-01", "TSG-LOG-01"], 1).recallAtK).toBe(0);
    expect(mean([1, 2, 3])).toBe(2);
    expect(percentile([9, 1, 5, 3], 0.5)).toBe(3);
  });
});
