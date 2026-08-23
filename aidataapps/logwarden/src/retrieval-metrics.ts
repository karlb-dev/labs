export interface RetrievalMetrics {
  answerable: boolean;
  recallAtK: number;
  reciprocalRank: number;
  ndcgAtK: number;
  noAnswerCorrect: boolean | null;
  relevantRunbooksReturned: string[];
}

export function parseRunbookRanking(source: string): string[] {
  const value = JSON.parse(source) as unknown;
  if (!Array.isArray(value) || !value.every((entry) => typeof entry === "string" && /^TSG-[A-Z]{3,5}-\d{2}$/.test(entry))) {
    throw new Error("Runbook ranking JSON violates its protected contract");
  }
  return [...value];
}

/**
 * Score ranked runbook IDs while counting each returned runbook at most once.
 * Recall is set recall over all acceptable runbooks; MRR and nDCG use the
 * first occurrence of each acceptable runbook. No-answer cells are scored by
 * noAnswerCorrect instead of overloading the relevance metrics.
 */
export function scoreRunbookRetrieval(
  expectedRunbooks: readonly string[],
  returnedRunbooks: readonly string[],
  topK: number,
): RetrievalMetrics {
  if (!Number.isSafeInteger(topK) || topK < 1) throw new Error("topK must be a positive integer");
  const expected = uniqueNonempty(expectedRunbooks, "expectedRunbooks");
  const returned = returnedRunbooks.slice(0, topK);
  for (const [index, value] of returned.entries()) {
    if (typeof value !== "string" || value.length === 0) throw new Error(`returnedRunbooks[${index}] must be non-empty`);
  }
  if (expected.length === 0) {
    return {
      answerable: false,
      recallAtK: 0,
      reciprocalRank: 0,
      ndcgAtK: 0,
      noAnswerCorrect: returned.length === 0,
      relevantRunbooksReturned: [],
    };
  }

  const relevant = new Set(expected);
  const seenRelevant = new Set<string>();
  const gains: number[] = returned.map((runbookId) => {
    if (!relevant.has(runbookId) || seenRelevant.has(runbookId)) return 0;
    seenRelevant.add(runbookId);
    return 1;
  });
  const firstRelevant = gains.findIndex((gain) => gain === 1);
  const dcg = gains.reduce((sum, gain, index) => sum + gain / Math.log2(index + 2), 0);
  const idealRelevant = Math.min(expected.length, topK);
  let idcg = 0;
  for (let index = 0; index < idealRelevant; index += 1) idcg += 1 / Math.log2(index + 2);
  return {
    answerable: true,
    recallAtK: seenRelevant.size / expected.length,
    reciprocalRank: firstRelevant < 0 ? 0 : 1 / (firstRelevant + 1),
    ndcgAtK: idcg === 0 ? 0 : dcg / idcg,
    noAnswerCorrect: null,
    relevantRunbooksReturned: [...seenRelevant],
  };
}

export function mean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function percentile(values: readonly number[], fraction: number): number {
  if (!Number.isFinite(fraction) || fraction < 0 || fraction > 1) throw new Error("fraction must be within 0..1");
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)]!;
}

function uniqueNonempty(values: readonly string[], label: string): string[] {
  const result: string[] = [];
  const seen = new Set<string>();
  for (const [index, value] of values.entries()) {
    if (typeof value !== "string" || value.length === 0) throw new Error(`${label}[${index}] must be non-empty`);
    if (!seen.has(value)) {
      seen.add(value);
      result.push(value);
    }
  }
  return result;
}
