import type { Decision } from "./types.js";

export interface Neighbor {
  model: string;
  distance: number;
  promptGroupId: string;
  textArtifactId: string;
}

export function weightedVote(neighbors: Neighbor[], tau: number): Record<string, number> {
  if (!(tau > 0)) throw new Error("tau must be positive");
  const seenGroups = new Set<string>();
  const seenTexts = new Set<string>();
  const totals = new Map<string, number>();
  for (const row of [...neighbors].sort((a, b) => a.distance - b.distance || a.textArtifactId.localeCompare(b.textArtifactId))) {
    if (seenGroups.has(row.promptGroupId) || seenTexts.has(row.textArtifactId)) continue;
    seenGroups.add(row.promptGroupId);
    seenTexts.add(row.textArtifactId);
    totals.set(row.model, (totals.get(row.model) ?? 0) + Math.exp(-row.distance / tau));
  }
  const denominator = [...totals.values()].reduce((sum, value) => sum + value, 0);
  return Object.fromEntries([...totals.entries()].map(([model, value]) => [model, denominator ? value / denominator : 0]));
}

export function conformalCandidateSet(probabilities: Record<string, number>, quantile: number): string[] {
  return Object.entries(probabilities)
    .filter(([, probability]) => 1 - probability <= quantile)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([model]) => model);
}

export function decide(options: {
  referenceTokens: number;
  supportedLanguage: boolean;
  novelty: number;
  noveltyThreshold: number;
  candidateSet: string[];
  topProbability: number;
}): Decision {
  if (!options.supportedLanguage) return "unsupported_language";
  if (options.referenceTokens < 16) return "insufficient_text";
  if (options.novelty > options.noveltyThreshold) return "unknown_or_out_of_distribution";
  if (options.candidateSet.length !== 1 || options.topProbability < 0.5) return "ambiguous";
  return "attributed";
}
