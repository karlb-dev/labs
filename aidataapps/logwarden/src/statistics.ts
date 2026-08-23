import { sha256 } from "./hash.js";

export interface PairedMetricRow {
  groupId: string;
  stratum: string;
  left: number;
  right: number;
}

export interface GroupedBootstrapResult {
  algorithm: "paired-cluster-percentile-bootstrap-v1";
  seed: number;
  replicates: number;
  confidenceLevel: number;
  rowCount: number;
  groupCount: number;
  observedDifference: number;
  ciLow: number;
  ciHigh: number;
  replicateMean: number;
  replicateSd: number;
  orderedReplicatesSha256: string;
  replicatesValues: number[];
}

export interface PairedPermutationResult {
  algorithm: "paired-cluster-sign-swap-v1";
  seed: number;
  replicates: number;
  rowCount: number;
  groupCount: number;
  observedDifference: number;
  nullMean: number;
  nullSd: number;
  pValueTwoSided: number;
  orderedNullSha256: string;
  nullValues: number[];
}

export interface GroupLabelRow<Label> {
  rowId: string;
  groupId: string;
  stratum: string;
  label: Label;
}

export interface GroupLabelPermutationResult {
  algorithm: "within-stratum-group-label-permutation-v1";
  seed: number;
  replicates: number;
  rowCount: number;
  groupCount: number;
  stratumCount: number;
  observedValue: number;
  nullMean: number;
  nullSd: number;
  pValueTwoSided: number;
  orderedNullSha256: string;
  nullValues: number[];
}

export function pairedGroupedBootstrap(
  rows: PairedMetricRow[],
  options: { seed: number; replicates: number; confidenceLevel?: number },
): GroupedBootstrapResult {
  const confidenceLevel = options.confidenceLevel ?? 0.95;
  validateRows(rows);
  validateSimulation(options.seed, options.replicates);
  if (!(confidenceLevel > 0 && confidenceLevel < 1)) throw new Error("confidenceLevel must be within (0,1)");
  const groups = grouped(rows);
  const groupIds = [...groups.keys()].sort();
  const random = mulberry32(options.seed);
  const replicatesValues: number[] = [];
  for (let replicate = 0; replicate < options.replicates; replicate += 1) {
    let differenceSum = 0;
    let rowCount = 0;
    for (let draw = 0; draw < groupIds.length; draw += 1) {
      const sampled = groups.get(groupIds[Math.floor(random() * groupIds.length)]!)!;
      for (const row of sampled) {
        differenceSum += row.left - row.right;
        rowCount += 1;
      }
    }
    replicatesValues.push(round(differenceSum / rowCount));
  }
  const ordered = [...replicatesValues].sort((left, right) => left - right);
  const alpha = 1 - confidenceLevel;
  return {
    algorithm: "paired-cluster-percentile-bootstrap-v1",
    seed: options.seed,
    replicates: options.replicates,
    confidenceLevel,
    rowCount: rows.length,
    groupCount: groupIds.length,
    observedDifference: difference(rows),
    ciLow: quantile(ordered, alpha / 2),
    ciHigh: quantile(ordered, 1 - alpha / 2),
    replicateMean: mean(replicatesValues),
    replicateSd: sampleSd(replicatesValues),
    orderedReplicatesSha256: sha256(JSON.stringify(ordered)),
    replicatesValues,
  };
}

export function pairedGroupedSignSwap(
  rows: PairedMetricRow[],
  options: { seed: number; replicates: number },
): PairedPermutationResult {
  validateRows(rows);
  validateSimulation(options.seed, options.replicates);
  const groups = grouped(rows);
  const groupIds = [...groups.keys()].sort();
  const random = mulberry32(options.seed);
  const observedDifference = difference(rows);
  const nullValues: number[] = [];
  for (let replicate = 0; replicate < options.replicates; replicate += 1) {
    let sum = 0;
    for (const groupId of groupIds) {
      const sign = random() < 0.5 ? -1 : 1;
      for (const row of groups.get(groupId)!) sum += sign * (row.left - row.right);
    }
    nullValues.push(round(sum / rows.length));
  }
  const extreme = nullValues.filter((value) => Math.abs(value) >= Math.abs(observedDifference)).length;
  const ordered = [...nullValues].sort((left, right) => left - right);
  return {
    algorithm: "paired-cluster-sign-swap-v1",
    seed: options.seed,
    replicates: options.replicates,
    rowCount: rows.length,
    groupCount: groupIds.length,
    observedDifference,
    nullMean: mean(nullValues),
    nullSd: sampleSd(nullValues),
    pValueTwoSided: round((extreme + 1) / (options.replicates + 1)),
    orderedNullSha256: sha256(JSON.stringify(ordered)),
    nullValues,
  };
}

export function withinStratumGroupLabelPermutation<Label>(
  rows: GroupLabelRow<Label>[],
  statistic: (permutedLabels: ReadonlyMap<string, Label>) => number,
  options: { seed: number; replicates: number; observedValue: number },
): GroupLabelPermutationResult {
  validateSimulation(options.seed, options.replicates);
  if (!Number.isFinite(options.observedValue)) throw new Error("observedValue must be finite");
  if (rows.length < 2 || new Set(rows.map((row) => row.rowId)).size !== rows.length) throw new Error("Label permutation requires unique rows");
  const strata = new Map<string, Map<string, GroupLabelRow<Label>[]>>();
  for (const [index, row] of rows.entries()) {
    if (row.rowId.length === 0 || row.groupId.length === 0 || row.stratum.length === 0) throw new Error(`Label row ${index} lacks identity`);
    const groups = strata.get(row.stratum) ?? new Map<string, GroupLabelRow<Label>[]>();
    groups.set(row.groupId, [...(groups.get(row.groupId) ?? []), row]);
    strata.set(row.stratum, groups);
  }
  for (const [stratum, groups] of strata) {
    const sizes = new Set([...groups.values()].map((values) => values.length));
    if (sizes.size !== 1) throw new Error(`Label-permutation groups have unequal sizes within ${stratum}`);
    for (const values of groups.values()) values.sort((left, right) => left.rowId.localeCompare(right.rowId));
  }
  const random = mulberry32(options.seed);
  const nullValues: number[] = [];
  for (let replicate = 0; replicate < options.replicates; replicate += 1) {
    const labels = new Map<string, Label>();
    for (const groups of [...strata.values()]) {
      const targetIds = [...groups.keys()].sort();
      const sourceIds = shuffled(targetIds, random);
      for (const [groupOrdinal, targetId] of targetIds.entries()) {
        const targetRows = groups.get(targetId)!;
        const sourceRows = groups.get(sourceIds[groupOrdinal]!)!;
        for (const [rowOrdinal, target] of targetRows.entries()) labels.set(target.rowId, sourceRows[rowOrdinal]!.label);
      }
    }
    const value = statistic(labels);
    if (!Number.isFinite(value)) throw new Error(`Label-permutation replicate ${replicate} returned a non-finite statistic`);
    nullValues.push(round(value));
  }
  const extreme = nullValues.filter((value) => Math.abs(value) >= Math.abs(options.observedValue)).length;
  const ordered = [...nullValues].sort((left, right) => left - right);
  return {
    algorithm: "within-stratum-group-label-permutation-v1",
    seed: options.seed,
    replicates: options.replicates,
    rowCount: rows.length,
    groupCount: new Set(rows.map((row) => row.groupId)).size,
    stratumCount: strata.size,
    observedValue: round(options.observedValue),
    nullMean: mean(nullValues),
    nullSd: sampleSd(nullValues),
    pValueTwoSided: round((extreme + 1) / (options.replicates + 1)),
    orderedNullSha256: sha256(JSON.stringify(ordered)),
    nullValues,
  };
}

export function holmAdjustedPValues(values: Array<{ id: string; pValue: number }>): Record<string, number> {
  if (new Set(values.map((value) => value.id)).size !== values.length) throw new Error("Holm inputs require unique IDs");
  for (const value of values) if (!(value.pValue >= 0 && value.pValue <= 1)) throw new Error(`Invalid p-value for ${value.id}`);
  const ordered = [...values].sort((left, right) => left.pValue - right.pValue || left.id.localeCompare(right.id));
  const output: Record<string, number> = {};
  let prior = 0;
  for (const [index, value] of ordered.entries()) {
    const adjusted = Math.min(1, (ordered.length - index) * value.pValue);
    prior = Math.max(prior, adjusted);
    output[value.id] = round(prior);
  }
  return output;
}

export function stableStatisticsSeed(baseSeed: number, identity: string): number {
  if (!Number.isSafeInteger(baseSeed) || baseSeed < 0) throw new Error("baseSeed must be a nonnegative safe integer");
  const digest = sha256(`${baseSeed}\0${identity}`);
  return Number.parseInt(digest.slice(0, 8), 16) >>> 0;
}

function validateRows(rows: PairedMetricRow[]): void {
  if (rows.length < 2) throw new Error("Paired analysis requires at least two rows");
  for (const [index, row] of rows.entries()) {
    if (row.groupId.length === 0 || row.stratum.length === 0) throw new Error(`Paired row ${index} lacks grouping identity`);
    if (!Number.isFinite(row.left) || !Number.isFinite(row.right)) throw new Error(`Paired row ${index} has a non-finite value`);
  }
  if (new Set(rows.map((row) => row.groupId)).size < 2) throw new Error("Paired analysis requires at least two groups");
}

function validateSimulation(seed: number, replicates: number): void {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffff_ffff) throw new Error("seed must be a uint32");
  if (!Number.isSafeInteger(replicates) || replicates < 100) throw new Error("at least 100 replicates are required");
}

function grouped(rows: PairedMetricRow[]): Map<string, PairedMetricRow[]> {
  const output = new Map<string, PairedMetricRow[]>();
  for (const row of rows) output.set(row.groupId, [...(output.get(row.groupId) ?? []), row]);
  return output;
}

function difference(rows: PairedMetricRow[]): number { return round(mean(rows.map((row) => row.left - row.right))); }
function mean(values: number[]): number { return round(values.reduce((sum, value) => sum + value, 0) / values.length); }
function sampleSd(values: number[]): number {
  if (values.length < 2) return 0;
  const center = mean(values);
  return round(Math.sqrt(values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1)));
}
function quantile(ordered: number[], probability: number): number {
  const position = (ordered.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower]!;
  return round(ordered[lower]! + (ordered[upper]! - ordered[lower]!) * (position - lower));
}
function shuffled<T>(values: T[], random: () => number): T[] {
  const output = [...values];
  for (let index = output.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    [output[index], output[swap]] = [output[swap]!, output[index]!];
  }
  return output;
}
function round(value: number): number { return Math.round(value * 1_000_000_000) / 1_000_000_000; }

// Small, deterministic, explicitly versioned PRNG. Statistical artifacts retain
// the algorithm name, seed, replicate count, and ordered-distribution hash.
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}
