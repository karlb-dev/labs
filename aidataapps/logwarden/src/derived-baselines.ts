import { readFileSync } from "node:fs";
import { z } from "zod";
import { actions, incidentClasses, severities } from "./contracts.js";
import { hashJson } from "./hash.js";
import { LAB_ROOT } from "./run.js";

const predictionSchema = z.object({
  incidentClass: z.enum(incidentClasses),
  severity: z.enum(severities),
  action: z.enum(actions),
  confidence: z.number().min(0).max(1),
  abstain: z.boolean(),
}).strict();

const derivedBaselineSchema = z.object({
  schemaVersion: z.literal(1),
  configId: z.literal("derived-baselines-v1"),
  authoredAtUtc: z.iso.datetime(),
  authoringProvenance: z.object({
    sources: z.array(z.string().min(1)).min(1),
    packetInspection: z.string().min(1),
    temporalDeviation: z.string().min(1),
    impact: z.string().min(1),
  }).strict(),
  majority: z.object({
    armId: z.literal("B0-majority-no-action-v1"),
    fitRole: z.literal("dev"),
    catalogSha256: z.string().regex(/^[0-9a-f]{64}$/),
    fitEpisodeCount: z.number().int().positive(),
    tieBreak: z.literal("count_desc_then_lexicographic_asc"),
    classCounts: z.record(z.string(), z.number().int().nonnegative()),
    severityCounts: z.record(z.string(), z.number().int().nonnegative()),
    actionCounts: z.record(z.string(), z.number().int().nonnegative()),
    prediction: predictionSchema,
  }).strict(),
  retrievalOnly: z.object({
    corpusId: z.literal("primary-v1"),
    topK: z.number().int().min(1).max(20),
    selection: z.literal("first_unique_returned_runbook"),
    severityPolicy: z.literal("top_runbook_severity_floor"),
    returnedPredictionConfidence: z.number().min(0).max(1),
    arms: z.array(z.object({
      armId: z.enum(["B2-lexical-v1", "B2-vector-v1", "B2-hybrid-v1"]),
      retrievalMode: z.enum(["lexical_fulltext", "vector_exact", "hybrid_rrf"]),
    }).strict()).length(3),
    actionByIncidentClass: z.record(z.enum(incidentClasses), z.enum(actions)),
    noResultPrediction: predictionSchema,
  }).strict(),
  oracle: z.object({
    armId: z.literal("B3-oracle-packet-v1"),
    evaluatorOnly: z.literal(true),
    source: z.literal("eval.ground_truth_episodes"),
    fieldPolicy: z.literal("copy_expected_class_severity_preferred_action_and_should_abstain"),
    confidence: z.literal(1),
  }).strict(),
  router: z.object({
    armId: z.literal("A-router"),
    rulesArmId: z.literal("B1-rules-v1"),
    fallbackArmId: z.literal("A-tools"),
    selection: z.literal("B1_when_prediction_json_resolved_true_else_profile_matched_A-tools"),
    inferenceCalls: z.literal(0),
    rulesCostPolicy: z.literal("zero_incremental_model_tokens_and_latency"),
    fallbackCostPolicy: z.literal("inherit_source_A-tools_costs"),
  }).strict(),
}).strict();

const scenarioCatalogSchema = z.object({
  scenarios: z.array(z.object({
    groundTruth: z.object({
      incidentClass: z.enum(incidentClasses),
      severity: z.enum(severities),
      preferredAction: z.enum(actions),
      shouldAbstain: z.boolean(),
    }).passthrough(),
    variants: z.array(z.object({ splitRole: z.string() }).passthrough()),
  }).passthrough()),
}).passthrough();

export type DerivedBaselineConfig = z.infer<typeof derivedBaselineSchema>;
export type DerivedPrediction = z.infer<typeof predictionSchema>;

export interface RetrievalMetadata {
  runbookId: string;
  incidentClass: typeof incidentClasses[number];
  severityFloor: typeof severities[number];
}

export interface RouterCandidate {
  predictionId: number;
  outcome: "decision" | "abstention" | "failure";
  predictionJson: Record<string, unknown>;
}

export function loadDerivedBaselineConfig(path = `${LAB_ROOT}/config/baselines/derived-v1.json`): DerivedBaselineConfig {
  const config = derivedBaselineSchema.parse(JSON.parse(readFileSync(path, "utf8")));
  if (new Set(config.retrievalOnly.arms.map((arm) => arm.armId)).size !== config.retrievalOnly.arms.length) {
    throw new Error("Duplicate B2 arm in derived baseline config");
  }
  if (new Set(config.retrievalOnly.arms.map((arm) => arm.retrievalMode)).size !== config.retrievalOnly.arms.length) {
    throw new Error("Duplicate B2 retrieval mode in derived baseline config");
  }
  for (const incidentClass of incidentClasses) {
    if (!(incidentClass in config.retrievalOnly.actionByIncidentClass)) {
      throw new Error(`Missing B2 action for ${incidentClass}`);
    }
  }
  return config;
}

export function verifyMajorityFit(
  config = loadDerivedBaselineConfig(),
  catalogPath = `${LAB_ROOT}/config/scenarios/standard-v1.json`,
): void {
  const rawCatalog = JSON.parse(readFileSync(catalogPath, "utf8")) as unknown;
  if (hashJson(rawCatalog) !== config.majority.catalogSha256) throw new Error("B0 catalog hash drift");
  const catalog = scenarioCatalogSchema.parse(rawCatalog);
  const rows = catalog.scenarios.flatMap((scenario) => scenario.variants
    .filter((variant) => variant.splitRole === config.majority.fitRole)
    .map(() => scenario.groundTruth));
  if (rows.length !== config.majority.fitEpisodeCount) throw new Error("B0 fit episode count drift");
  verifyCounts("class", count(rows.map((row) => row.incidentClass)), config.majority.classCounts);
  verifyCounts("severity", count(rows.map((row) => row.severity)), config.majority.severityCounts);
  verifyCounts("action", count(rows.map((row) => row.preferredAction)), config.majority.actionCounts);
  const expected = {
    incidentClass: winner(config.majority.classCounts),
    severity: winner(config.majority.severityCounts),
    action: "no_action" as const,
    confidence: Math.max(...Object.values(config.majority.classCounts)) / config.majority.fitEpisodeCount,
    abstain: false,
  };
  if (hashJson(expected) !== hashJson(config.majority.prediction)) throw new Error("B0 prediction does not match its frozen fit policy");
}

export function majorityPrediction(config = loadDerivedBaselineConfig()): DerivedPrediction {
  verifyMajorityFit(config);
  return { ...config.majority.prediction };
}

export function retrievalOnlyPrediction(
  metadata: RetrievalMetadata | null,
  config = loadDerivedBaselineConfig(),
): DerivedPrediction {
  if (metadata === null) return { ...config.retrievalOnly.noResultPrediction };
  const action = config.retrievalOnly.actionByIncidentClass[metadata.incidentClass];
  return {
    incidentClass: metadata.incidentClass,
    severity: metadata.severityFloor,
    action,
    confidence: config.retrievalOnly.returnedPredictionConfidence,
    abstain: metadata.incidentClass === "unknown_ambiguous",
  };
}

export function oraclePrediction(
  truth: { expectedClass: string; expectedSeverity: string; expectedAction: string; shouldAbstain: boolean },
  config = loadDerivedBaselineConfig(),
): DerivedPrediction {
  return predictionSchema.parse({
    incidentClass: truth.expectedClass,
    severity: truth.expectedSeverity,
    action: truth.expectedAction,
    confidence: config.oracle.confidence,
    abstain: truth.shouldAbstain,
  });
}

export function selectRouterCandidate(b1: RouterCandidate, tools: RouterCandidate): { sourceArmId: "B1-rules-v1" | "A-tools"; candidate: RouterCandidate } {
  if (b1.predictionJson.resolved === true) return { sourceArmId: "B1-rules-v1", candidate: b1 };
  return { sourceArmId: "A-tools", candidate: tools };
}

function count(values: string[]): Record<string, number> {
  const result: Record<string, number> = {};
  for (const value of values) result[value] = (result[value] ?? 0) + 1;
  return Object.fromEntries(Object.entries(result).sort(([left], [right]) => left.localeCompare(right)));
}

function verifyCounts(label: string, actual: Record<string, number>, expected: Record<string, number>): void {
  if (hashJson(actual) !== hashJson(expected)) throw new Error(`B0 ${label} counts drift`);
}

function winner(counts: Record<string, number>): string {
  const ordered = Object.entries(counts).sort(([leftName, leftCount], [rightName, rightCount]) => rightCount - leftCount || leftName.localeCompare(rightName));
  if (ordered[0] === undefined) throw new Error("Cannot select a majority from an empty count map");
  return ordered[0][0];
}
