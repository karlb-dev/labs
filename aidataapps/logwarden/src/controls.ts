import { readFileSync } from "node:fs";
import { z } from "zod";
import { hashJson } from "./hash.js";
import { LAB_ROOT } from "./run.js";

const controlIdSchema = z.enum(["error-number-mask-v1", "shuffled-runbooks-v1", "batching-sequential-v1"]);

const policySchema = z.object({
  schemaVersion: z.literal(1),
  policyId: z.literal("tier1-controls-v1"),
  authoredAtUtc: z.iso.datetime(),
  authoringProvenance: z.object({
    sources: z.array(z.string().min(1)).min(1),
    packetInspection: z.string().min(1),
    temporalDeviation: z.string().min(1),
    impact: z.string().min(1),
  }).strict(),
  controls: z.object({
    "error-number-mask-v1": z.object({
      kind: z.literal("packet_and_tool_transform"), armId: z.literal("A-tools"), subset: z.literal("frozen-control-96"),
      fieldNames: z.array(z.string().min(1)).min(1), numericSignatures: z.array(z.number().int()).min(1),
      signatureRegexes: z.array(z.string().min(1)).min(1), numberReplacement: z.string().min(1), signatureReplacement: z.string().min(1),
    }).strict(),
    "shuffled-runbooks-v1": z.object({
      kind: z.literal("retrieval_override"), armId: z.literal("A-tools"), subset: z.literal("frozen-control-96"),
      sourceMode: z.literal("shuffled_runbook"), sourcePolicy: z.literal("deterministic_hash_ranked_nonacceptable_runbooks_v1"),
      agentVisibleCloneEvaluatorOnly: z.literal(false),
    }).strict(),
    "batching-sequential-v1": z.object({
      kind: z.literal("concurrency_invariance"), armId: z.literal("A-tools"), subset: z.literal("first-48-of-frozen-control-96"),
      selection: z.literal("control_manifest_order"), workers: z.literal(1), comparisonSource: z.literal("primary-workers-16"),
      requiredComparisons: z.array(z.enum(["raw_response_sha256", "decision_sha256", "ordered_tool_call_sha256"])).length(3),
    }).strict(),
    "label-permutation-v1": z.object({
      kind: z.literal("analysis_null"), subset: z.literal("all-frozen-quality-roles"),
      within: z.tuple([z.literal("family"), z.literal("split_role")]), groupingUnit: z.literal("scenario_group_id"),
      seed: z.number().int(), replicates: z.number().int().min(1000), inferenceRequired: z.literal(false),
    }).strict(),
  }).strict(),
}).strict();

export type Tier1ControlPolicy = z.infer<typeof policySchema>;
export type InferenceControlId = z.infer<typeof controlIdSchema>;

export interface ControlTransformResult {
  value: unknown;
  replacementCount: number;
  originalSha256: string;
  transformedSha256: string;
}

export function loadTier1ControlPolicy(path = `${LAB_ROOT}/config/controls/tier1-v1.json`): Tier1ControlPolicy {
  const policy = policySchema.parse(JSON.parse(readFileSync(path, "utf8")));
  for (const source of policy.controls["error-number-mask-v1"].signatureRegexes) compile(source);
  return policy;
}

export function parseInferenceControlId(value: string): InferenceControlId {
  return controlIdSchema.parse(value);
}

export function maskErrorNumbersAndSignatures(
  input: unknown,
  policy = loadTier1ControlPolicy().controls["error-number-mask-v1"],
): ControlTransformResult {
  const originalSha256 = hashJson(input);
  const fields = new Set(policy.fieldNames.map((value) => value.toLowerCase()));
  const numbers = new Set(policy.numericSignatures);
  const patterns = policy.signatureRegexes.map((source) => compile(source));
  let replacementCount = 0;

  const visit = (value: unknown, key?: string): unknown => {
    if (key !== undefined && fields.has(key.toLowerCase())) {
      if (value !== null) replacementCount += 1;
      return null;
    }
    if (typeof value === "number" && numbers.has(value)) {
      replacementCount += 1;
      return policy.numberReplacement;
    }
    if (typeof value === "string") {
      let transformed = value;
      for (const number of [...numbers].sort((left, right) => String(right).length - String(left).length || left - right)) {
        transformed = transformed.replace(new RegExp(`(?<!\\d)${number}(?!\\d)`, "gu"), () => {
          replacementCount += 1;
          return policy.numberReplacement;
        });
      }
      for (const pattern of patterns) transformed = transformed.replace(pattern, () => {
        replacementCount += 1;
        return policy.signatureReplacement;
      });
      return transformed;
    }
    if (Array.isArray(value)) return value.map((child) => visit(child));
    if (value !== null && typeof value === "object") {
      return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([childKey, child]) => [childKey, visit(child, childKey)]));
    }
    return value;
  };

  const value = visit(input);
  return { value, replacementCount, originalSha256, transformedSha256: hashJson(value) };
}

function compile(source: string): RegExp {
  try {
    return new RegExp(source, "giu");
  } catch (error) {
    throw new Error(`Invalid frozen control regex ${JSON.stringify(source)}`, { cause: error });
  }
}
