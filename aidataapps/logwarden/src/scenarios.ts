import { z } from "zod";
import { hashJson } from "./hash.js";
import { actions, incidentClasses, severities } from "./contracts.js";

const expectedEvidenceSchema = z.object({
  source: z.enum(["xe", "errorlog", "query_store"]),
  event: z.string().min(1).max(120),
  errorNumber: z.number().int().positive().optional(),
  minimumCount: z.number().int().min(0).default(1),
  maximumCount: z.number().int().min(0).optional(),
  required: z.boolean().optional(),
});

const splitRoleSchema = z.enum(["dev", "calibration", "test_id", "test_variant_holdout", "test_unknown", "test_live_parity", "test_storm"]);
const snapshotPlanSchema = z.object({
  tool: z.enum(["get_blocking_snapshot", "get_log_space", "get_active_transactions", "get_backup_history", "get_deadlock_graph"]),
  atMsAfterStart: z.number().int().min(0).max(120_000),
  arguments: z.record(z.string(), z.unknown()),
}).strict();
const groundTruthSchema = z.object({
  incidentClass: z.enum(incidentClasses),
  severity: z.enum(severities),
  correctActions: z.array(z.enum(actions)).min(1),
  preferredAction: z.enum(actions),
  acceptableRunbooks: z.array(z.string().regex(/^TSG-[A-Z]{3,5}-\d{2}$/)).max(20),
  requiredTools: z.array(z.string().min(1).max(80)).max(4),
  optionalTools: z.array(z.string().min(1).max(80)).max(7),
  forbiddenTools: z.array(z.string().min(1).max(80)).max(7),
  argumentConstraints: z.record(z.string(), z.unknown()),
  shouldAbstain: z.boolean(),
  maxIndependentDecisions: z.literal(1),
  costWeights: z.object({
    miss: z.number().positive(),
    falseAlarm: z.number().nonnegative(),
    unnecessaryTool: z.number().nonnegative(),
  }).strict(),
}).strict();
const variantSchema = z.object({
  id: z.string().regex(/^[a-z0-9-]{3,120}$/),
  splitRole: splitRoleSchema,
  parameters: z.record(z.string(), z.unknown()),
  messageViewPolicy: z.enum(["raw-minimal-v1", "error-number-masked-v1", "truncated-v1", "paraphrased-v1"]),
  rateContextId: z.string().regex(/^[a-z0-9-]{3,80}$/).nullable().default(null),
}).strict();

const scenarioSchema = z.object({
  id: z.string().regex(/^[a-z0-9-]{3,100}$/),
  groupId: z.string().regex(/^[a-z0-9-]{3,100}$/),
  family: z.string().regex(/^[a-z0-9_]{3,80}$/),
  regime: z.enum(["K", "C", "U", "M", "N"]),
  description: z.string().min(1).max(1000),
  injector: z.enum([
    "conversion_error", "missing_object", "duplicate_key", "truncation_error",
    "controlled_signal", "bad_login", "backup_failure", "query_pressure",
    "benign_noise", "ambiguous_signal", "deadlock", "blocking", "transaction_log_full",
    "divide_by_zero",
  ]),
  expectedClass: z.string().min(1).max(80),
  expectedSeverity: z.string().min(1).max(24),
  shouldAbstain: z.boolean(),
  expectedEvidence: z.array(expectedEvidenceSchema).min(1),
  maxRuntimeSeconds: z.number().int().min(1).max(3600).optional(),
  isMultiEvent: z.boolean().optional(),
  isContextDependent: z.boolean().optional(),
  packetWindow: z.object({
    beforeSeconds: z.number().int().min(0).max(300),
    afterSeconds: z.number().int().min(1).max(600),
  }).strict().optional(),
  snapshotPlan: z.array(snapshotPlanSchema).max(8).optional(),
  groundTruth: groundTruthSchema.optional(),
  variants: z.array(variantSchema).min(1).max(100).optional(),
});

export const scenarioCatalogSchema = z.object({
  schemaVersion: z.literal(1),
  catalogId: z.string().min(1).max(80),
  scenarios: z.array(scenarioSchema).min(1),
});

export type ScenarioCatalog = z.infer<typeof scenarioCatalogSchema>;
export type ScenarioDefinition = ScenarioCatalog["scenarios"][number];
export type ScenarioVariant = z.infer<typeof variantSchema>;

export function scenarioIdentity(catalogId: string, scenario: ScenarioDefinition) {
  const config = {
    schemaVersion: 1,
    catalogId,
    injector: scenario.injector,
    packetWindow: scenario.packetWindow ?? { beforeSeconds: 30, afterSeconds: 90 },
    snapshotPlan: scenario.snapshotPlan ?? [],
    ...(scenario.maxRuntimeSeconds === undefined ? {} : { maxRuntimeSeconds: scenario.maxRuntimeSeconds }),
  };
  const groundTruth = scenario.groundTruth ?? {
    schemaVersion: 1,
    incidentClass: scenario.expectedClass,
    severity: scenario.expectedSeverity,
    correctActions: scenario.shouldAbstain ? ["needs_human_review"] : ["run_tsg", "open_work_item"],
    preferredAction: scenario.shouldAbstain ? "needs_human_review" : "run_tsg",
    acceptableRunbooks: [],
    requiredTools: [],
    optionalTools: ["runbook_search", "get_recent_incident_counts"],
    forbiddenTools: [],
    argumentConstraints: {},
    shouldAbstain: scenario.shouldAbstain,
    maxIndependentDecisions: 1,
    costWeights: { miss: 10, falseAlarm: 2, unnecessaryTool: 0.25 },
  };
  const identity = {
    id: scenario.id,
    groupId: scenario.groupId,
    family: scenario.family,
    regime: scenario.regime,
    description: scenario.description,
    injector: scenario.injector,
    expectedClass: scenario.expectedClass,
    expectedSeverity: scenario.expectedSeverity,
    shouldAbstain: scenario.shouldAbstain,
    ...(scenario.maxRuntimeSeconds === undefined ? {} : { maxRuntimeSeconds: scenario.maxRuntimeSeconds }),
    ...(scenario.isMultiEvent === undefined ? {} : { isMultiEvent: scenario.isMultiEvent }),
    ...(scenario.isContextDependent === undefined ? {} : { isContextDependent: scenario.isContextDependent }),
    config,
    groundTruth,
    expectedEvidence: scenario.expectedEvidence,
  };
  return { config, groundTruth, sha256: hashJson(identity) };
}

export function scenarioVariants(scenario: ScenarioDefinition, fallbackRole: z.infer<typeof splitRoleSchema>): ScenarioVariant[] {
  if (scenario.variants !== undefined) return scenario.variants;
  return [{
    id: `${scenario.id}-dev-v1`,
    splitRole: fallbackRole,
    parameters: { injector: scenario.injector, messageVariant: 1 },
    messageViewPolicy: "raw-minimal-v1",
    rateContextId: null,
  }];
}

export function scenarioVariantIdentity(catalogId: string, scenario: ScenarioDefinition, variant: ScenarioVariant) {
  return {
    schemaVersion: 1,
    catalogId,
    scenarioId: scenario.id,
    variantGroupId: scenario.groupId,
    splitRole: variant.splitRole,
    parameters: variant.parameters,
    messageViewPolicy: variant.messageViewPolicy,
    rateContextId: variant.rateContextId,
  };
}

export function scenarioCatalogManifest(catalog: ScenarioCatalog, fallbackRole: z.infer<typeof splitRoleSchema>) {
  return catalog.scenarios.map((scenario) => ({
    id: scenario.id,
    sha256: scenarioIdentity(catalog.catalogId, scenario).sha256,
    variants: scenarioVariants(scenario, fallbackRole).map((variant) => ({
      id: variant.id,
      sha256: hashJson(scenarioVariantIdentity(catalog.catalogId, scenario, variant)),
    })),
  }));
}

export function validateScenarioCatalog(catalog: ScenarioCatalog): void {
  const scenarioIds = new Set<string>();
  const variantIds = new Set<string>();
  const groupRoles = new Map<string, string>();
  for (const scenario of catalog.scenarios) {
    if (scenarioIds.has(scenario.id)) throw new Error(`Duplicate scenario id: ${scenario.id}`);
    scenarioIds.add(scenario.id);
    if (scenario.groundTruth !== undefined &&
        (scenario.groundTruth.incidentClass !== scenario.expectedClass || scenario.groundTruth.severity !== scenario.expectedSeverity || scenario.groundTruth.shouldAbstain !== scenario.shouldAbstain))
      throw new Error(`Scenario top-level/truth drift: ${scenario.id}`);
    for (const variant of scenario.variants ?? []) {
      if (variantIds.has(variant.id)) throw new Error(`Duplicate scenario variant id: ${variant.id}`);
      variantIds.add(variant.id);
      const priorRole = groupRoles.get(scenario.groupId);
      if (priorRole !== undefined && priorRole !== variant.splitRole)
        throw new Error(`Scenario group crosses split roles: ${scenario.groupId}`);
      groupRoles.set(scenario.groupId, variant.splitRole);
    }
  }
}
