import { z } from "zod";
import { hashJson } from "./hash.js";

const expectedEvidenceSchema = z.object({
  source: z.enum(["xe", "errorlog", "query_store"]),
  event: z.string().min(1).max(120),
  errorNumber: z.number().int().positive().optional(),
  minimumCount: z.number().int().min(0).default(1),
});

const scenarioSchema = z.object({
  id: z.string().regex(/^[a-z0-9-]{3,100}$/),
  groupId: z.string().regex(/^[a-z0-9-]{3,100}$/),
  family: z.string().regex(/^[a-z0-9_]{3,80}$/),
  regime: z.enum(["K", "C", "U", "M", "N"]),
  description: z.string().min(1).max(1000),
  injector: z.enum([
    "conversion_error", "missing_object", "duplicate_key", "truncation_error",
    "controlled_signal", "bad_login", "backup_failure", "query_pressure",
    "benign_noise", "ambiguous_signal",
  ]),
  expectedClass: z.string().min(1).max(80),
  expectedSeverity: z.string().min(1).max(24),
  shouldAbstain: z.boolean(),
  expectedEvidence: z.array(expectedEvidenceSchema).min(1),
});

export const scenarioCatalogSchema = z.object({
  schemaVersion: z.literal(1),
  catalogId: z.string().min(1).max(80),
  scenarios: z.array(scenarioSchema).min(1),
});

export type ScenarioCatalog = z.infer<typeof scenarioCatalogSchema>;
export type ScenarioDefinition = ScenarioCatalog["scenarios"][number];

export function scenarioIdentity(catalogId: string, scenario: ScenarioDefinition) {
  const config = {
    schemaVersion: 1,
    catalogId,
    injector: scenario.injector,
    packetWindow: { beforeSeconds: 30, afterSeconds: 90 },
    snapshotPlan: [],
  };
  const groundTruth = {
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
    config,
    groundTruth,
    expectedEvidence: scenario.expectedEvidence,
  };
  return { config, groundTruth, sha256: hashJson(identity) };
}
