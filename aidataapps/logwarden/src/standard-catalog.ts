import { scenarioCatalogSchema, validateScenarioCatalog, type ScenarioCatalog, type ScenarioDefinition } from "./scenarios.js";

type Mode = "known" | "context" | "multi" | "near_miss" | "unknown";
type SplitRole = NonNullable<ScenarioDefinition["variants"]>[number]["splitRole"];

interface TemplateSlot { role: SplitRole; mode: Mode }
interface FamilyPlan {
  code: string;
  family: string;
  incidentClass: string;
  injector: ScenarioDefinition["injector"];
  severity: "medium" | "high";
  runbookCode: string;
  description: string;
  requiredTools: string[];
  optionalTools: string[];
  snapshots: ScenarioDefinition["snapshotPlan"];
  expectedEvidence: ScenarioDefinition["expectedEvidence"];
  maxRuntimeSeconds?: number;
  packetAfterSeconds?: number;
}

const allTools = [
  "runbook_search", "get_recent_incident_counts", "get_blocking_snapshot",
  "get_log_space", "get_active_transactions", "get_backup_history",
  "get_deadlock_graph",
];

const plans: FamilyPlan[] = [
  {
    code: "dlk", family: "deadlock", incidentClass: "deadlock", injector: "deadlock", severity: "high", runbookCode: "DLK",
    description: "A controlled cyclic lock dependency with bounded victim recovery.",
    requiredTools: ["get_deadlock_graph", "runbook_search"], optionalTools: ["get_recent_incident_counts"],
    snapshots: [{ tool: "get_deadlock_graph", atMsAfterStart: 0, arguments: { maxRows: 5 } }],
    expectedEvidence: [
      { source: "xe", event: "xml_deadlock_report", minimumCount: 1, required: true },
      { source: "xe", event: "error_reported", errorNumber: 1205, minimumCount: 1, required: true },
    ], maxRuntimeSeconds: 20,
  },
  {
    code: "blk", family: "blocking_waits", incidentClass: "blocking", injector: "blocking", severity: "medium", runbookCode: "BLK",
    description: "A controlled head blocker and waiter held beyond the frozen threshold.",
    requiredTools: ["get_blocking_snapshot", "runbook_search"], optionalTools: ["get_active_transactions", "get_recent_incident_counts"],
    snapshots: [
      { tool: "get_blocking_snapshot", atMsAfterStart: 2500, arguments: { databaseName: "LogWardenWorkload", maxRows: 20 } },
      { tool: "get_active_transactions", atMsAfterStart: 2600, arguments: { databaseName: "LogWardenWorkload", maxRows: 20 } },
    ],
    expectedEvidence: [{ source: "xe", event: "blocked_process_report", minimumCount: 1, required: true }],
    maxRuntimeSeconds: 20, packetAfterSeconds: 120,
  },
  {
    code: "log", family: "transaction_log_space", incidentClass: "transaction_log_full", injector: "transaction_log_full", severity: "high", runbookCode: "LOG",
    description: "A capped disposable database log fills while a transaction prevents reuse.",
    requiredTools: ["get_log_space", "runbook_search"], optionalTools: ["get_active_transactions", "get_backup_history"],
    snapshots: [
      { tool: "get_log_space", atMsAfterStart: 0, arguments: { databaseName: "TARGET_DATABASE" } },
      { tool: "get_active_transactions", atMsAfterStart: 0, arguments: { databaseName: "TARGET_DATABASE", maxRows: 20 } },
    ],
    expectedEvidence: [{ source: "xe", event: "error_reported", errorNumber: 9002, minimumCount: 1, required: true }],
    maxRuntimeSeconds: 45, packetAfterSeconds: 120,
  },
  {
    code: "aut", family: "authentication_access", incidentClass: "authentication_failure", injector: "bad_login", severity: "high", runbookCode: "AUT",
    description: "A rejected disposable identity produces correlated login-failure evidence.",
    requiredTools: ["get_recent_incident_counts", "runbook_search"], optionalTools: [],
    snapshots: [{ tool: "get_recent_incident_counts", atMsAfterStart: 0, arguments: { incidentClass: "authentication_failure", windowMinutes: 60 } }],
    expectedEvidence: [
      { source: "xe", event: "error_reported", errorNumber: 18456, minimumCount: 1, required: true },
      { source: "errorlog", event: "errorlog", errorNumber: 18456, minimumCount: 1, required: true },
    ],
  },
  {
    code: "int", family: "integrity_signal", incidentClass: "integrity_signal", injector: "controlled_signal", severity: "high", runbookCode: "INT",
    description: "A logged, lab-authored integrity signal exercises conservative escalation without page damage.",
    requiredTools: ["runbook_search"], optionalTools: ["get_recent_incident_counts", "get_backup_history"],
    snapshots: [{ tool: "get_recent_incident_counts", atMsAfterStart: 0, arguments: { incidentClass: "integrity_signal", windowMinutes: 60 } }],
    expectedEvidence: [
      { source: "xe", event: "error_reported", errorNumber: 50000, minimumCount: 1, required: true },
      { source: "errorlog", event: "errorlog", errorNumber: 50000, minimumCount: 1, required: true },
    ],
  },
  {
    code: "bak", family: "backup_restore_recovery", incidentClass: "backup_restore_failure", injector: "backup_failure", severity: "high", runbookCode: "BAK",
    description: "A backup targets a controlled inaccessible destination and preserves attempt history.",
    requiredTools: ["get_backup_history", "runbook_search"], optionalTools: ["get_recent_incident_counts"],
    snapshots: [{ tool: "get_backup_history", atMsAfterStart: 0, arguments: { databaseName: "LogWardenWorkload", maxRows: 20 } }],
    expectedEvidence: [{ source: "xe", event: "error_reported", errorNumber: 3201, minimumCount: 1, required: true }],
  },
  {
    code: "qry", family: "query_resource_pressure", incidentClass: "query_resource_pressure", injector: "query_pressure", severity: "medium", runbookCode: "QRY",
    description: "A bounded CPU-heavy query records duration and resource evidence without host exhaustion.",
    requiredTools: ["runbook_search"], optionalTools: ["get_recent_incident_counts"],
    snapshots: [{ tool: "get_recent_incident_counts", atMsAfterStart: 0, arguments: { incidentClass: "query_resource_pressure", windowMinutes: 60 } }],
    expectedEvidence: [{ source: "xe", event: "sql_batch_completed", minimumCount: 1, required: true }],
  },
  {
    code: "dat", family: "schema_data_errors", incidentClass: "schema_data_error", injector: "conversion_error", severity: "medium", runbookCode: "DAT",
    description: "A deterministic schema or value contract violation produces a stable engine error.",
    requiredTools: ["runbook_search"], optionalTools: ["get_recent_incident_counts"], snapshots: [],
    expectedEvidence: [{ source: "xe", event: "error_reported", errorNumber: 245, minimumCount: 1, required: true }],
  },
  {
    code: "unk", family: "unknown_ambiguous", incidentClass: "unknown_ambiguous", injector: "ambiguous_signal", severity: "medium", runbookCode: "UNK",
    description: "Incomplete or conflicting evidence requires explicit abstention and human review.",
    requiredTools: [], optionalTools: ["runbook_search", "get_recent_incident_counts"], snapshots: [],
    expectedEvidence: [{ source: "xe", event: "error_reported", errorNumber: 50000, minimumCount: 1, required: true }],
  },
  {
    code: "noi", family: "benign_noise", incidentClass: "benign_noise", injector: "benign_noise", severity: "medium", runbookCode: "NOI",
    description: "Successful or subthreshold activity must remain a no-action observation.",
    requiredTools: [], optionalTools: ["runbook_search"], snapshots: [],
    expectedEvidence: [{ source: "xe", event: "sql_batch_completed", minimumCount: 1, required: true }],
  },
];

const nonTestRoles: Array<[SplitRole, SplitRole?]> = [
  ["dev"],
  ["calibration"],
  ["dev", "test_unknown"],
  ["calibration", "test_unknown"],
  ["dev", "calibration"],
  ["dev", "test_unknown"],
  ["calibration", "test_unknown"],
  ["dev", "calibration"],
  ["calibration", "test_unknown"],
  ["dev", "test_unknown"],
];

export function generateStandardScenarioCatalog(): ScenarioCatalog {
  const scenarios = plans.flatMap((plan, familyIndex) => templateSlots(familyIndex).map((slot, slotIndex) =>
    buildScenario(plan, familyIndex, slot, slotIndex)));
  const catalog = scenarioCatalogSchema.parse({ schemaVersion: 1, catalogId: "logwarden-standard-v1", scenarios });
  validateScenarioCatalog(catalog);
  return catalog;
}

function templateSlots(familyIndex: number): TemplateSlot[] {
  const core: TemplateSlot[] = [
    { role: "test_id", mode: familyIndex === 8 ? "unknown" : familyIndex === 9 ? "near_miss" : "known" },
    { role: "test_id", mode: familyIndex === 8 ? "unknown" : familyIndex === 9 ? "near_miss" : "context" },
    { role: "test_id", mode: familyIndex === 8 ? "unknown" : familyIndex === 9 ? "near_miss" : "multi" },
    { role: "test_variant_holdout", mode: familyIndex >= 8 ? (familyIndex === 8 ? "unknown" : "near_miss") : "context" },
  ];
  if (familyIndex < 2) {
    core.push({ role: "test_variant_holdout", mode: "near_miss" }, { role: nonTestRoles[familyIndex]![0], mode: "known" });
  } else {
    const roles = nonTestRoles[familyIndex]!;
    core.push({ role: roles[0], mode: familyIndex === 9 || familyIndex % 2 === 0 ? "near_miss" : "known" });
    core.push({ role: roles[1]!, mode: roles[1] === "test_unknown" ? "unknown" : familyIndex === 8 ? "unknown" : "context" });
  }
  return core;
}

function buildScenario(plan: FamilyPlan, familyIndex: number, slot: TemplateSlot, slotIndex: number): Record<string, unknown> {
  const number = String(slotIndex + 1).padStart(2, "0");
  const mode = slot.mode;
  const isUnknown = mode === "unknown";
  const isNearMiss = mode === "near_miss";
  const expectedClass = isUnknown ? "unknown_ambiguous" : isNearMiss ? "benign_noise" : plan.incidentClass;
  const expectedSeverity = isUnknown ? "low" : isNearMiss ? "info" : mode === "multi" && plan.severity === "medium" ? "high" : plan.severity;
  const injector = isUnknown ? "ambiguous_signal" : isNearMiss ? "benign_noise" : resolveInjector(plan, slotIndex);
  const evidence = isUnknown
    ? [{ source: "xe", event: "error_reported", errorNumber: 50000, minimumCount: 1, required: true }]
    : isNearMiss
      ? [{ source: "xe", event: "sql_batch_completed", minimumCount: 1, required: true }]
      : resolveEvidence(plan, injector);
  const toolPolicy = toolPolicyFor(expectedClass, plan);
  const runbookCode = isUnknown ? "UNK" : isNearMiss ? "NOI" : plan.runbookCode;
  const noAnswer = isUnknown && familyIndex % 2 === 0;
  const acceptableRunbooks = noAnswer ? [] : Array.from({ length: 6 }, (_, index) => `TSG-${runbookCode}-${String(index + 1).padStart(2, "0")}`);
  const variants = Array.from({ length: 10 }, (_, index) => ({
    id: `std-${plan.code}-${number}-v${String(index + 1).padStart(2, "0")}`,
    splitRole: slot.role,
    parameters: {
      schemaVersion: 1,
      variantOrdinal: index + 1,
      messageVariant: (index % 5) + 1,
      signalCount: mode === "multi" ? 3 : 1,
      recurrenceOrdinal: index + 1,
      mode,
      injector,
    },
    messageViewPolicy: isUnknown ? (index % 2 === 0 ? "error-number-masked-v1" : "truncated-v1")
      : slot.role === "test_variant_holdout" ? (index % 2 === 0 ? "paraphrased-v1" : "raw-minimal-v1") : "raw-minimal-v1",
    rateContextId: mode === "multi" ? "burst-10" : mode === "context" ? "recurrent-10" : "isolated",
  }));
  return {
    id: `std-${plan.code}-${number}`,
    groupId: `grp-${plan.code}-${number}`,
    family: plan.family,
    regime: mode === "known" ? "K" : mode === "context" ? "C" : mode === "multi" ? "M" : mode === "near_miss" ? "N" : "U",
    description: `${plan.description} Template ${number} is assigned wholly to ${slot.role}.`,
    injector,
    expectedClass,
    expectedSeverity,
    shouldAbstain: isUnknown,
    expectedEvidence: evidence,
    maxRuntimeSeconds: plan.maxRuntimeSeconds ?? 30,
    isMultiEvent: mode === "multi",
    isContextDependent: mode === "context" || mode === "multi",
    packetWindow: { beforeSeconds: 30, afterSeconds: plan.packetAfterSeconds ?? 90 },
    snapshotPlan: isUnknown || isNearMiss ? [] : plan.snapshots,
    groundTruth: {
      incidentClass: expectedClass,
      severity: expectedSeverity,
      correctActions: isUnknown ? ["escalate_to_human"] : isNearMiss ? ["no_action"] : ["run_tsg", "open_work_item"],
      preferredAction: isUnknown ? "escalate_to_human" : isNearMiss ? "no_action" : "run_tsg",
      acceptableRunbooks,
      ...toolPolicy,
      argumentConstraints: toolArgumentConstraints(toolPolicy.requiredTools),
      shouldAbstain: isUnknown,
      maxIndependentDecisions: 1,
      costWeights: { miss: isNearMiss ? 2 : 10, falseAlarm: isNearMiss ? 5 : 2, unnecessaryTool: 0.25 },
    },
    variants,
  };
}

function resolveInjector(plan: FamilyPlan, slotIndex: number): ScenarioDefinition["injector"] {
  if (plan.code !== "dat") return plan.injector;
  return (["conversion_error", "missing_object", "duplicate_key", "truncation_error", "divide_by_zero", "conversion_error"] as const)[slotIndex]!;
}

function resolveEvidence(plan: FamilyPlan, injector: ScenarioDefinition["injector"]): ScenarioDefinition["expectedEvidence"] {
  if (plan.code !== "dat") return plan.expectedEvidence;
  const errors: Partial<Record<ScenarioDefinition["injector"], number>> = {
    conversion_error: 245, missing_object: 208, duplicate_key: 2627, truncation_error: 2628, divide_by_zero: 8134,
  };
  return [{ source: "xe", event: "error_reported", errorNumber: errors[injector]!, minimumCount: 1, required: true }];
}

function toolPolicyFor(expectedClass: string, plan: FamilyPlan): { requiredTools: string[]; optionalTools: string[]; forbiddenTools: string[] } {
  if (expectedClass === "unknown_ambiguous") return { requiredTools: [], optionalTools: ["runbook_search", "get_recent_incident_counts"], forbiddenTools: [] };
  if (expectedClass === "benign_noise") return { requiredTools: [], optionalTools: ["runbook_search"], forbiddenTools: allTools.filter((tool) => tool !== "runbook_search") };
  const allowed = new Set([...plan.requiredTools, ...plan.optionalTools]);
  return { requiredTools: plan.requiredTools, optionalTools: plan.optionalTools, forbiddenTools: allTools.filter((tool) => !allowed.has(tool)) };
}

function toolArgumentConstraints(requiredTools: string[]): Record<string, unknown> {
  const constraints: Record<string, unknown> = {};
  for (const tool of requiredTools) {
    if (["get_blocking_snapshot", "get_log_space", "get_active_transactions", "get_backup_history"].includes(tool))
      constraints[`${tool}.databaseName`] = tool === "get_log_space" || tool === "get_active_transactions" ? "TARGET_DATABASE" : "LogWardenWorkload";
    if (tool === "runbook_search") constraints["runbook_search.corpusId"] = "primary-v1";
  }
  return constraints;
}
