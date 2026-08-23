import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { hashFile, hashJson } from "../src/hash.js";
import { LAB_ROOT, atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";

type JsonObject = Record<string, unknown>;
type CsvRow = Record<string, string>;

interface ExpectedOutput {
  path: string;
  bytes: number;
  sha256: string;
}

interface ExpectedOutputsReceipt extends JsonObject {
  runId: string;
  outputs: ExpectedOutput[];
  receiptSha256: string;
}

interface FigureManifest extends JsonObject {
  runId: string;
  figures: Array<{
    figure: string;
    source: string;
    sourceSha256: string;
    caption: string;
    renderingTolerance: string;
  }>;
  receiptSha256: string;
}

const modelDisplay: Record<string, { name: string; short: string; slot: number }> = {
  "qwen-3.8-27b": { name: "Qwen 3.8 27B", short: "Qwen", slot: 1 },
  "gemma-4-31b": { name: "Gemma 4 31B", short: "Gemma", slot: 3 },
  "muse-glimmer-30b": { name: "Muse Glimmer 30B", short: "Muse", slot: 8 },
  "olmo-3.1-32b-instruct": { name: "OLMo 3.1 32B", short: "OLMo", slot: 2 },
};
const modelOrder = ["qwen-3.8-27b", "gemma-4-31b", "muse-glimmer-30b"];
const armDisplay: Record<string, string> = {
  "A-direct": "Direct",
  "A-rag": "RAG",
  "A-tools": "Tools",
  "A-router": "Rules router",
};
const armOrder = ["A-direct", "A-rag", "A-tools", "A-router"];

const runDirectory = resolveRunDirectory();
const runId = basename(runDirectory);
const outputPath = valueAfter("--out") ?? `${runDirectory}/reports/LOGWARDEN_TIER1_REPORT.html`;
const dataPath = valueAfter("--data-out") ?? `${runDirectory}/reports/LOGWARDEN_TIER1_REPORT_DATA.json`;

const expected = readJson<ExpectedOutputsReceipt>(`${runDirectory}/repro/expected-report-outputs.json`);
verifyReceipt(expected, "expected report outputs");
if (expected.runId !== runId) throw new Error(`Expected-output run mismatch: ${expected.runId} != ${runId}`);
const expectedByPath = new Map(expected.outputs.map((item) => [item.path, item]));

const frozenTablePaths = [
  "tables/scorecard.csv",
  "tables/agent-cell-performance.csv",
  "tables/model-service-performance.csv",
  "tables/calibration-summary.csv",
  "tables/negative-controls.jsonl",
  "tables/F02_failure_funnel.csv",
  "tables/F05_tool_precision_recall.csv",
  "tables/F06_grounding_causal_effect.csv",
  "tables/F07_retrieval_scorecard.csv",
  "tables/F09_masking_dropoff.csv",
  "tables/F11_latency_decomposition.csv",
] as const;
for (const relativePath of frozenTablePaths) {
  const expectedOutput = expectedByPath.get(relativePath);
  if (!expectedOutput) throw new Error(`Frozen output receipt omits ${relativePath}`);
  const actual = await hashFile(`${runDirectory}/${relativePath}`);
  if (actual !== expectedOutput.sha256) throw new Error(`Frozen output hash mismatch for ${relativePath}`);
}

const campaign = readJson<JsonObject & {
  runId: string;
  finishedAtUtc: string;
  input: {
    predictions: number;
    analyzedProfiles: string[];
    portExclusions: Array<{ profileKey: string; disposition: string; gateCount: number; gates: JsonObject[] }>;
  };
  output: { metricRows: number; contrastRows: number };
  replicates: { bootstrap: number; permutation: number };
  artifacts: { contrasts: { sha256: string } };
  disposition: string;
  receiptSha256: string;
}>(`${runDirectory}/metrics/campaign-analysis.json`);
verifyReceipt(campaign, "campaign analysis");
if (campaign.runId !== runId || campaign.disposition !== "PASS") throw new Error("Campaign analysis is not a passing run receipt");

const telemetry = readJson<JsonObject & {
  runId: string;
  observed: Record<string, number>;
  disposition: string;
  receiptSha256: string;
}>(`${runDirectory}/telemetry/reconciliation.json`);
verifyReceipt(telemetry, "telemetry reconciliation");
if (telemetry.runId !== runId || telemetry.disposition !== "PASS") throw new Error("Telemetry is not reconciled");

const finalization = readJson<JsonObject & {
  runId: string;
  taxonomyRows: number;
  claimRows: number;
  reportSnapshots: number;
  disposition: string;
  receiptSha256: string;
}>(`${runDirectory}/metrics/results-finalization.json`);
verifyReceipt(finalization, "result finalization");

const rowPass = readJson<JsonObject & { runId: string; disposition: string; receiptSha256: string }>(`${runDirectory}/repro/rows-pass.json`);
const restorePass = readJson<JsonObject & { runId: string; disposition: string; receiptSha256: string }>(`${runDirectory}/repro/restore-pass.json`);
verifyReceipt(rowPass, "row reconstruction");
verifyReceipt(restorePass, "restore reconstruction");
if (rowPass.disposition !== "PASS" || restorePass.disposition !== "PASS") throw new Error("Reproduction receipts are not passing");

const backup = readJson<JsonObject & {
  runId: string;
  createdAtUtc: string;
  receiptSha256: string;
}>(`${runDirectory}/database/backup-receipt.json`);
verifyReceipt(backup, "final database backup");

const figureManifest = readJson<FigureManifest>(`${runDirectory}/figures/manifest.json`);
verifyReceipt(figureManifest, "figure manifest");
if (figureManifest.runId !== runId) throw new Error("Figure-manifest run mismatch");
for (const item of figureManifest.figures) {
  const sourcePath = `tables/${item.source}`;
  const frozen = expectedByPath.get(sourcePath);
  if (!frozen || frozen.sha256 !== item.sourceSha256) throw new Error(`Figure source is not frozen: ${sourcePath}`);
}

const scoreRows = parseCsv(readFileSync(`${runDirectory}/tables/scorecard.csv`, "utf8"));
const performanceRows = parseCsv(readFileSync(`${runDirectory}/tables/agent-cell-performance.csv`, "utf8"));
const performanceByCell = new Map(performanceRows.map((row) => [
  `${requiredText(row, "profile")}\u0000${requiredText(row, "arm")}`,
  row,
]));

const baselines = scoreRows
  .filter((row) => row.kind === "BASELINE")
  .map((row) => scorecardRow(row, null))
  .sort((left, right) => baselineRank(left.armId) - baselineRank(right.armId));
const agents = scoreRows
  .filter((row) => row.kind === "AGENT")
  .map((row) => scorecardRow(row, performanceByCell.get(
    `${requiredText(row, "profile")}\u0000${requiredText(row, "arm")}`,
  ) ?? null))
  .sort(cellSort);
const routers = scoreRows
  .filter((row) => row.kind === "DERIVED")
  .map((row) => scorecardRow(row, null))
  .sort(cellSort);
const rules = baselines.find((row) => row.armId === "B1-rules-v1");
const oracle = baselines.find((row) => row.armId === "B3-oracle-packet-v1");
if (!rules || !oracle) throw new Error("Scorecard lacks the B1 rules or B3 oracle row");

const pairedContrasts = readJsonl<{
  contrastId: string;
  profileKey: string;
  leftArm: string;
  rightArm: string;
  metricName: string;
  observedDifference: number;
  ciLow: number;
  ciHigh: number;
  holmAdjustedPValue: number;
  rowCount: number;
}>(`${runDirectory}/tables/paired-contrasts.jsonl`);
if (await hashFile(`${runDirectory}/tables/paired-contrasts.jsonl`) !== campaign.artifacts.contrasts.sha256) {
  throw new Error("Paired-contrast artifact differs from the campaign receipt");
}
const formalPositiveCount = pairedContrasts.filter((row) => row.ciLow > 0 && row.holmAdjustedPValue <= 0.05).length;

const ragAndToolContrasts = parseCsv(readFileSync(`${runDirectory}/tables/F06_grounding_causal_effect.csv`, "utf8"))
  .filter((row) => row.record_type === "paired_contrast")
  .map((row) => {
    const contrastId = requiredText(row, "contrast_id");
    const profileKey = profileFromContrast(contrastId);
    return {
      profileKey,
      model: modelDisplay[profileKey]?.name ?? profileKey,
      contrastId,
      metric: requiredText(row, "metric_name"),
      sampleCount: requiredNumber(row, "sample_count"),
      difference: requiredNumber(row, "observed_difference"),
      ciLow: requiredNumber(row, "ci_low"),
      ciHigh: requiredNumber(row, "ci_high"),
      holmP: requiredNumber(row, "holm_adjusted_p_value"),
    };
  });

const controls = readJsonl<{
  profile: string;
  control: string;
  pairCount: number;
  invariance: string;
  meanActionScoreDelta: number;
  semanticDecisionAgreement: { rate: number };
  orderedToolCallAgreement: { rate: number };
  receiptSha256: string;
}>(`${runDirectory}/tables/negative-controls.jsonl`).sort((left, right) =>
  modelOrder.indexOf(left.profile) - modelOrder.indexOf(right.profile) || left.control.localeCompare(right.control));

const calibration = parseCsv(readFileSync(`${runDirectory}/tables/calibration-summary.csv`, "utf8"))
  .filter((row) => row.arm !== "A-router")
  .map((row) => {
    const profileKey = requiredText(row, "profile");
    const armId = requiredText(row, "arm");
    return {
      profileKey,
      model: modelDisplay[profileKey]?.name ?? profileKey,
      armId,
      arm: armDisplay[armId] ?? armId,
      rawEce: nullableNumber(row.raw_ece),
      calibratedEce: nullableNumber(row.calibrated_ece),
      selectiveAccuracy: nullableNumber(row.selective_accuracy),
      selectiveCoverage: nullableNumber(row.selective_coverage),
      converged: row.calibration_converged === "True",
      testRows: nullableNumber(row.test_rows),
    };
  }).sort(cellSort);

const failures = parseCsv(readFileSync(`${runDirectory}/tables/F02_failure_funnel.csv`, "utf8"))
  .filter((row) => modelOrder.includes(row.profile_key ?? "") && armOrder.includes(row.agent_arm_id ?? ""))
  .map((row) => ({
    profileKey: requiredText(row, "profile_key"),
    armId: requiredText(row, "agent_arm_id"),
    failure: requiredText(row, "primary_failure"),
    episodes: requiredNumber(row, "episodes"),
  }));

const retrievalBenchmarks = parseCsv(readFileSync(`${runDirectory}/tables/F07_retrieval_scorecard.csv`, "utf8"))
  .filter((row) => row.profile_key === "retrieval-benchmark")
  .map((row) => ({
    method: requiredText(row, "agent_arm_id"),
    rows: requiredNumber(row, "rows"),
    recallAt5: nullableNumber(row.recall_at_5),
    mrr: nullableNumber(row.mrr),
    ndcgAt5: nullableNumber(row.ndcg_at_5),
    noAnswerAccuracy: nullableNumber(row.no_answer_accuracy),
  }));

const service = parseCsv(readFileSync(`${runDirectory}/tables/model-service-performance.csv`, "utf8"))
  .filter((row) => modelOrder.includes(row.model_profile_id ?? "") && (row.phase ?? "").endsWith("-residency"))
  .map((row) => {
    const profileKey = requiredText(row, "model_profile_id");
    return {
      profileKey,
      model: modelDisplay[profileKey]?.name ?? profileKey,
      promptTokens: nullableNumber(row.prompt_token_delta),
      generationTokens: nullableNumber(row.generation_token_delta),
      residencySeconds: nullableNumber(row.residency_window_seconds),
      meanGpuUtilizationPct: nullableNumber(row.mean_gpu_utilization_pct),
      maxGpuUtilizationPct: nullableNumber(row.max_gpu_utilization_pct),
      maxGpuMemoryUsedMiB: nullableNumber(row.max_gpu_memory_used_mib),
      maxGpuPowerDrawW: nullableNumber(row.max_gpu_power_draw_w),
      maxGpuTemperatureC: nullableNumber(row.max_gpu_temperature_c),
      maxRunningRequests: nullableNumber(row.max_running_requests),
      maxWaitingRequests: nullableNumber(row.max_waiting_requests),
      preemptions: nullableNumber(row.preemption_delta),
      prefixCacheHitRatio: nullableNumber(row.prefix_cache_hit_ratio),
    };
  }).sort((left, right) => modelOrder.indexOf(left.profileKey) - modelOrder.indexOf(right.profileKey));

const figureKeys = new Set([
  "F01_lift_over_rules.png",
  "F02_failure_funnel.png",
  "F05_tool_precision_recall.png",
  "F06_grounding_causal_effect.png",
  "F08_reliability_risk_coverage.png",
  "F11_latency_decomposition.png",
]);
const figures = [];
for (const item of figureManifest.figures.filter((figure) => figureKeys.has(figure.figure))) {
  const path = `${runDirectory}/figures/${item.figure}`;
  figures.push({
    key: item.figure.replace(/\.png$/, ""),
    title: figureTitle(item.figure),
    caption: item.caption,
    source: item.source,
    sourceSha256: item.sourceSha256,
    pngSha256: await hashFile(path),
    dataUri: `data:image/png;base64,${readFileSync(path).toString("base64")}`,
  });
}

const sourcePaths = [
  ...frozenTablePaths,
  "tables/paired-contrasts.jsonl",
  "metrics/campaign-analysis.json",
  "metrics/results-finalization.json",
  "telemetry/reconciliation.json",
  "repro/rows-pass.json",
  "repro/restore-pass.json",
  "database/backup-receipt.json",
  "figures/manifest.json",
];
const sourceArtifacts = [];
for (const relativePath of sourcePaths) sourceArtifacts.push({ relativePath, sha256: await hashFile(`${runDirectory}/${relativePath}`) });

const bestAgent = [...agents].sort((left, right) => (right.actionAccuracy ?? -1) - (left.actionAccuracy ?? -1))[0];
const fastestDirect = agents.filter((row) => row.armId === "A-direct")
  .sort((left, right) => (left.agentLatencyP50Ms ?? Infinity) - (right.agentLatencyP50Ms ?? Infinity))[0];
if (!bestAgent || !fastestDirect) throw new Error("Could not derive report headline cells");

const body = {
  schemaVersion: 1,
  reportId: "logwarden-tier1-model-comparison-20260823",
  runId,
  publishedFromFinalBoundaryAtUtc: backup.createdAtUtc,
  primaryAdjudication: "CLEAN_NULL",
  title: "Rules win. Retrieval helps. Autonomy does not.",
  subtitle: "A frozen SQL Server incident-triage benchmark across Muse Glimmer 30B, Gemma 4 31B, Qwen 3.8 27B, deterministic baselines, retrieval, tools, calibration controls, and the full vLLM/agent pipeline.",
  summary: {
    corpusEpisodes: 600,
    predictions: campaign.input.predictions,
    metricRows: campaign.output.metricRows,
    formalContrasts: campaign.output.contrastRows,
    formalPositiveCount,
    bootstrapReplicates: campaign.replicates.bootstrap,
    permutationReplicates: campaign.replicates.permutation,
    completedModels: campaign.input.analyzedProfiles.length,
    targetModels: campaign.input.analyzedProfiles.length + campaign.input.portExclusions.length,
    rulesActionAccuracy: rules.actionAccuracy,
    rulesEndToEndSuccess: rules.endToEndSuccess,
    bestAgent: { profileKey: bestAgent.profileKey, model: bestAgent.model, arm: bestAgent.arm, actionAccuracy: bestAgent.actionAccuracy },
    fastestDirect: { profileKey: fastestDirect.profileKey, model: fastestDirect.model, latencyP50Ms: fastestDirect.agentLatencyP50Ms },
    telemetry: telemetry.observed,
    taxonomyRows: finalization.taxonomyRows,
    claimRows: finalization.claimRows,
    reportSnapshots: finalization.reportSnapshots,
  },
  models: modelOrder.map((profileKey) => ({ profileKey, ...modelDisplay[profileKey]! })),
  baselines,
  rules,
  oracle,
  agents,
  routers,
  ragAndToolContrasts,
  formalContrasts: pairedContrasts,
  controls,
  calibration,
  failures,
  retrievalBenchmarks,
  service,
  olmo: campaign.input.portExclusions[0] ?? null,
  figures,
  receipts: {
    campaignAnalysis: campaign.receiptSha256,
    telemetryReconciliation: telemetry.receiptSha256,
    resultsFinalization: finalization.receiptSha256,
    rowReproduction: rowPass.receiptSha256,
    restoreReproduction: restorePass.receiptSha256,
    finalBackup: backup.receiptSha256,
    expectedOutputs: expected.receiptSha256,
    figureManifest: figureManifest.receiptSha256,
  },
  caveats: [
    "The primary result is CLEAN_NULL: a passing benchmark is not evidence that an LLM beat the deterministic rules baseline.",
    "B3 is an evaluator-only oracle packet ceiling and is never treated as a deployable arm.",
    "OLMo is STOP_PORT after two unconstrained structured-contract gates; its separate guided-decoding diagnostic is Tier 2 and excluded here.",
    "Holm-adjusted p-values are 1 under the declared permutation family. The report preserves the positive-claim gate and does not promote descriptive lifts.",
    "Live replay parity, event-rate capacity, raw-event correlation, Query Store intervals, ANN, and production safety remain outside Tier 1.",
    "True TTFT, inter-token latency, vLLM throughput gauges, and KV occupancy were unavailable from the retained interfaces and are not replaced with proxies.",
  ],
  sourceArtifacts,
};
const data = { ...body, receiptSha256: hashJson(body) };

const template = readFileSync(`${LAB_ROOT}/templates/tier1-model-comparison.template.html`, "utf8");
if (template.split("__DATA_JSON__").length !== 2) throw new Error("Tier-1 report template must contain exactly one data placeholder");
const embedded = JSON.stringify(data, null, 1).replace(/<\/script/gi, "<\\/script");
const html = template.replace("__DATA_JSON__", embedded);
await atomicWrite(outputPath, html);
await atomicWrite(dataPath, `${JSON.stringify(data, null, 2)}\n`);

console.log(JSON.stringify({
  runId,
  outputPath,
  dataPath,
  reportReceiptSha256: data.receiptSha256,
  cells: agents.length,
  baselines: baselines.length,
  figures: figures.length,
  formalPositiveCount,
}, null, 2));

function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

function readJsonl<T>(path: string): T[] {
  return readFileSync(path, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line) as T);
}

function verifyReceipt(value: JsonObject & { receiptSha256: string }, label: string): void {
  const { receiptSha256, ...bodyValue } = value;
  if (!/^[0-9a-f]{64}$/.test(receiptSha256) || hashJson(bodyValue) !== receiptSha256) {
    throw new Error(`${label} receipt validation failed`);
  }
}

function parseCsv(value: string): CsvRow[] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index]!;
    if (quoted) {
      if (character === '"' && value[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') quoted = false;
      else field += character;
    } else if (character === '"') quoted = true;
    else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else field += character;
  }
  if (quoted) throw new Error("Unterminated quoted CSV field");
  if (field.length > 0 || row.length > 0) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const header = rows.shift();
  if (!header || new Set(header).size !== header.length) throw new Error("CSV header is missing or duplicated");
  return rows.filter((fields) => fields.some((item) => item !== "")).map((fields, rowIndex) => {
    if (fields.length !== header.length) throw new Error(`CSV row ${rowIndex + 2} has ${fields.length} fields; expected ${header.length}`);
    return Object.fromEntries(header.map((name, index) => [name!, fields[index]!])) as CsvRow;
  });
}

function nullableNumber(value: string | undefined): number | null {
  if (value === undefined || value === "") return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`Expected a finite number, received ${value}`);
  return parsed;
}

function requiredNumber(row: CsvRow, key: string): number {
  const value = nullableNumber(row[key]);
  if (value === null) throw new Error(`Missing required numeric field ${key}`);
  return value;
}

function requiredText(row: CsvRow, key: string): string {
  const value = row[key];
  if (value === undefined || value === "") throw new Error(`Missing required text field ${key}`);
  return value;
}

function scorecardRow(row: CsvRow, performance: CsvRow | null): Record<string, unknown> & {
  profileKey: string;
  model: string;
  armId: string;
  arm: string;
  actionAccuracy: number | null;
  endToEndSuccess: number | null;
  agentLatencyP50Ms: number | null;
} {
  const profileKey = requiredText(row, "profile");
  const armId = requiredText(row, "arm");
  const meta = modelDisplay[profileKey];
  return {
    profileKey,
    model: meta?.name ?? (profileKey === "baseline" ? "Baseline" : profileKey),
    modelShort: meta?.short ?? "Baseline",
    slot: meta?.slot ?? 0,
    armId,
    arm: armDisplay[armId] ?? armId,
    kind: requiredText(row, "kind"),
    disposition: requiredText(row, "disposition"),
    eligible: nullableNumber(row.eligible),
    completed: nullableNumber(row.completed),
    endToEndSuccess: nullableNumber(row.end_to_end_success),
    macroF1: nullableNumber(row.macro_f1),
    actionAccuracy: nullableNumber(row.action_accuracy),
    costWeightedLoss: nullableNumber(row.cost_weighted_loss),
    contractSuccess: nullableNumber(row.contract_success),
    requiredToolRecall: nullableNumber(row.required_tool_recall),
    forbiddenToolRate: nullableNumber(row.forbidden_tool_rate),
    argumentValidRate: nullableNumber(row.argument_valid_rate),
    snapshotMissRate: nullableNumber(row.snapshot_miss_rate),
    retrievalRecallAt5: nullableNumber(row.retrieval_recall_at_5),
    groundingRate: nullableNumber(row.grounding_rate),
    firstPassValidRate: nullableNumber(row.first_pass_valid_rate),
    repairedResponseRate: nullableNumber(row.repaired_response_rate),
    rejectedResponseRate: nullableNumber(row.rejected_response_rate),
    actionDeltaVsB1: nullableNumber(row.action_delta_vs_b1),
    actionDeltaCiLow: nullableNumber(row.action_delta_ci_low),
    actionDeltaCiHigh: nullableNumber(row.action_delta_ci_high),
    actionDeltaHolmP: nullableNumber(row.action_delta_holm_p),
    rawEce: nullableNumber(row.raw_ece),
    calibratedEce: nullableNumber(row.calibrated_ece),
    selectiveAccuracy: nullableNumber(row.selective_accuracy),
    selectiveCoverage: nullableNumber(row.selective_coverage),
    calibrationConverged: row.calibration_converged === "True",
    agentLatencyP50Ms: nullableNumber(performance?.agent_p50_ms ?? row.agent_latency_p50_ms),
    agentLatencyP90Ms: nullableNumber(performance?.agent_p90_ms),
    agentLatencyP95Ms: nullableNumber(performance?.agent_p95_ms ?? row.agent_latency_p95_ms),
    agentLatencyP99Ms: nullableNumber(performance?.agent_p99_ms),
    modelLatencyP50Ms: nullableNumber(performance?.model_client_p50_ms ?? row.model_latency_p50_ms),
    modelLatencyP95Ms: nullableNumber(performance?.model_client_p95_ms ?? row.model_latency_p95_ms),
    modelRequests: nullableNumber(performance?.model_requests),
    promptTokens: nullableNumber(performance?.prompt_tokens),
    completionTokens: nullableNumber(performance?.completion_tokens),
    tokensPerEpisode: nullableNumber(performance?.tokens_per_episode ?? row.tokens_per_episode),
    modelErrors: nullableNumber(performance?.model_errors),
    lengthFinishes: nullableNumber(performance?.length_finishes),
    validationFailures: nullableNumber(performance?.validation_failures),
    taxonomyLabels: row.taxonomy_labels ? row.taxonomy_labels.split(";") : [],
  };
}

function baselineRank(armId: string): number {
  const order = ["B0-majority-no-action-v1", "B1-rules-v1", "B2-lexical-v1", "B2-vector-v1", "B2-hybrid-v1", "B3-oracle-packet-v1"];
  const rank = order.indexOf(armId);
  return rank < 0 ? order.length : rank;
}

function cellSort(left: { profileKey: string; armId: string }, right: { profileKey: string; armId: string }): number {
  return modelOrder.indexOf(left.profileKey) - modelOrder.indexOf(right.profileKey)
    || armOrder.indexOf(left.armId) - armOrder.indexOf(right.armId);
}

function profileFromContrast(contrastId: string): string {
  return modelOrder.find((profile) => contrastId.startsWith(`${profile}-`)) ?? contrastId.split("-A-")[0] ?? contrastId;
}

function figureTitle(fileName: string): string {
  const names: Record<string, string> = {
    "F01_lift_over_rules.png": "Lift over deterministic rules",
    "F02_failure_funnel.png": "Where agent episodes fail",
    "F05_tool_precision_recall.png": "Tool precision and recall",
    "F06_grounding_causal_effect.png": "Retrieval and tool-context effects",
    "F08_reliability_risk_coverage.png": "Calibration and selective risk",
    "F11_latency_decomposition.png": "End-to-end latency decomposition",
  };
  return names[fileName] ?? fileName;
}
