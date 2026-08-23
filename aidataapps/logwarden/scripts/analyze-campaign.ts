import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadStandardCampaign } from "../src/campaign.js";
import { loadConfig } from "../src/config.js";
import { loadTier1ControlPolicy } from "../src/controls.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { scorePrediction, type PredictionScore, type ScorePrediction, type ScoreToolInvocation, type ScoreTruth } from "../src/scoring.js";
import {
  holmAdjustedPValues,
  pairedGroupedBootstrap,
  stableStatisticsSeed,
  withinStratumGroupLabelPermutation,
} from "../src/statistics.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";

interface AnalysisRow {
  predictionId: number;
  episodeId: string;
  profileKey: string | null;
  armId: string;
  predictionSource: string;
  splitRole: string;
  family: string;
  regime: string;
  scenarioGroupId: string;
  outcome: "decision" | "abstention" | "failure";
  confidence: number | null;
  score: PredictionScore;
  truth: ScoreTruth;
  prediction: ScorePrediction;
  observed: {
    toolInvocations: ScoreToolInvocation[];
    returnedRunbooks: string[];
    citedRunbooks: string[];
  };
  scoringArmId: string;
}

interface QueryRow {
  prediction_id: string;
  episode_id: string;
  model_profile_id: string | null;
  agent_arm_id: string;
  prediction_source: string;
  outcome: AnalysisRow["outcome"];
  confidence: number | null;
  split_role: string;
  family: string;
  regime: string;
  scenario_group_id: string;
  detail_json: string;
}

interface MetricDefinition {
  name: string;
  analysisSet: "end_to_end" | "complete_case";
  select: (row: AnalysisRow) => boolean;
  value: (row: AnalysisRow) => number;
}

interface MetricOutput extends Record<string, unknown> {
  metricName: string;
  profileKey: string | null;
  armId: string;
  splitRole: string | null;
  family: string | null;
  regime: string | null;
  analysisSet: string;
  numerator: number;
  denominator: number;
  eligibleDenominator: number;
  completeCaseDenominator: number;
  metricValue: number;
  standardError: number;
}

type ContrastMetric = "acceptable_action_accuracy" | "cost_weighted_loss_reduction" | "end_to_end_success";

interface ContrastPlan {
  contrastId: string;
  hypothesisClass: string;
  profileKey: string;
  leftArm: string;
  rightArm: string;
  metric: ContrastMetric;
  eligibility: "all_common" | "retrieval_covered" | "tool_required";
}

interface ContrastOutput extends Record<string, unknown> {
  contrastId: string;
  hypothesisClass: string;
  metricName: ContrastMetric;
  profileKey: string;
  leftArm: string;
  rightArm: string;
  rowCount: number;
  groupCount: number;
  observedDifference: number;
  ciLow: number;
  ciHigh: number;
  permutationPValue: number;
  holmAdjustedPValue?: number;
  positiveDirection: "higher_is_better";
  bootstrap: Omit<ReturnType<typeof pairedGroupedBootstrap>, "replicatesValues">;
  permutation: Omit<ReturnType<typeof withinStratumGroupLabelPermutation<ScoreTruth>>, "nullValues">;
  bootstrapValues: number[];
  nullValues: number[];
}

const startedAtUtc = new Date().toISOString();
const campaign = loadStandardCampaign();
const controlPolicy = loadTier1ControlPolicy();
const roles = listArgument("--roles", "test_id,test_variant_holdout,test_unknown");
const bootstrapReplicates = integerArgument("--bootstrap-replicates", campaign.statistics.bootstrapReplicates, 1_000, 100_000);
const permutationReplicates = integerArgument("--permutation-replicates", campaign.statistics.permutationReplicates, 1_000, 100_000);
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const freeze = await validatedFreeze();
const freezeHash = String(freeze.freezeHash);
const bootstrapBaseSeed = Number.parseInt(sha256(`${freezeHash}\0bootstrap-v1`).slice(0, 8), 16) >>> 0;
const permutationBaseSeed = controlPolicy.controls["label-permutation-v1"].seed >>> 0;
const pool = await connect(config.databases.lab, config.databases.controlName, 600_000);

try {
  await assertAuthorized();
  const rows = await loadRows();
  assertCoverage(rows);
  const metricRows = buildMetricRows(rows);
  let insertedMetrics = 0;
  for (const row of metricRows) insertedMetrics += await persistMetric(row);
  const contrasts = await analyzeContrasts(rows);
  const adjusted = holmAdjustedPValues(contrasts.map((row) => ({ id: `${row.contrastId}\0${row.metricName}`, pValue: row.permutationPValue })));
  for (const contrast of contrasts) contrast.holmAdjustedPValue = adjusted[`${contrast.contrastId}\0${contrast.metricName}`]!;
  let insertedBootstraps = 0;
  let insertedPermutations = 0;
  for (const contrast of contrasts) {
    insertedBootstraps += await persistBootstrap(contrast);
    insertedPermutations += await persistPermutation(contrast);
  }

  const metricTable = `${metricRows.map((row) => canonicalJson(row)).join("\n")}\n`;
  const contrastTable = `${contrasts.map(({ bootstrapValues: _bootstrap, nullValues: _null, ...row }) => canonicalJson(row)).join("\n")}\n`;
  const bootstrapTable = distributionTable(contrasts, "bootstrapValues", "bootstrap");
  const permutationTable = distributionTable(contrasts, "nullValues", "permutation");
  const paths = {
    metrics: `${runDirectory}/tables/metric-results.jsonl`,
    contrasts: `${runDirectory}/tables/paired-contrasts.jsonl`,
    bootstrapReplicates: `${runDirectory}/tables/bootstrap-replicates.jsonl`,
    permutationReplicates: `${runDirectory}/tables/permutation-replicates.jsonl`,
  };
  await Promise.all([
    atomicWrite(paths.metrics, metricTable, 0o600),
    atomicWrite(paths.contrasts, contrastTable, 0o600),
    atomicWrite(paths.bootstrapReplicates, bootstrapTable, 0o600),
    atomicWrite(paths.permutationReplicates, permutationTable, 0o600),
  ]);
  const artifacts = Object.fromEntries(Object.entries({
    metrics: metricTable, contrasts: contrastTable, bootstrapReplicates: bootstrapTable, permutationReplicates: permutationTable,
  }).map(([key, body]) => [key, { path: paths[key as keyof typeof paths], bytes: Buffer.byteLength(body), sha256: sha256(body) }]));
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    freezeHash,
    roles,
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    algorithms: {
      bootstrap: "paired-cluster-percentile-bootstrap-v1",
      permutation: "within-stratum-group-label-permutation-v1",
      permutationStrata: controlPolicy.controls["label-permutation-v1"].within,
      groupingUnit: campaign.statistics.resamplingGroup,
      multiplicity: `Holm over ${contrasts.length} materialized primary cell/metric tests`,
    },
    seeds: { bootstrapBaseSeed, permutationBaseSeed },
    replicates: { bootstrap: bootstrapReplicates, permutation: permutationReplicates },
    input: { predictions: rows.length, orderedPredictionSetSha256: hashJson(rows.map((row) => ({ predictionId: row.predictionId, score: row.score }))) },
    output: { metricRows: metricRows.length, contrastRows: contrasts.length, insertedMetrics, insertedBootstraps, insertedPermutations },
    artifacts,
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const receiptPath = `${runDirectory}/metrics/campaign-analysis.json`;
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  await appendExperimentLog(`Materialized ${metricRows.length} metric rows and ${contrasts.length} paired grouped contrasts with ${bootstrapReplicates} bootstraps and ${permutationReplicates} within-stratum group-label permutations; Holm family=${contrasts.length}; receipt ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({ roles, predictions: rows.length, metricRows: metricRows.length, contrasts: contrasts.length, insertedMetrics, insertedBootstraps, insertedPermutations, receiptPath, receiptSha256: receipt.receiptSha256, disposition: "PASS" }, null, 2));
} finally {
  await pool.close();
}

async function assertAuthorized(): Promise<void> {
  if (freeze.disposition !== "FROZEN") throw new Error("Campaign analysis requires the immutable campaign freeze");
  const result = await pool.request().input("run", sql.VarChar(120), run.runId).query<{ status: string }>(`
    SELECT campaign.status FROM control.campaigns campaign INNER JOIN control.runs run ON run.campaign_id=campaign.campaign_id WHERE run.run_id=@run;
  `);
  if (!["running", "complete"].includes(result.recordset[0]?.status ?? "")) throw new Error("Campaign analysis requires a running or complete campaign");
  if (canonicalJson([...roles].sort()) !== canonicalJson(["test_id", "test_unknown", "test_variant_holdout"])) {
    throw new Error("Primary campaign analysis requires every frozen test role and excludes calibration/dev");
  }
}

async function loadRows(): Promise<AnalysisRow[]> {
  const arms = [...new Set([...campaign.mandatoryInferenceArms, campaign.retrievalArm.armId, ...campaign.derivedArms, ...campaign.baselines])];
  const result = await pool.request().input("run", sql.VarChar(120), run.runId)
    .input("roles", sql.NVarChar(sql.MAX), JSON.stringify(roles)).input("arms", sql.NVarChar(sql.MAX), JSON.stringify(arms))
    .input("profiles", sql.NVarChar(sql.MAX), JSON.stringify(campaign.targetProfiles)).query<QueryRow>(`
      WITH selected_roles AS (SELECT CONVERT(varchar(40),value) value FROM OPENJSON(@roles)),
           selected_arms AS (SELECT CONVERT(varchar(80),value) value FROM OPENJSON(@arms)),
           selected_profiles AS (SELECT CONVERT(varchar(80),value) value FROM OPENJSON(@profiles)),
           current_campaign AS
           (
             SELECT campaign_id FROM control.runs WHERE run_id=@run
           )
      SELECT prediction.prediction_id,prediction.episode_id,prediction.model_profile_id,prediction.agent_arm_id,
        prediction.prediction_source,prediction.outcome,prediction.confidence,truth.split_role,truth.family,truth.regime,
        truth.scenario_group_id,score.detail_json
      FROM current_campaign INNER JOIN control.jobs job ON job.campaign_id=current_campaign.campaign_id
      INNER JOIN eval.predictions prediction ON prediction.job_id=job.job_id
      INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=prediction.episode_id
      INNER JOIN eval.decision_scores score ON score.prediction_id=prediction.prediction_id
      INNER JOIN selected_roles role ON role.value=truth.split_role
      INNER JOIN selected_arms arm ON arm.value=prediction.agent_arm_id
      LEFT JOIN selected_profiles profile ON profile.value=prediction.model_profile_id
      WHERE prediction.control_id IS NULL AND (prediction.model_profile_id IS NULL OR profile.value IS NOT NULL)
      ORDER BY prediction.model_profile_id,prediction.agent_arm_id,truth.split_role,prediction.episode_id;
    `);
  return result.recordset.map(parseAnalysisRow);
}

function parseAnalysisRow(row: QueryRow): AnalysisRow {
  const detail = JSON.parse(row.detail_json) as Record<string, unknown>;
  const score = detail.score as PredictionScore | undefined;
  const truth = detail.truth as ScoreTruth | undefined;
  const prediction = detail.prediction as ScorePrediction | undefined;
  const observed = detail.observed as AnalysisRow["observed"] | undefined;
  const scoringArmId = detail.scoringArmId;
  if (score === undefined || truth === undefined || prediction === undefined || observed === undefined || typeof scoringArmId !== "string") {
    throw new Error(`Prediction ${row.prediction_id} lacks recomputable score detail; re-run campaign:score with the frozen scorer`);
  }
  return {
    predictionId: Number(row.prediction_id), episodeId: row.episode_id, profileKey: row.model_profile_id,
    armId: row.agent_arm_id, predictionSource: row.prediction_source, splitRole: row.split_role,
    family: row.family, regime: row.regime, scenarioGroupId: row.scenario_group_id,
    outcome: row.outcome, confidence: row.confidence === null ? null : Number(row.confidence),
    score, truth, prediction, observed, scoringArmId,
  };
}

function assertCoverage(rows: AnalysisRow[]): void {
  for (const profile of campaign.targetProfiles) {
    for (const arm of [...campaign.mandatoryInferenceArms, ...campaign.derivedArms]) {
      const count = rows.filter((row) => row.profileKey === profile && row.armId === arm).length;
      if (count !== 480) throw new Error(`Primary analysis found ${count}/480 rows for ${profile}/${arm}`);
    }
    const rag = rows.filter((row) => row.profileKey === profile && row.armId === campaign.retrievalArm.armId).length;
    if (rag === 0) throw new Error(`Primary analysis found no retrieval-covered rows for ${profile}`);
  }
  for (const arm of campaign.baselines) {
    const count = rows.filter((row) => row.profileKey === null && row.armId === arm).length;
    if (count !== 480) throw new Error(`Primary analysis found ${count}/480 rows for baseline ${arm}`);
  }
}

const metricDefinitions: MetricDefinition[] = [
  booleanMetric("end_to_end_success", (row) => row.score.endToEndSuccess),
  booleanMetric("class_accuracy", (row) => row.score.classCorrect),
  booleanMetric("severity_accuracy", (row) => row.score.severityCorrect),
  booleanMetric("acceptable_action_accuracy", (row) => row.score.actionCorrect),
  booleanMetric("abstention_accuracy", (row) => row.score.abstentionCorrect),
  booleanMetric("contract_success", (row) => row.score.contractSuccess),
  booleanMetric("required_tools_satisfied", (row) => row.score.requiredToolsSatisfied),
  booleanMetric("forbidden_tools_avoided", (row) => row.score.forbiddenToolsAvoided),
  booleanMetric("citation_policy_satisfied", (row) => row.score.citationPolicySatisfied),
  { name: "cost_weighted_loss", analysisSet: "end_to_end", select: () => true, value: (row) => row.score.costWeightedLoss },
  { name: "severity_ordinal_cost", analysisSet: "end_to_end", select: () => true, value: (row) => row.score.severityOrdinalCost },
  { name: "composite_score", analysisSet: "end_to_end", select: () => true, value: (row) => row.score.compositeScore },
  { name: "terminal_failure_rate", analysisSet: "end_to_end", select: () => true, value: (row) => Number(row.outcome === "failure") },
  { name: "coverage", analysisSet: "end_to_end", select: () => true, value: (row) => Number(row.outcome !== "failure" && row.prediction.abstained === false) },
  { name: "acceptable_action_accuracy_complete_case", analysisSet: "complete_case", select: (row) => row.outcome !== "failure", value: (row) => Number(row.score.actionCorrect) },
];

function booleanMetric(name: string, value: (row: AnalysisRow) => boolean): MetricDefinition {
  return { name, analysisSet: "end_to_end", select: () => true, value: (row) => Number(value(row)) };
}

function buildMetricRows(rows: AnalysisRow[]): MetricOutput[] {
  const cells = groupBy(rows, (row) => `${row.profileKey ?? "baseline"}\0${row.armId}`);
  const output: MetricOutput[] = [];
  for (const values of cells.values()) {
    const first = values[0]!;
    const strata: Array<{ splitRole: string | null; family: string | null; regime: string | null; rows: AnalysisRow[] }> = [
      { splitRole: null, family: null, regime: null, rows: values },
      ...unique(values.map((row) => row.splitRole)).map((splitRole) => ({ splitRole, family: null, regime: null, rows: values.filter((row) => row.splitRole === splitRole) })),
      ...unique(values.map((row) => row.family)).map((family) => ({ splitRole: null, family, regime: null, rows: values.filter((row) => row.family === family) })),
      ...unique(values.map((row) => row.regime)).map((regime) => ({ splitRole: null, family: null, regime, rows: values.filter((row) => row.regime === regime) })),
    ];
    for (const stratum of strata) for (const metric of metricDefinitions) {
      const selected = stratum.rows.filter(metric.select);
      if (selected.length === 0) continue;
      const metricValues = selected.map(metric.value);
      const numerator = sum(metricValues);
      const metricValue = numerator / selected.length;
      output.push({
        schemaVersion: 1, runId: run.runId, metricName: metric.name, profileKey: first.profileKey, armId: first.armId,
        splitRole: stratum.splitRole, family: stratum.family, regime: stratum.regime, analysisSet: metric.analysisSet,
        numerator: round(numerator), denominator: selected.length, eligibleDenominator: stratum.rows.length,
        completeCaseDenominator: stratum.rows.filter((row) => row.outcome !== "failure").length,
        metricValue: round(metricValue), standardError: standardError(metricValues),
        inputSetSha256: hashJson(selected.map((row) => ({ predictionId: row.predictionId, value: metric.value(row) }))),
      });
    }
  }
  return output.sort(metricSort);
}

async function analyzeContrasts(rows: AnalysisRow[]): Promise<ContrastOutput[]> {
  const plans = contrastPlans();
  const byCell = groupBy(rows, (row) => `${row.profileKey ?? "baseline"}\0${row.armId}`);
  const output: ContrastOutput[] = [];
  for (const plan of plans) {
    const left = byCell.get(`${plan.profileKey}\0${plan.leftArm}`) ?? [];
    const rightProfile = plan.rightArm === "B1-rules-v1" ? "baseline" : plan.profileKey;
    const right = new Map((byCell.get(`${rightProfile}\0${plan.rightArm}`) ?? []).map((row) => [row.episodeId, row]));
    const pairs = left.map((leftRow) => ({ left: leftRow, right: right.get(leftRow.episodeId) })).filter((pair): pair is { left: AnalysisRow; right: AnalysisRow } => pair.right !== undefined)
      .filter((pair) => eligiblePair(pair.left, plan.eligibility));
    if (pairs.length < 20 || new Set(pairs.map((pair) => pair.left.scenarioGroupId)).size < 2) throw new Error(`Contrast ${plan.contrastId}/${plan.metric} has insufficient paired rows`);
    const identity = `${plan.contrastId}\0${plan.metric}`;
    const bootstrap = pairedGroupedBootstrap(pairs.map((pair) => ({
      groupId: pair.left.scenarioGroupId, stratum: `${pair.left.family}|${pair.left.splitRole}`,
      left: contrastValue(pair.left, plan.metric), right: contrastValue(pair.right, plan.metric),
    })), {
      seed: stableStatisticsSeed(bootstrapBaseSeed, identity), replicates: bootstrapReplicates,
      confidenceLevel: 1 - campaign.statistics.alpha / campaign.statistics.primaryHypothesisFamilySize,
    });
    const labels = pairs.map((pair) => ({
      rowId: pair.left.episodeId, groupId: pair.left.scenarioGroupId,
      stratum: `${pair.left.family}|${pair.left.splitRole}`, label: pair.left.truth,
    }));
    const permutation = withinStratumGroupLabelPermutation(labels, (permuted) => mean(pairs.map((pair) => {
      const truth = permuted.get(pair.left.episodeId)!;
      return contrastValue(pair.left, plan.metric, truth) - contrastValue(pair.right, plan.metric, truth);
    })), {
      seed: stableStatisticsSeed(permutationBaseSeed, identity), replicates: permutationReplicates,
      observedValue: bootstrap.observedDifference,
    });
    const { replicatesValues, ...bootstrapSummary } = bootstrap;
    const { nullValues, ...permutationSummary } = permutation;
    output.push({
      schemaVersion: 1, runId: run.runId, ...plan, metricName: plan.metric,
      rowCount: pairs.length, groupCount: bootstrap.groupCount, observedDifference: bootstrap.observedDifference,
      ciLow: bootstrap.ciLow, ciHigh: bootstrap.ciHigh, permutationPValue: permutation.pValueTwoSided,
      positiveDirection: "higher_is_better", bootstrap: bootstrapSummary, permutation: permutationSummary,
      bootstrapValues: replicatesValues, nullValues,
      orderedPairSetSha256: hashJson(pairs.map((pair) => ({ left: pair.left.predictionId, right: pair.right.predictionId }))),
    });
  }
  return output;
}

function contrastPlans(): ContrastPlan[] {
  const plans: ContrastPlan[] = [];
  for (const profileKey of campaign.targetProfiles) {
    for (const arm of [...campaign.mandatoryInferenceArms, campaign.retrievalArm.armId, ...campaign.derivedArms]) {
      for (const metric of ["acceptable_action_accuracy", "cost_weighted_loss_reduction"] as const) plans.push({
        contrastId: `${profileKey}-${arm}-vs-B1`, hypothesisClass: "model_arm_minus_rules", profileKey,
        leftArm: arm, rightArm: "B1-rules-v1", metric, eligibility: "all_common",
      });
    }
    for (const metric of ["acceptable_action_accuracy", "end_to_end_success"] as const) {
      plans.push({ contrastId: `${profileKey}-A-rag-vs-A-direct`, hypothesisClass: "retrieval_ablation", profileKey,
        leftArm: "A-rag", rightArm: "A-direct", metric, eligibility: "retrieval_covered" });
      plans.push({ contrastId: `${profileKey}-A-tools-vs-A-rag`, hypothesisClass: "tool_context_ablation", profileKey,
        leftArm: "A-tools", rightArm: "A-rag", metric, eligibility: "tool_required" });
    }
  }
  return plans;
}

function eligiblePair(row: AnalysisRow, eligibility: ContrastPlan["eligibility"]): boolean {
  if (eligibility === "all_common") return true;
  if (eligibility === "retrieval_covered") return row.truth.expectedRunbooks.length > 0;
  return row.truth.requiredTools.length > 0;
}

function contrastValue(row: AnalysisRow, metric: ContrastMetric, truth = row.truth): number {
  const score = truth === row.truth ? row.score : scorePrediction({
    truth, prediction: row.prediction, armId: row.scoringArmId, tools: row.observed.toolInvocations,
    citedRunbooks: row.observed.citedRunbooks, returnedRunbooks: row.observed.returnedRunbooks,
  });
  if (metric === "acceptable_action_accuracy") return Number(score.actionCorrect);
  if (metric === "end_to_end_success") return Number(score.endToEndSuccess);
  return -score.costWeightedLoss;
}

async function persistMetric(row: MetricOutput): Promise<number> {
  const detailJson = canonicalJson(row);
  const request = pool.request().input("run", sql.VarChar(120), run.runId).input("metric", sql.VarChar(160), row.metricName)
    .input("profile", sql.VarChar(80), row.profileKey).input("arm", sql.VarChar(80), row.armId)
    .input("split", sql.VarChar(40), row.splitRole).input("family", sql.VarChar(80), row.family)
    .input("regime", sql.Char(1), row.regime).input("set", sql.VarChar(32), row.analysisSet);
  const prior = await request.query<{ detail_json: string }>(`
    SELECT detail_json FROM eval.metric_results WHERE run_id=@run AND metric_name=@metric AND agent_arm_id=@arm
      AND ((model_profile_id=@profile) OR (model_profile_id IS NULL AND @profile IS NULL))
      AND ((split_role=@split) OR (split_role IS NULL AND @split IS NULL))
      AND ((family=@family) OR (family IS NULL AND @family IS NULL))
      AND ((regime=@regime) OR (regime IS NULL AND @regime IS NULL)) AND analysis_set=@set;
  `);
  if (prior.recordset.length > 1) throw new Error(`Duplicate metric identity ${row.metricName}/${row.profileKey}/${row.armId}`);
  if (prior.recordset[0] !== undefined) {
    if (canonicalJson(JSON.parse(prior.recordset[0].detail_json)) !== detailJson) throw new Error(`Metric drift ${row.metricName}/${row.profileKey}/${row.armId}`);
    return 0;
  }
  await pool.request().input("run", sql.VarChar(120), run.runId).input("metric", sql.VarChar(160), row.metricName)
    .input("profile", sql.VarChar(80), row.profileKey).input("arm", sql.VarChar(80), row.armId)
    .input("split", sql.VarChar(40), row.splitRole).input("family", sql.VarChar(80), row.family)
    .input("regime", sql.Char(1), row.regime).input("set", sql.VarChar(32), row.analysisSet)
    .input("numerator", sql.Float, row.numerator).input("denominator", sql.BigInt, row.denominator)
    .input("eligible", sql.BigInt, row.eligibleDenominator).input("complete", sql.BigInt, row.completeCaseDenominator)
    .input("value", sql.Float, row.metricValue).input("se", sql.Float, row.standardError)
    .input("detail", sql.NVarChar(sql.MAX), detailJson).query(`
      INSERT eval.metric_results(run_id,metric_name,model_profile_id,agent_arm_id,split_role,family,regime,
        analysis_set,numerator,denominator,eligible_denominator,complete_case_denominator,metric_value,standard_error,detail_json)
      VALUES(@run,@metric,@profile,@arm,@split,@family,@regime,@set,@numerator,@denominator,@eligible,@complete,@value,@se,@detail);
    `);
  return 1;
}

async function persistBootstrap(row: ContrastOutput): Promise<number> {
  const summary = { ...row.bootstrap, holmAdjustedPValue: row.holmAdjustedPValue, permutationPValue: row.permutationPValue,
    distributionArtifact: "tables/bootstrap-replicates.jsonl" };
  const resultSha256 = hashJson({ contrastId: row.contrastId, metricName: row.metricName, profileKey: row.profileKey, summary });
  const prior = await pool.request().input("run", sql.VarChar(120), run.runId).input("contrast", sql.VarChar(160), row.contrastId)
    .input("metric", sql.VarChar(160), row.metricName).query<{ result_sha256: string }>(`
      SELECT result_sha256 FROM eval.bootstrap_results WHERE run_id=@run AND contrast_id=@contrast AND metric_name=@metric;
    `);
  if (prior.recordset[0] !== undefined) {
    if (prior.recordset[0].result_sha256 !== resultSha256) throw new Error(`Bootstrap drift ${row.contrastId}/${row.metricName}`);
    return 0;
  }
  await pool.request().input("run", sql.VarChar(120), run.runId).input("contrast", sql.VarChar(160), row.contrastId)
    .input("metric", sql.VarChar(160), row.metricName).input("left", sql.NVarChar(sql.MAX), canonicalJson({ profileKey: row.profileKey, armId: row.leftArm }))
    .input("right", sql.NVarChar(sql.MAX), canonicalJson({ profileKey: row.rightArm === "B1-rules-v1" ? null : row.profileKey, armId: row.rightArm }))
    .input("reps", sql.Int, row.bootstrap.replicates).input("seed", sql.BigInt, row.bootstrap.seed)
    .input("observed", sql.Float, row.observedDifference).input("confidence", sql.Float, row.bootstrap.confidenceLevel)
    .input("low", sql.Float, row.ciLow).input("high", sql.Float, row.ciHigh).input("p", sql.Float, row.permutationPValue)
    .input("count", sql.Int, row.rowCount).input("hash", sql.Char(64), resultSha256).query(`
      INSERT eval.bootstrap_results(run_id,contrast_id,metric_name,left_cell_json,right_cell_json,grouped_by,paired,
        repetitions,seed,observed_difference,confidence_level,ci_low,ci_high,p_value,sample_count,result_sha256)
      VALUES(@run,@contrast,@metric,@left,@right,'scenario_group_id',1,@reps,@seed,@observed,@confidence,@low,@high,@p,@count,@hash);
    `);
  return 1;
}

async function persistPermutation(row: ContrastOutput): Promise<number> {
  const detail = canonicalJson({ ...row.permutation, holmAdjustedPValue: row.holmAdjustedPValue,
    hypothesisClass: row.hypothesisClass, distributionArtifact: "tables/permutation-replicates.jsonl" });
  const prior = await pool.request().input("run", sql.VarChar(120), run.runId).input("id", sql.VarChar(160), row.contrastId)
    .input("metric", sql.VarChar(160), row.metricName).query<{ detail_json: string }>(`
      SELECT detail_json FROM eval.permutation_results WHERE run_id=@run AND permutation_id=@id AND metric_name=@metric;
    `);
  if (prior.recordset[0] !== undefined) {
    if (canonicalJson(JSON.parse(prior.recordset[0].detail_json)) !== detail) throw new Error(`Permutation drift ${row.contrastId}/${row.metricName}`);
    return 0;
  }
  await pool.request().input("run", sql.VarChar(120), run.runId).input("id", sql.VarChar(160), row.contrastId)
    .input("metric", sql.VarChar(160), row.metricName).input("reps", sql.Int, row.permutation.replicates)
    .input("seed", sql.BigInt, row.permutation.seed).input("observed", sql.Float, row.observedDifference)
    .input("mean", sql.Float, row.permutation.nullMean).input("sd", sql.Float, row.permutation.nullSd)
    .input("p", sql.Float, row.permutationPValue).input("detail", sql.NVarChar(sql.MAX), detail).query(`
      INSERT eval.permutation_results(run_id,permutation_id,metric_name,permutations,seed,observed_value,null_mean,null_sd,p_value,detail_json)
      VALUES(@run,@id,@metric,@reps,@seed,@observed,@mean,@sd,@p,@detail);
    `);
  return 1;
}

async function validatedFreeze(): Promise<Record<string, unknown>> {
  const parsed = JSON.parse(await readFile(`${runDirectory}/manifests/freeze.json`, "utf8")) as Record<string, unknown>;
  const hash = parsed.receiptSha256;
  const { receiptSha256: _ignored, ...body } = parsed;
  if (typeof hash !== "string" || hashJson(body) !== hash) throw new Error("Campaign freeze receipt hash drift");
  return parsed;
}

function distributionTable(rows: ContrastOutput[], field: "bootstrapValues" | "nullValues", kind: string): string {
  const lines: string[] = [];
  for (const row of rows) for (const [replicate, value] of row[field].entries()) lines.push(canonicalJson({
    schemaVersion: 1, kind, contrastId: row.contrastId, metricName: row.metricName, replicate, value,
  }));
  return `${lines.join("\n")}\n`;
}

function groupBy<T>(values: T[], key: (value: T) => string): Map<string, T[]> {
  const output = new Map<string, T[]>();
  for (const value of values) output.set(key(value), [...(output.get(key(value)) ?? []), value]);
  return output;
}
function unique(values: string[]): string[] { return [...new Set(values)].sort(); }
function sum(values: number[]): number { return values.reduce((total, value) => total + value, 0); }
function mean(values: number[]): number { return sum(values) / values.length; }
function standardError(values: number[]): number {
  if (values.length < 2) return 0;
  const center = mean(values);
  return round(Math.sqrt(sum(values.map((value) => (value - center) ** 2)) / (values.length - 1)) / Math.sqrt(values.length));
}
function round(value: number): number { return Math.round(value * 1_000_000_000) / 1_000_000_000; }
function metricSort(left: MetricOutput, right: MetricOutput): number {
  return String(left.profileKey).localeCompare(String(right.profileKey)) || left.armId.localeCompare(right.armId)
    || String(left.splitRole).localeCompare(String(right.splitRole)) || String(left.family).localeCompare(String(right.family))
    || String(left.regime).localeCompare(String(right.regime)) || left.metricName.localeCompare(right.metricName);
}
function listArgument(name: string, fallback: string): string[] {
  const values = [...new Set((argument(name) ?? fallback).split(",").map((value) => value.trim()).filter(Boolean))];
  if (values.length === 0) throw new Error(`${name} cannot be empty`);
  return values;
}
function integerArgument(name: string, fallback: number, minimum: number, maximum: number): number {
  const value = Number(argument(name) ?? fallback);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(`${name} must be an integer from ${minimum} through ${maximum}`);
  return value;
}
function argument(name: string): string | undefined { const index = process.argv.indexOf(name); return index < 0 ? undefined : process.argv[index + 1]; }
