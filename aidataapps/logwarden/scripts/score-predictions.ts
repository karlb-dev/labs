import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { parseInferenceControlId, type InferenceControlId } from "../src/controls.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { scoreRunbookRetrieval } from "../src/retrieval-metrics.js";
import { parseReplayRoles, type ReplayRole } from "../src/replay.js";
import { scorePrediction, toolScoreRows, type ScoreToolInvocation, type ScoreTruth } from "../src/scoring.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";

interface PredictionRow {
  prediction_id: string;
  job_id: string;
  agent_run_id: string | null;
  episode_id: string;
  model_profile_id: string | null;
  agent_arm_id: string;
  prediction_source: string;
  predicted_class: string | null;
  predicted_severity: string | null;
  predicted_action: string | null;
  abstained: boolean | null;
  outcome: "decision" | "abstention" | "failure";
  split_role: string;
  expected_class: string;
  expected_severity: string;
  expected_action: string;
  acceptable_actions_json: string;
  should_abstain: boolean;
  expected_runbooks_json: string;
  protected_truth_json: string;
  prediction_json: string;
  contract_valid: boolean | null;
  policy_valid: boolean | null;
  control_id: string | null;
  source_prediction_id: string | null;
}
interface ToolRow {
  prediction_id: string;
  tool_name: string;
  status: string;
  policy_status: string;
  snapshot_miss: boolean;
}
interface RunbookRow {
  prediction_id: string;
  retrieval_run_id: string | null;
  rank_ordinal: number;
  runbook_id: string;
}

const startedAtUtc = new Date().toISOString();
const roles = parseReplayRoles(argument("--roles") ?? "dev");
const arms = listArgument("--arms", "B1-rules-v1,A-direct,A-rag,A-tools");
const profiles = listArgument("--profiles", "qwen-smoke");
const controlId = controlArgument();
const controlSuffix = controlId === null ? "" : `-${safeName(controlId)}`;
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const pool = await connect(config.databases.lab, config.databases.controlName, 600_000);

try {
  await assertScoringAuthorized();
  const predictions = await loadPredictions();
  if (predictions.length === 0) throw new Error("No predictions matched the requested scoring cells");
  const ids = JSON.stringify(predictions.map((row) => Number(row.prediction_id)));
  const [tools, returned, cited] = await Promise.all([
    loadTools(ids), loadReturnedRunbooks(ids), loadCitedRunbooks(ids),
  ]);
  const toolMap = groupBy(tools, (row) => row.prediction_id);
  const returnedMap = groupBy(returned, (row) => row.prediction_id);
  const citedMap = groupBy(cited, (row) => row.prediction_id);
  const rows: Array<Record<string, unknown>> = [];
  let insertedDecisionScores = 0;
  let insertedToolScores = 0;
  let insertedRetrievalScores = 0;
  for (const prediction of predictions) {
    const predictionId = prediction.prediction_id;
    const protectedTruth = JSON.parse(prediction.protected_truth_json) as Record<string, unknown>;
    const truth = parseTruth(prediction, protectedTruth);
    const toolInvocations = (toolMap.get(predictionId) ?? []).map((row): ScoreToolInvocation => ({
      toolName: row.tool_name,
      status: row.status,
      policyStatus: row.policy_status,
      snapshotMiss: Boolean(row.snapshot_miss),
    }));
    const returnedRows = (returnedMap.get(predictionId) ?? []).sort((left, right) => left.rank_ordinal - right.rank_ordinal);
    const returnedRunbooks = unique(returnedRows.map((row) => row.runbook_id));
    const citedRunbooks = unique((citedMap.get(predictionId) ?? []).map((row) => row.runbook_id));
    const scoringArmId = scoringIdentity(prediction);
    const score = scorePrediction({
      truth,
      prediction: {
        outcome: prediction.outcome,
        predictedClass: prediction.predicted_class,
        predictedSeverity: prediction.predicted_severity,
        predictedAction: prediction.predicted_action,
        abstained: prediction.abstained === null ? null : Boolean(prediction.abstained),
        contractValid: prediction.prediction_source !== "agent" ? prediction.outcome !== "failure" : Boolean(prediction.contract_valid),
        policyValid: prediction.prediction_source !== "agent" ? true : Boolean(prediction.policy_valid),
      },
      armId: scoringArmId,
      tools: toolInvocations,
      citedRunbooks,
      returnedRunbooks,
    });
    const toolScores = toolScoreRows(truth, toolInvocations);
    const retrieval = scoreRunbookRetrieval(truth.expectedRunbooks, returnedRunbooks, 5);
    const citationPrecision = citedRunbooks.length === 0 ? (truth.expectedRunbooks.length === 0 ? 1 : 0)
      : citedRunbooks.filter((value) => truth.expectedRunbooks.includes(value)).length / citedRunbooks.length;
    const citationRecall = truth.expectedRunbooks.length === 0 ? (citedRunbooks.length === 0 ? 1 : 0)
      : new Set(citedRunbooks.filter((value) => truth.expectedRunbooks.includes(value))).size / truth.expectedRunbooks.length;
    const detail = {
      schemaVersion: 1,
      predictionId: Number(predictionId),
      episodeId: prediction.episode_id,
      modelProfileId: prediction.model_profile_id,
      agentArmId: prediction.agent_arm_id,
      controlId: prediction.control_id,
      sourcePredictionId: prediction.source_prediction_id === null ? null : Number(prediction.source_prediction_id),
      scoringArmId,
      splitRole: prediction.split_role,
      score,
      prediction: {
        outcome: prediction.outcome,
        predictedClass: prediction.predicted_class,
        predictedSeverity: prediction.predicted_severity,
        predictedAction: prediction.predicted_action,
        abstained: prediction.abstained === null ? null : Boolean(prediction.abstained),
        contractValid: prediction.prediction_source !== "agent" ? prediction.outcome !== "failure" : Boolean(prediction.contract_valid),
        policyValid: prediction.prediction_source !== "agent" ? true : Boolean(prediction.policy_valid),
      },
      truth: {
        expectedClass: truth.expectedClass,
        expectedSeverity: truth.expectedSeverity,
        expectedAction: truth.expectedAction,
        acceptableActions: truth.acceptableActions,
        shouldAbstain: truth.shouldAbstain,
        requiredTools: truth.requiredTools,
        optionalTools: truth.optionalTools,
        forbiddenTools: truth.forbiddenTools,
        expectedRunbooks: truth.expectedRunbooks,
        costWeights: truth.costWeights,
      },
      observed: { toolInvocations, returnedRunbooks, citedRunbooks },
    };
    insertedDecisionScores += await persistDecisionScore(predictionId, score, detail);
    insertedToolScores += await persistToolScores(predictionId, toolScores);
    insertedRetrievalScores += await persistRetrievalScore(
      predictionId,
      returnedRows[0]?.retrieval_run_id ?? evaluatorRetrievalRunId(prediction),
      retrieval,
      returnedRunbooks.length > 0,
      citationPrecision,
      citationRecall,
      (prediction.agent_arm_id === "A-rag" || prediction.agent_arm_id === "A-tools")
        && returnedRunbooks.length > 0 && score.citationPolicySatisfied,
      { expectedRunbooks: truth.expectedRunbooks, returnedRunbooks, citedRunbooks },
    );
    rows.push({
      predictionId: Number(predictionId),
      episodeId: prediction.episode_id,
      modelProfileId: prediction.model_profile_id,
      agentArmId: prediction.agent_arm_id,
      controlId: prediction.control_id,
      splitRole: prediction.split_role,
      ...score,
      retrieval,
      citationPrecision,
      citationRecall,
    });
  }
  rows.sort((left, right) => String(left.splitRole).localeCompare(String(right.splitRole))
    || String(left.modelProfileId).localeCompare(String(right.modelProfileId))
    || String(left.agentArmId).localeCompare(String(right.agentArmId))
    || String(left.episodeId).localeCompare(String(right.episodeId)));
  const tableBody = rows.map((row) => canonicalJson(row)).join("\n") + "\n";
  const tablePath = `${runDirectory}/tables/prediction-scores-${safeName(roles.join("-"))}-${safeName(profiles.join("-"))}${controlSuffix}.jsonl`;
  await atomicWrite(tablePath, tableBody, 0o600);
  const counts = Object.fromEntries(arms.map((arm) => [arm, rows.filter((row) => row.agentArmId === arm).length]));
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    roles,
    profiles,
    arms,
    controlId,
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    predictionCount: rows.length,
    counts,
    insertedThisInvocation: { decisionScores: insertedDecisionScores, toolScores: insertedToolScores, retrievalScores: insertedRetrievalScores },
    resumedRows: rows.length - insertedDecisionScores,
    orderedScoreSetSha256: hashJson(rows),
    tablePath,
    tableBytes: Buffer.byteLength(tableBody),
    tableSha256: sha256(tableBody),
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const receiptPath = `${runDirectory}/metrics/prediction-scores-${safeName(roles.join("-"))}-${safeName(profiles.join("-"))}${controlSuffix}.json`;
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  await appendExperimentLog(`Scored ${rows.length} ${controlId === null ? "primary" : `control ${controlId}`} ${roles.join(",")} predictions for ${profiles.join(",")} across ${arms.join(",")}; receipt ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({ roles, profiles, arms, controlId, predictionCount: rows.length, counts, insertedDecisionScores, insertedToolScores, insertedRetrievalScores, receiptPath, receiptSha256: receipt.receiptSha256, disposition: "PASS" }, null, 2));
} finally {
  await pool.close();
}

async function assertScoringAuthorized(): Promise<void> {
  if (controlId !== null && (arms.length !== 1 || arms[0] !== "A-tools")) throw new Error("Tier 1 inference-control scoring is restricted to A-tools");
  const status = await pool.request().input("run", sql.VarChar(120), run.runId).query<{ status: string }>(`
    SELECT campaign.status FROM control.campaigns campaign INNER JOIN control.runs run ON run.campaign_id=campaign.campaign_id WHERE run.run_id=@run;
  `);
  const campaignStatus = status.recordset[0]?.status;
  if (roles.some((role) => role.startsWith("test_")) && !["frozen", "running", "complete"].includes(campaignStatus ?? "")) {
    throw new Error("Test-role scoring is forbidden before campaign freeze");
  }
}

async function loadPredictions(): Promise<PredictionRow[]> {
  const result = await pool.request().input("roles", sql.NVarChar(sql.MAX), JSON.stringify(roles))
    .input("arms", sql.NVarChar(sql.MAX), JSON.stringify(arms)).input("profiles", sql.NVarChar(sql.MAX), JSON.stringify(profiles))
    .input("control", sql.VarChar(80), controlId)
    .query<PredictionRow>(`
      WITH selected_roles AS (SELECT CONVERT(varchar(40),value) value FROM OPENJSON(@roles)),
           selected_arms AS (SELECT CONVERT(varchar(80),value) value FROM OPENJSON(@arms)),
           selected_profiles AS (SELECT CONVERT(varchar(80),value) value FROM OPENJSON(@profiles))
      SELECT prediction.prediction_id,prediction.job_id,prediction.agent_run_id,prediction.episode_id,
        prediction.model_profile_id,prediction.agent_arm_id,prediction.prediction_source,
        prediction.predicted_class,prediction.predicted_severity,prediction.predicted_action,
        prediction.abstained,prediction.outcome,truth.split_role,truth.expected_class,truth.expected_severity,
        truth.expected_action,truth.acceptable_actions_json,truth.should_abstain,truth.expected_runbooks_json,
        truth.protected_truth_json,prediction.prediction_json,decision.contract_valid,decision.policy_valid,
        prediction.control_id,prediction.source_prediction_id
      FROM eval.predictions prediction INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=prediction.episode_id
      INNER JOIN selected_roles role ON role.value=truth.split_role
      INNER JOIN selected_arms arm ON arm.value=prediction.agent_arm_id
      LEFT JOIN selected_profiles profile ON profile.value=prediction.model_profile_id
      LEFT JOIN agent.decisions decision ON decision.decision_id=prediction.decision_id
      WHERE (prediction.model_profile_id IS NULL OR profile.value IS NOT NULL)
        AND ((@control IS NULL AND prediction.control_id IS NULL) OR prediction.control_id=@control)
      ORDER BY truth.split_role,prediction.model_profile_id,prediction.agent_arm_id,prediction.episode_id;
    `);
  return result.recordset;
}

async function loadTools(ids: string): Promise<ToolRow[]> {
  const result = await pool.request().input("ids", sql.NVarChar(sql.MAX), ids).input("run", sql.VarChar(120), run.runId).query<ToolRow>(`
    WITH selected AS (SELECT CONVERT(bigint,value) prediction_id FROM OPENJSON(@ids)),
    observed AS
    (
      SELECT CONVERT(varchar(30),prediction.prediction_id) prediction_id,tool.tool_name,tool.status,
        tool.policy_status,tool.snapshot_miss,tool.tool_invocation_id sort_ordinal
      FROM selected INNER JOIN eval.predictions prediction ON prediction.prediction_id=selected.prediction_id
      INNER JOIN agent.tool_invocations tool ON tool.agent_run_id=prediction.agent_run_id
      UNION ALL
      SELECT CONVERT(varchar(30),prediction.prediction_id),'runbook_search','success','allowed',CONVERT(bit,0),CONVERT(bigint,0)
      FROM selected INNER JOIN eval.predictions prediction ON prediction.prediction_id=selected.prediction_id
      INNER JOIN eval.retrieval_benchmark_results benchmark ON benchmark.run_id=@run AND benchmark.episode_id=prediction.episode_id
        AND benchmark.retrieval_mode=CASE prediction.agent_arm_id
          WHEN 'B2-lexical-v1' THEN 'lexical_fulltext'
          WHEN 'B2-vector-v1' THEN 'vector_exact'
          WHEN 'B2-hybrid-v1' THEN 'hybrid_rrf' END
      WHERE prediction.prediction_source='B2' AND benchmark.evaluator_only=1
    )
    SELECT prediction_id,tool_name,status,policy_status,snapshot_miss FROM observed ORDER BY prediction_id,sort_ordinal;
  `);
  return result.recordset;
}

async function loadReturnedRunbooks(ids: string): Promise<RunbookRow[]> {
  const result = await pool.request().input("ids", sql.NVarChar(sql.MAX), ids).input("run", sql.VarChar(120), run.runId).query<RunbookRow>(`
    WITH selected AS (SELECT CONVERT(bigint,value) prediction_id FROM OPENJSON(@ids))
    SELECT CONVERT(varchar(30),prediction.prediction_id) prediction_id,CONVERT(varchar(30),retrieval.retrieval_run_id) retrieval_run_id,
      result.rank_ordinal,chunk.runbook_id
    FROM selected INNER JOIN eval.predictions prediction ON prediction.prediction_id=selected.prediction_id
    INNER JOIN agent.tool_invocations tool ON tool.agent_run_id=prediction.agent_run_id AND tool.retrieval_run_id IS NOT NULL
    INNER JOIN kb.retrieval_runs retrieval ON retrieval.retrieval_run_id=tool.retrieval_run_id
    INNER JOIN kb.retrieval_results result ON result.retrieval_run_id=retrieval.retrieval_run_id
    INNER JOIN kb.runbook_chunks chunk ON chunk.chunk_id=result.chunk_id
    WHERE result.returned_to_agent=1
    UNION ALL
    SELECT CONVERT(varchar(30),prediction.prediction_id),CONVERT(varchar(30),benchmark.retrieval_run_id),
      CONVERT(int,returned.[key])+1,CONVERT(varchar(100),returned.value)
    FROM selected INNER JOIN eval.predictions prediction ON prediction.prediction_id=selected.prediction_id
    INNER JOIN eval.retrieval_benchmark_results benchmark ON benchmark.run_id=@run AND benchmark.episode_id=prediction.episode_id
      AND benchmark.retrieval_mode=CASE prediction.agent_arm_id
        WHEN 'B2-lexical-v1' THEN 'lexical_fulltext'
        WHEN 'B2-vector-v1' THEN 'vector_exact'
        WHEN 'B2-hybrid-v1' THEN 'hybrid_rrf' END
    CROSS APPLY OPENJSON(benchmark.returned_runbooks_json) returned
    WHERE prediction.prediction_source='B2' AND benchmark.evaluator_only=1
    ORDER BY prediction_id,rank_ordinal;
  `);
  return result.recordset;
}

async function loadCitedRunbooks(ids: string): Promise<RunbookRow[]> {
  const result = await pool.request().input("ids", sql.NVarChar(sql.MAX), ids).query<RunbookRow>(`
    WITH selected AS (SELECT CONVERT(bigint,value) prediction_id FROM OPENJSON(@ids))
    SELECT CONVERT(varchar(30),prediction.prediction_id) prediction_id,CONVERT(varchar(30),citation.retrieval_run_id) retrieval_run_id,
      citation.citation_ordinal rank_ordinal,chunk.runbook_id
    FROM selected INNER JOIN eval.predictions prediction ON prediction.prediction_id=selected.prediction_id
    INNER JOIN agent.decision_citations citation ON citation.decision_id=prediction.decision_id
    INNER JOIN kb.runbook_chunks chunk ON chunk.chunk_id=citation.chunk_id
    ORDER BY prediction.prediction_id,citation.citation_ordinal;
  `);
  return result.recordset;
}

function parseTruth(row: PredictionRow, protectedTruth: Record<string, unknown>): ScoreTruth {
  const strings = (value: unknown, label: string): string[] => {
    if (!Array.isArray(value) || !value.every((child) => typeof child === "string")) throw new Error(`Invalid ${label} for ${row.episode_id}`);
    return value;
  };
  const costs = protectedTruth.costWeights as Record<string, unknown> | undefined;
  if (costs === undefined) throw new Error(`Missing cost weights for ${row.episode_id}`);
  return {
    expectedClass: row.expected_class,
    expectedSeverity: row.expected_severity,
    expectedAction: row.expected_action,
    acceptableActions: strings(JSON.parse(row.acceptable_actions_json), "acceptable actions"),
    shouldAbstain: Boolean(row.should_abstain),
    expectedRunbooks: strings(JSON.parse(row.expected_runbooks_json), "expected runbooks"),
    requiredTools: strings(protectedTruth.requiredTools, "required tools"),
    optionalTools: strings(protectedTruth.optionalTools, "optional tools"),
    forbiddenTools: strings(protectedTruth.forbiddenTools, "forbidden tools"),
    costWeights: {
      miss: Number(costs.miss),
      falseAlarm: Number(costs.falseAlarm),
      unnecessaryTool: Number(costs.unnecessaryTool),
    },
  };
}

function scoringIdentity(prediction: PredictionRow): string {
  if (prediction.agent_arm_id !== "A-router") return prediction.agent_arm_id;
  const parsed = JSON.parse(prediction.prediction_json) as Record<string, unknown>;
  if (parsed.derivedFromArm !== "B1-rules-v1" && parsed.derivedFromArm !== "A-tools") {
    throw new Error(`A-router prediction ${prediction.prediction_id} has no governed source arm`);
  }
  return parsed.derivedFromArm;
}

function evaluatorRetrievalRunId(prediction: PredictionRow): string | null {
  if (prediction.prediction_source !== "B2") return null;
  const parsed = JSON.parse(prediction.prediction_json) as Record<string, unknown>;
  const value = Number(parsed.retrievalRunId);
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`B2 prediction ${prediction.prediction_id} has no evaluator retrieval run`);
  return String(value);
}

async function persistDecisionScore(predictionId: string, score: ReturnType<typeof scorePrediction>, detail: unknown): Promise<number> {
  const detailJson = canonicalJson(detail);
  const prior = await pool.request().input("prediction", sql.BigInt, predictionId)
    .query<{ detail_json: string }>("SELECT detail_json FROM eval.decision_scores WHERE prediction_id=@prediction;");
  if (prior.recordset[0] !== undefined) {
    if (canonicalJson(JSON.parse(prior.recordset[0].detail_json)) !== detailJson) throw new Error(`Decision-score drift for prediction ${predictionId}`);
    return 0;
  }
  await pool.request().input("prediction", sql.BigInt, predictionId)
    .input("e2e", sql.Bit, score.endToEndSuccess).input("class", sql.Bit, score.classCorrect)
    .input("severity", sql.Bit, score.severityCorrect).input("severity_cost", sql.Decimal(9, 4), score.severityOrdinalCost)
    .input("action", sql.Bit, score.actionCorrect).input("abstention", sql.Bit, score.abstentionCorrect)
    .input("contract", sql.Bit, score.contractSuccess).input("required", sql.Bit, score.requiredToolsSatisfied)
    .input("forbidden", sql.Bit, score.forbiddenToolsAvoided).input("citation", sql.Bit, score.citationPolicySatisfied)
    .input("composite", sql.Decimal(9, 6), score.compositeScore).input("detail", sql.NVarChar(sql.MAX), detailJson).query(`
      INSERT eval.decision_scores(prediction_id,end_to_end_success,class_correct,severity_correct,severity_ordinal_cost,
        action_correct,abstention_correct,contract_success,required_tools_satisfied,forbidden_tools_avoided,
        citation_policy_satisfied,composite_score,detail_json)
      VALUES(@prediction,@e2e,@class,@severity,@severity_cost,@action,@abstention,@contract,@required,@forbidden,@citation,@composite,@detail);
    `);
  return 1;
}

async function persistToolScores(predictionId: string, rows: ReturnType<typeof toolScoreRows>): Promise<number> {
  let inserted = 0;
  for (const row of rows) {
    const detail = canonicalJson(row);
    const prior = await pool.request().input("prediction", sql.BigInt, predictionId).input("tool", sql.VarChar(80), row.toolName)
      .query<{ detail_json: string }>("SELECT detail_json FROM eval.tool_scores WHERE prediction_id=@prediction AND tool_name=@tool;");
    if (prior.recordset[0] !== undefined) {
      if (canonicalJson(JSON.parse(prior.recordset[0].detail_json)) !== detail) throw new Error(`Tool-score drift for ${predictionId}/${row.toolName}`);
      continue;
    }
    await pool.request().input("prediction", sql.BigInt, predictionId).input("tool", sql.VarChar(80), row.toolName)
      .input("expectation", sql.VarChar(24), row.expectation).input("called", sql.Int, row.calledCount)
      .input("valid", sql.Int, row.validCallCount).input("miss", sql.Int, row.snapshotMissCount)
      .input("errors", sql.Int, row.toolErrorCount).input("correct", sql.Bit, row.correctUse)
      .input("score", sql.Decimal(9, 6), row.score).input("detail", sql.NVarChar(sql.MAX), detail).query(`
        INSERT eval.tool_scores(prediction_id,tool_name,expectation,called_count,valid_call_count,snapshot_miss_count,
          tool_error_count,correct_use,score,detail_json)
        VALUES(@prediction,@tool,@expectation,@called,@valid,@miss,@errors,@correct,@score,@detail);
      `);
    inserted += 1;
  }
  return inserted;
}

async function persistRetrievalScore(
  predictionId: string,
  retrievalRunId: string | null,
  score: ReturnType<typeof scoreRunbookRetrieval>,
  retrievedAny: boolean,
  citationPrecision: number,
  citationRecall: number,
  grounded: boolean,
  detail: unknown,
): Promise<number> {
  const detailJson = canonicalJson({ ...detail as Record<string, unknown>, score, citationPrecision, citationRecall, grounded });
  const prior = await pool.request().input("prediction", sql.BigInt, predictionId)
    .query<{ detail_json: string }>("SELECT detail_json FROM eval.retrieval_scores WHERE prediction_id=@prediction;");
  if (prior.recordset[0] !== undefined) {
    if (canonicalJson(JSON.parse(prior.recordset[0].detail_json)) !== detailJson) throw new Error(`Retrieval-score drift for prediction ${predictionId}`);
    return 0;
  }
  await pool.request().input("prediction", sql.BigInt, predictionId)
    .input("retrieval", sql.BigInt, retrievalRunId === null ? null : Number(retrievalRunId))
    .input("answerable", sql.Bit, score.answerable).input("any", sql.Bit, retrievedAny)
    .input("recall", sql.Decimal(9, 6), score.recallAtK).input("mrr", sql.Decimal(9, 6), score.reciprocalRank)
    .input("ndcg", sql.Decimal(9, 6), score.ndcgAtK).input("no_answer", sql.Bit, score.noAnswerCorrect)
    .input("precision", sql.Decimal(9, 6), citationPrecision).input("citation_recall", sql.Decimal(9, 6), citationRecall)
    .input("grounded", sql.Bit, grounded).input("detail", sql.NVarChar(sql.MAX), detailJson).query(`
      INSERT eval.retrieval_scores(prediction_id,retrieval_run_id,answerable,retrieved_any,recall_at_k,reciprocal_rank,
        ndcg_at_k,no_answer_correct,citation_precision,citation_recall,grounded,detail_json)
      VALUES(@prediction,@retrieval,@answerable,@any,@recall,@mrr,@ndcg,@no_answer,@precision,@citation_recall,@grounded,@detail);
    `);
  return 1;
}

function groupBy<T>(values: T[], key: (value: T) => string): Map<string, T[]> {
  const output = new Map<string, T[]>();
  for (const value of values) output.set(key(value), [...(output.get(key(value)) ?? []), value]);
  return output;
}
function unique(values: string[]): string[] { return [...new Set(values)]; }
function listArgument(name: string, fallback: string): string[] {
  const values = [...new Set((argument(name) ?? fallback).split(",").map((value) => value.trim()).filter(Boolean))];
  if (values.length === 0) throw new Error(`${name} cannot be empty`);
  return values;
}
function argument(name: string): string | undefined { const index = process.argv.indexOf(name); return index < 0 ? undefined : process.argv[index + 1]; }
function controlArgument(): InferenceControlId | null {
  const value = argument("--control");
  return value === undefined || value === "primary" ? null : parseInferenceControlId(value);
}
function safeName(value: string): string { return value.replace(/[^a-z0-9_.-]+/gi, "-").toLowerCase(); }
