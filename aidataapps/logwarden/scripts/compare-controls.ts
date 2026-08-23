import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { loadTier1ControlPolicy, parseInferenceControlId, type InferenceControlId } from "../src/controls.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";

interface PairRow {
  control_prediction_id: string;
  primary_prediction_id: string;
  episode_id: string;
  split_role: string;
  control_agent_run_id: string | null;
  primary_agent_run_id: string | null;
  control_decision_sha256: string | null;
  primary_decision_sha256: string | null;
  control_class: string | null;
  primary_class: string | null;
  control_severity: string | null;
  primary_severity: string | null;
  control_action: string | null;
  primary_action: string | null;
  control_abstained: boolean | null;
  primary_abstained: boolean | null;
  control_outcome: string;
  primary_outcome: string;
  control_confidence: number | null;
  primary_confidence: number | null;
  control_action_score: boolean | null;
  primary_action_score: boolean | null;
}

interface ResponseRow {
  prediction_id: string;
  turn_ordinal: number;
  retry_ordinal: number;
  response_body_sha256: string;
}

interface ToolRow {
  prediction_id: string;
  turn_ordinal: number;
  tool_invocation_id: string;
  tool_name: string;
  canonical_args_sha256: string;
  result_sha256: string | null;
  raw_result_sha256: string | null;
}

interface ComparisonRow extends Record<string, unknown> {
  controlPredictionId: number;
  primaryPredictionId: number;
  episodeId: string;
  splitRole: string;
  rawResponseAgreement: boolean | null;
  decisionAgreement: boolean | null;
  semanticDecisionAgreement: boolean;
  orderedToolCallAgreement: boolean | null;
  transmittedToolResultAgreement: boolean | null;
  rawToolResultAgreement: boolean | null;
  actionScoreDelta: number | null;
  confidenceDelta: number | null;
}

const startedAtUtc = new Date().toISOString();
const profileKey = requiredArgument("--profile");
const controlId = parseInferenceControlId(requiredArgument("--control"));
const policy = loadTier1ControlPolicy();
const policySha256 = hashJson(policy);
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const pool = await connect(config.databases.lab, config.databases.controlName, 600_000);

try {
  const pairs = await loadPairs();
  const expected = controlId === "batching-sequential-v1" ? 48 : 96;
  if (pairs.length !== expected) throw new Error(`Control comparison found ${pairs.length}/${expected} frozen pairs for ${profileKey}/${controlId}`);
  const predictionIds = pairs.flatMap((row) => [Number(row.control_prediction_id), Number(row.primary_prediction_id)]);
  const idsJson = JSON.stringify(predictionIds);
  const [responses, tools] = await Promise.all([loadResponses(idsJson), loadTools(idsJson)]);
  const responsesByPrediction = groupBy(responses, (row) => row.prediction_id);
  const toolsByPrediction = groupBy(tools, (row) => row.prediction_id);
  const rows: ComparisonRow[] = [];
  let inserted = 0;
  for (const pair of pairs) {
    const controlResponses = responseSequence(responsesByPrediction.get(pair.control_prediction_id) ?? []);
    const primaryResponses = responseSequence(responsesByPrediction.get(pair.primary_prediction_id) ?? []);
    const controlTools = toolSequences(toolsByPrediction.get(pair.control_prediction_id) ?? []);
    const primaryTools = toolSequences(toolsByPrediction.get(pair.primary_prediction_id) ?? []);
    const rawResponseAgreement = nullableSequenceAgreement(controlResponses, primaryResponses);
    const decisionAgreement = pair.control_decision_sha256 === null || pair.primary_decision_sha256 === null
      ? null : pair.control_decision_sha256 === pair.primary_decision_sha256;
    const orderedToolCallAgreement = pair.control_agent_run_id === null || pair.primary_agent_run_id === null
      ? null : hashJson(controlTools.calls) === hashJson(primaryTools.calls);
    const transmittedToolResultAgreement = pair.control_agent_run_id === null || pair.primary_agent_run_id === null
      ? null : hashJson(controlTools.transmittedResults) === hashJson(primaryTools.transmittedResults);
    const rawToolResultAgreement = pair.control_agent_run_id === null || pair.primary_agent_run_id === null
      ? null : hashJson(controlTools.rawResults) === hashJson(primaryTools.rawResults);
    const semanticDecisionAgreement = canonicalJson(semanticDecision(pair, "control")) === canonicalJson(semanticDecision(pair, "primary"));
    const actionScoreDelta = pair.control_action_score === null || pair.primary_action_score === null
      ? null : Number(pair.control_action_score) - Number(pair.primary_action_score);
    const confidenceDelta = pair.control_confidence === null || pair.primary_confidence === null
      ? null : round(Number(pair.control_confidence) - Number(pair.primary_confidence));
    const row: ComparisonRow = {
      schemaVersion: 1,
      runId: run.runId,
      profileKey,
      controlId,
      controlPolicySha256: policySha256,
      controlPredictionId: Number(pair.control_prediction_id),
      primaryPredictionId: Number(pair.primary_prediction_id),
      episodeId: pair.episode_id,
      splitRole: pair.split_role,
      rawResponseAgreement,
      decisionAgreement,
      semanticDecisionAgreement,
      orderedToolCallAgreement,
      transmittedToolResultAgreement,
      rawToolResultAgreement,
      actionScoreDelta,
      confidenceDelta,
      evidence: {
        rawResponses: {
          controlCount: controlResponses.length,
          primaryCount: primaryResponses.length,
          controlOrderedSha256: hashJson(controlResponses),
          primaryOrderedSha256: hashJson(primaryResponses),
          control: controlResponses,
          primary: primaryResponses,
        },
        decisions: {
          controlSha256: pair.control_decision_sha256,
          primarySha256: pair.primary_decision_sha256,
          controlSemantic: semanticDecision(pair, "control"),
          primarySemantic: semanticDecision(pair, "primary"),
        },
        toolCalls: {
          controlCount: controlTools.calls.length,
          primaryCount: primaryTools.calls.length,
          controlOrderedSha256: hashJson(controlTools.calls),
          primaryOrderedSha256: hashJson(primaryTools.calls),
          control: controlTools.calls,
          primary: primaryTools.calls,
        },
        transmittedToolResults: {
          controlOrderedSha256: hashJson(controlTools.transmittedResults),
          primaryOrderedSha256: hashJson(primaryTools.transmittedResults),
        },
        rawToolResults: {
          controlOrderedSha256: hashJson(controlTools.rawResults),
          primaryOrderedSha256: hashJson(primaryTools.rawResults),
        },
      },
    };
    inserted += await persistComparison(row);
    rows.push(row);
  }
  rows.sort((left, right) => left.splitRole.localeCompare(right.splitRole) || left.episodeId.localeCompare(right.episodeId));
  const tableBody = `${rows.map((row) => canonicalJson(row)).join("\n")}\n`;
  const tablePath = `${runDirectory}/tables/control-comparison-${safeName(profileKey)}-${safeName(controlId)}.jsonl`;
  await atomicWrite(tablePath, tableBody, 0o600);
  const summary = {
    pairCount: rows.length,
    insertedThisInvocation: inserted,
    rawResponseAgreement: summarizeBoolean(rows.map((row) => row.rawResponseAgreement)),
    decisionAgreement: summarizeBoolean(rows.map((row) => row.decisionAgreement)),
    semanticDecisionAgreement: summarizeBoolean(rows.map((row) => row.semanticDecisionAgreement)),
    orderedToolCallAgreement: summarizeBoolean(rows.map((row) => row.orderedToolCallAgreement)),
    transmittedToolResultAgreement: summarizeBoolean(rows.map((row) => row.transmittedToolResultAgreement)),
    rawToolResultAgreement: summarizeBoolean(rows.map((row) => row.rawToolResultAgreement)),
    meanActionScoreDelta: mean(rows.map((row) => row.actionScoreDelta)),
    meanConfidenceDelta: mean(rows.map((row) => row.confidenceDelta)),
  };
  const invarianceOutcome = controlId !== "batching-sequential-v1" ? "NOT_APPLICABLE"
    : [summary.rawResponseAgreement, summary.decisionAgreement, summary.orderedToolCallAgreement]
      .every((value) => value.nullCount === 0 && value.falseCount === 0) ? "INVARIANT" : "NON_INVARIANT";
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    profileKey,
    controlId,
    controlPolicySha256: policySha256,
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    summary,
    invarianceOutcome,
    orderedComparisonSetSha256: hashJson(rows),
    tablePath,
    tableBytes: Buffer.byteLength(tableBody),
    tableSha256: sha256(tableBody),
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const receiptPath = `${runDirectory}/metrics/control-comparison-${safeName(profileKey)}-${safeName(controlId)}.json`;
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  await persistEvidence(receipt);
  await appendExperimentLog(`Compared ${rows.length} ${profileKey}/${controlId} predictions with their frozen primary sources: raw=${formatRate(summary.rawResponseAgreement)}, decision=${formatRate(summary.decisionAgreement)}, tools=${formatRate(summary.orderedToolCallAgreement)}, invariance=${invarianceOutcome}; receipt ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({ profileKey, controlId, summary, invarianceOutcome, receiptPath, receiptSha256: receipt.receiptSha256, disposition: "PASS" }, null, 2));
} finally {
  await pool.close();
}

async function loadPairs(): Promise<PairRow[]> {
  const result = await pool.request().input("run", sql.VarChar(120), run.runId)
    .input("profile", sql.VarChar(80), profileKey).input("control", sql.VarChar(80), controlId)
    .query<PairRow>(`
      SELECT control_prediction.prediction_id control_prediction_id,
        primary_prediction.prediction_id primary_prediction_id,control_prediction.episode_id,truth.split_role,
        control_prediction.agent_run_id control_agent_run_id,primary_prediction.agent_run_id primary_agent_run_id,
        control_decision.decision_sha256 control_decision_sha256,primary_decision.decision_sha256 primary_decision_sha256,
        control_prediction.predicted_class control_class,primary_prediction.predicted_class primary_class,
        control_prediction.predicted_severity control_severity,primary_prediction.predicted_severity primary_severity,
        control_prediction.predicted_action control_action,primary_prediction.predicted_action primary_action,
        control_prediction.abstained control_abstained,primary_prediction.abstained primary_abstained,
        control_prediction.outcome control_outcome,primary_prediction.outcome primary_outcome,
        control_prediction.confidence control_confidence,primary_prediction.confidence primary_confidence,
        control_score.action_correct control_action_score,primary_score.action_correct primary_action_score
      FROM eval.predictions control_prediction
      INNER JOIN ops.work_items item ON item.job_id=control_prediction.job_id AND item.run_id=@run
      INNER JOIN eval.predictions primary_prediction ON primary_prediction.prediction_id=control_prediction.source_prediction_id
      INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=control_prediction.episode_id
      LEFT JOIN agent.decisions control_decision ON control_decision.decision_id=control_prediction.decision_id
      LEFT JOIN agent.decisions primary_decision ON primary_decision.decision_id=primary_prediction.decision_id
      LEFT JOIN eval.decision_scores control_score ON control_score.prediction_id=control_prediction.prediction_id
      LEFT JOIN eval.decision_scores primary_score ON primary_score.prediction_id=primary_prediction.prediction_id
      WHERE control_prediction.control_id=@control AND control_prediction.model_profile_id=@profile
        AND control_prediction.agent_arm_id='A-tools' AND primary_prediction.control_id IS NULL
        AND primary_prediction.model_profile_id=control_prediction.model_profile_id
        AND primary_prediction.agent_arm_id=control_prediction.agent_arm_id
        AND primary_prediction.episode_id=control_prediction.episode_id
      ORDER BY truth.split_role,control_prediction.episode_id;
    `);
  if (result.recordset.some((row) => row.control_action_score === null || row.primary_action_score === null)) {
    throw new Error(`Control and primary predictions must both be scored before comparison: ${profileKey}/${controlId}`);
  }
  return result.recordset;
}

async function loadResponses(ids: string): Promise<ResponseRow[]> {
  const result = await pool.request().input("ids", sql.NVarChar(sql.MAX), ids).query<ResponseRow>(`
    WITH selected AS (SELECT CONVERT(bigint,value) prediction_id FROM OPENJSON(@ids))
    SELECT CONVERT(varchar(30),prediction.prediction_id) prediction_id,turn.turn_ordinal,request.retry_ordinal,response.response_body_sha256
    FROM selected INNER JOIN eval.predictions prediction ON prediction.prediction_id=selected.prediction_id
    INNER JOIN agent.turns turn ON turn.agent_run_id=prediction.agent_run_id
    INNER JOIN agent.model_requests request ON request.turn_id=turn.turn_id
    INNER JOIN agent.model_responses response ON response.model_request_id=request.model_request_id
    ORDER BY prediction.prediction_id,turn.turn_ordinal,request.retry_ordinal,response.model_response_id;
  `);
  return result.recordset;
}

async function loadTools(ids: string): Promise<ToolRow[]> {
  const result = await pool.request().input("ids", sql.NVarChar(sql.MAX), ids).query<ToolRow>(`
    WITH selected AS (SELECT CONVERT(bigint,value) prediction_id FROM OPENJSON(@ids))
    SELECT CONVERT(varchar(30),prediction.prediction_id) prediction_id,turn.turn_ordinal,
      CONVERT(varchar(30),tool.tool_invocation_id) tool_invocation_id,tool.tool_name,tool.canonical_args_sha256,
      tool.result_sha256,tool.raw_result_sha256
    FROM selected INNER JOIN eval.predictions prediction ON prediction.prediction_id=selected.prediction_id
    INNER JOIN agent.tool_invocations tool ON tool.agent_run_id=prediction.agent_run_id
    INNER JOIN agent.turns turn ON turn.turn_id=tool.turn_id
    ORDER BY prediction.prediction_id,turn.turn_ordinal,tool.tool_invocation_id;
  `);
  return result.recordset;
}

async function persistComparison(row: ComparisonRow): Promise<number> {
  const detailJson = canonicalJson(row);
  const prior = await pool.request().input("prediction", sql.BigInt, row.controlPredictionId)
    .query<{ detail_json: string }>("SELECT detail_json FROM eval.control_comparisons WHERE control_prediction_id=@prediction;");
  if (prior.recordset[0] !== undefined) {
    if (canonicalJson(JSON.parse(prior.recordset[0].detail_json)) !== detailJson) throw new Error(`Control comparison drift for prediction ${row.controlPredictionId}`);
    return 0;
  }
  await pool.request().input("control_prediction", sql.BigInt, row.controlPredictionId)
    .input("primary_prediction", sql.BigInt, row.primaryPredictionId)
    .input("raw", sql.Bit, row.rawResponseAgreement).input("decision", sql.Bit, row.decisionAgreement)
    .input("tools", sql.Bit, row.orderedToolCallAgreement).input("action", sql.Decimal(9, 6), row.actionScoreDelta)
    .input("confidence", sql.Decimal(9, 6), row.confidenceDelta).input("detail", sql.NVarChar(sql.MAX), detailJson).query(`
      INSERT eval.control_comparisons(control_prediction_id,primary_prediction_id,raw_response_agreement,
        decision_agreement,ordered_tool_call_agreement,action_score_delta,confidence_delta,detail_json)
      VALUES(@control_prediction,@primary_prediction,@raw,@decision,@tools,@action,@confidence,@detail);
    `);
  return 1;
}

async function persistEvidence(receipt: Record<string, unknown>): Promise<void> {
  const detail = canonicalJson(receipt);
  await pool.request().input("key", sql.VarChar(120), `control-compare-${safeName(profileKey)}-${safeName(controlId)}`)
    .input("run", sql.VarChar(120), run.runId).input("detail", sql.NVarChar(sql.MAX), detail)
    .input("hash", sql.Char(64), sha256(detail)).query(`
      IF NOT EXISTS(SELECT 1 FROM control.evidence_events WHERE event_key=@key)
        INSERT control.evidence_events(event_key,run_id,stage,scientific_tier,disposition,detail_json,detail_sha256)
        VALUES(@key,@run,'negative_control','tier1','PASS',@detail,@hash);
    `);
}

function responseSequence(rows: ResponseRow[]): string[] {
  return rows.map((row) => `${row.turn_ordinal}:${row.retry_ordinal}:${row.response_body_sha256}`);
}

function toolSequences(rows: ToolRow[]): { calls: unknown[]; transmittedResults: unknown[]; rawResults: unknown[] } {
  return {
    calls: rows.map((row) => ({ turnOrdinal: row.turn_ordinal, toolName: row.tool_name, canonicalArgsSha256: row.canonical_args_sha256 })),
    transmittedResults: rows.map((row) => ({ turnOrdinal: row.turn_ordinal, toolName: row.tool_name, resultSha256: row.result_sha256 })),
    rawResults: rows.map((row) => ({ turnOrdinal: row.turn_ordinal, toolName: row.tool_name, rawResultSha256: row.raw_result_sha256 })),
  };
}

function semanticDecision(row: PairRow, side: "control" | "primary"): Record<string, unknown> {
  return {
    outcome: row[`${side}_outcome`],
    incidentClass: row[`${side}_class`],
    severity: row[`${side}_severity`],
    action: row[`${side}_action`],
    abstained: row[`${side}_abstained`],
  };
}

function nullableSequenceAgreement(left: string[], right: string[]): boolean | null {
  return left.length === 0 || right.length === 0 ? null : hashJson(left) === hashJson(right);
}

function summarizeBoolean(values: Array<boolean | null>): { trueCount: number; falseCount: number; nullCount: number; denominator: number; rate: number | null } {
  const trueCount = values.filter((value) => value === true).length;
  const falseCount = values.filter((value) => value === false).length;
  const nullCount = values.length - trueCount - falseCount;
  const denominator = trueCount + falseCount;
  return { trueCount, falseCount, nullCount, denominator, rate: denominator === 0 ? null : round(trueCount / denominator) };
}

function mean(values: Array<number | null>): number | null {
  const present = values.filter((value): value is number => value !== null);
  return present.length === 0 ? null : round(present.reduce((sum, value) => sum + value, 0) / present.length);
}

function groupBy<T>(values: T[], key: (value: T) => string): Map<string, T[]> {
  const output = new Map<string, T[]>();
  for (const value of values) output.set(key(value), [...(output.get(key(value)) ?? []), value]);
  return output;
}

function formatRate(value: { rate: number | null; denominator: number }): string {
  return value.rate === null ? `NA/0` : `${value.rate}/${value.denominator}`;
}

function round(value: number): number { return Math.round(value * 1_000_000) / 1_000_000; }
function requiredArgument(name: string): string {
  const value = argument(name);
  if (value === undefined) throw new Error(`${name} is required`);
  return value;
}
function argument(name: string): string | undefined { const index = process.argv.indexOf(name); return index < 0 ? undefined : process.argv[index + 1]; }
function safeName(value: string): string { return value.replace(/[^a-z0-9_.-]+/gi, "-").toLowerCase(); }

