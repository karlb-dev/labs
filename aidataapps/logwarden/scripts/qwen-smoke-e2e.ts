import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";

const profileKey = "qwen-smoke";
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const evidencePaths = {
  port: `${runDirectory}/metrics/chat-port-gate-qwen-smoke.json`,
  replay: `${runDirectory}/metrics/replay-qwen-smoke-dev.json`,
  scores: `${runDirectory}/metrics/prediction-scores-dev-qwen-smoke.json`,
  baseline: `${runDirectory}/metrics/baseline-B1-dev.json`,
};
const evidence = {
  port: await validatedReceipt(evidencePaths.port, "PASS"),
  replay: await validatedReceipt(evidencePaths.replay, "PASS"),
  scores: await validatedReceipt(evidencePaths.scores, "PASS"),
  baseline: await validatedReceipt(evidencePaths.baseline, "PASS"),
};
const replayStarted = dateValue(evidence.replay.startedAtUtc, "replay.startedAtUtc");
const replayFinished = dateValue(evidence.replay.finishedAtUtc, "replay.finishedAtUtc");
if (replayStarted >= replayFinished) throw new Error("Qwen replay time window is invalid");
const pool = await connect(config.databases.lab, config.databases.controlName, 600_000);

try {
  const expectedRag = await pool.request().input("run", sql.VarChar(120), run.runId).query<{ expected_rag: number }>(`
    SELECT COUNT(*) expected_rag FROM eval.ground_truth_episodes truth
    INNER JOIN workload.injection_executions execution ON execution.episode_id=truth.episode_id AND execution.run_id=@run
    INNER JOIN workload.schedule_items schedule_item ON schedule_item.schedule_item_id=execution.schedule_item_id
    INNER JOIN workload.schedules schedule ON schedule.schedule_id=schedule_item.schedule_id
    WHERE schedule.schedule_name='standard-v1' AND truth.split_role='dev' AND truth.expected_runbooks_json<>N'[]';
  `);
  const expectedRagCount = Number(expectedRag.recordset[0]?.expected_rag ?? 0);
  const observed = await pool.request().input("profile", sql.VarChar(80), profileKey)
    .input("run", sql.VarChar(120), run.runId).input("started", sql.DateTime2(7), replayStarted)
    .input("finished", sql.DateTime2(7), replayFinished)
    .query<Record<string, unknown>>(`
      SELECT
        (SELECT COUNT(*) FROM eval.predictions prediction INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=prediction.episode_id WHERE prediction.model_profile_id=@profile AND prediction.agent_arm_id='A-direct' AND truth.split_role='dev') direct_predictions,
        (SELECT COUNT(*) FROM eval.predictions prediction INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=prediction.episode_id WHERE prediction.model_profile_id=@profile AND prediction.agent_arm_id='A-tools' AND truth.split_role='dev') tools_predictions,
        (SELECT COUNT(*) FROM eval.predictions prediction INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=prediction.episode_id WHERE prediction.model_profile_id=@profile AND prediction.agent_arm_id='A-rag' AND truth.split_role='dev') rag_predictions,
        (SELECT COUNT(*) FROM eval.predictions prediction INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=prediction.episode_id WHERE prediction.agent_arm_id='B1-rules-v1' AND prediction.model_profile_id IS NULL AND truth.split_role='dev') baseline_predictions,
        (SELECT COUNT(*) FROM eval.predictions prediction INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=prediction.episode_id INNER JOIN eval.decision_scores score ON score.prediction_id=prediction.prediction_id WHERE truth.split_role='dev' AND (prediction.model_profile_id=@profile OR (prediction.model_profile_id IS NULL AND prediction.agent_arm_id='B1-rules-v1'))) scored_predictions,
        (SELECT COUNT(*) FROM agent.agent_runs agent_run INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=agent_run.episode_id WHERE agent_run.model_profile_id=@profile AND truth.split_role='dev') agent_runs,
        (SELECT COUNT(*) FROM agent.agent_runs agent_run INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=agent_run.episode_id WHERE agent_run.model_profile_id=@profile AND truth.split_role='dev' AND agent_run.status='running') open_agent_runs,
        (SELECT COUNT(*) FROM agent.turns turn INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=turn.agent_run_id INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=agent_run.episode_id WHERE agent_run.model_profile_id=@profile AND truth.split_role='dev') turns,
        (SELECT COUNT(*) FROM agent.model_requests request INNER JOIN agent.turns turn ON turn.turn_id=request.turn_id INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=turn.agent_run_id INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=agent_run.episode_id WHERE agent_run.model_profile_id=@profile AND truth.split_role='dev') requests,
        (SELECT COUNT(*) FROM agent.model_responses response INNER JOIN agent.model_requests request ON request.model_request_id=response.model_request_id INNER JOIN agent.turns turn ON turn.turn_id=request.turn_id INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=turn.agent_run_id INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=agent_run.episode_id WHERE agent_run.model_profile_id=@profile AND truth.split_role='dev') responses,
        (SELECT COUNT(*) FROM agent.model_responses response INNER JOIN agent.model_requests request ON request.model_request_id=response.model_request_id INNER JOIN agent.turns turn ON turn.turn_id=request.turn_id INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=turn.agent_run_id INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=agent_run.episode_id WHERE agent_run.model_profile_id=@profile AND truth.split_role='dev' AND (response.prompt_tokens IS NULL OR response.completion_tokens IS NULL OR response.total_tokens IS NULL)) missing_token_rows,
        (SELECT COUNT(*) FROM agent.model_responses response INNER JOIN agent.model_requests request ON request.model_request_id=response.model_request_id INNER JOIN agent.turns turn ON turn.turn_id=request.turn_id INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=turn.agent_run_id INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=agent_run.episode_id WHERE agent_run.model_profile_id=@profile AND truth.split_role='dev' AND ((response.reasoning_content IS NULL AND (response.reasoning_sha256 IS NOT NULL OR response.reasoning_bytes IS NOT NULL)) OR (response.reasoning_content IS NOT NULL AND (response.reasoning_sha256 IS NULL OR response.reasoning_bytes IS NULL)))) reasoning_provenance_failures,
        (SELECT COUNT(*) FROM agent.tool_invocations tool INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=tool.agent_run_id INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=agent_run.episode_id WHERE agent_run.model_profile_id=@profile AND truth.split_role='dev') tool_calls,
        (SELECT COUNT(*) FROM agent.tool_invocations tool INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=tool.agent_run_id INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=agent_run.episode_id WHERE agent_run.model_profile_id=@profile AND truth.split_role='dev' AND tool.retrieval_run_id IS NOT NULL) retrieval_tool_calls,
        (SELECT COUNT(*) FROM telemetry.spans span INNER JOIN telemetry.traces trace ON trace.trace_id=span.trace_id INNER JOIN control.jobs job ON job.job_id=trace.job_id WHERE job.model_profile_id=@profile AND span.span_name='model.request' AND span.model_request_id IS NULL) unlinked_model_spans,
        (SELECT COUNT(*) FROM telemetry.spans span INNER JOIN telemetry.traces trace ON trace.trace_id=span.trace_id INNER JOIN control.jobs job ON job.job_id=trace.job_id WHERE job.model_profile_id=@profile AND span.span_name='tool.operation' AND span.tool_invocation_id IS NULL) unlinked_tool_spans,
        (SELECT COUNT(*) FROM telemetry.spans span INNER JOIN telemetry.traces trace ON trace.trace_id=span.trace_id INNER JOIN control.jobs job ON job.job_id=trace.job_id WHERE job.model_profile_id=@profile AND span.finished_at_utc IS NULL) open_spans,
        (SELECT COUNT(*) FROM telemetry.journal_events event INNER JOIN control.jobs job ON job.job_id=event.job_id WHERE job.model_profile_id=@profile) journal_events,
        (SELECT COUNT(*) FROM telemetry.model_service_samples WHERE run_id=@run AND model_profile_id=@profile AND phase='qwen-smoke-replay' AND sampled_at_utc BETWEEN @started AND @finished) service_samples,
        (SELECT COUNT(*) FROM telemetry.gpu_samples WHERE run_id=@run AND sampled_at_utc BETWEEN @started AND @finished) gpu_samples,
        (SELECT MAX(memory_used_mib) FROM telemetry.gpu_samples WHERE run_id=@run AND sampled_at_utc BETWEEN @started AND @finished) max_gpu_memory_mib,
        (SELECT COUNT(*) FROM ops.action_proposals proposal INNER JOIN agent.decisions decision ON decision.decision_id=proposal.decision_id INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=decision.agent_run_id WHERE agent_run.model_profile_id=@profile AND (proposal.caller_opted_in=1 OR proposal.executed_at_utc IS NOT NULL OR proposal.execution_status<>'not_executed')) unsafe_actions;
    `);
  const row = observed.recordset[0]!;
  const expectedAgentCells = 120 + expectedRagCount;
  const expectedScored = 180 + expectedRagCount;
  const checks = {
    completeCellGrid: Number(row.direct_predictions) === 60 && Number(row.tools_predictions) === 60
      && Number(row.rag_predictions) === expectedRagCount && Number(row.baseline_predictions) === 60,
    completeScores: Number(row.scored_predictions) === expectedScored,
    agentRunCoverage: Number(row.agent_runs) === expectedAgentCells && Number(row.open_agent_runs) === 0,
    requestResponseParity: Number(row.turns) === Number(row.requests) && Number(row.requests) === Number(row.responses) && Number(row.requests) >= expectedAgentCells,
    tokenAndReasoningProvenance: Number(row.missing_token_rows) === 0 && Number(row.reasoning_provenance_failures) === 0,
    realToolPath: Number(row.tool_calls) > 0 && Number(row.retrieval_tool_calls) > 0,
    telemetryLinkedAndClosed: Number(row.unlinked_model_spans) === 0 && Number(row.unlinked_tool_spans) === 0 && Number(row.open_spans) === 0 && Number(row.journal_events) > Number(row.requests),
    continuousServiceTelemetry: Number(row.service_samples) >= 2 && Number(row.gpu_samples) >= 2 && Number(row.max_gpu_memory_mib) >= 8_000,
    safety: Number(row.unsafe_actions) === 0,
  };
  if (Object.values(checks).some((value) => !value)) throw new Error(`Qwen end-to-end gate failed: ${JSON.stringify({ checks, observed: row, expectedRagCount })}`);

  const pairsResult = await pool.request().input("profile", sql.VarChar(80), profileKey)
    .input("run", sql.VarChar(120), run.runId).query<{
    episode_id: string; scenario_group_id: string; baseline_success: boolean; treatment_success: boolean;
  }>(`
    SELECT truth.episode_id,truth.scenario_group_id,baseline_score.action_correct baseline_success,
      treatment_score.action_correct treatment_success
    FROM eval.ground_truth_episodes truth
    INNER JOIN eval.predictions baseline ON baseline.episode_id=truth.episode_id AND baseline.agent_arm_id='B1-rules-v1' AND baseline.model_profile_id IS NULL
    INNER JOIN eval.decision_scores baseline_score ON baseline_score.prediction_id=baseline.prediction_id
    INNER JOIN eval.predictions treatment ON treatment.episode_id=truth.episode_id AND treatment.agent_arm_id='A-tools' AND treatment.model_profile_id=@profile
    INNER JOIN eval.decision_scores treatment_score ON treatment_score.prediction_id=treatment.prediction_id
    INNER JOIN workload.injection_executions execution ON execution.episode_id=truth.episode_id AND execution.run_id=@run
    INNER JOIN workload.schedule_items schedule_item ON schedule_item.schedule_item_id=execution.schedule_item_id
    INNER JOIN workload.schedules schedule ON schedule.schedule_id=schedule_item.schedule_id
    WHERE schedule.schedule_name='standard-v1' AND truth.split_role='dev'
    ORDER BY truth.scenario_group_id,truth.episode_id;
  `);
  const pairRows = pairsResult.recordset.map((pair) => ({
    episode_id: pair.episode_id,
    scenario_group_id: pair.scenario_group_id,
    baseline_success: Number(pair.baseline_success),
    treatment_success: Number(pair.treatment_success),
  }));
  if (pairRows.length !== 60 || new Set(pairRows.map((pair) => pair.scenario_group_id)).size < 4) throw new Error("Qwen dev power pairs are incomplete");
  const pairsBody = {
    schemaVersion: 1,
    runId: run.runId,
    baseline: "B1-rules-v1",
    treatmentProfile: profileKey,
    treatmentArm: "A-tools",
    outcome: "acceptable_action_correct",
    rows: pairRows,
    orderedRowsSha256: hashJson(pairRows),
  };
  const pairsReceipt = { ...pairsBody, receiptSha256: hashJson(pairsBody) };
  const pairsPath = `${runDirectory}/metrics/power-dev-pairs.json`;
  await atomicWrite(pairsPath, `${JSON.stringify(pairsReceipt, null, 2)}\n`, 0o600);
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    profileKey,
    verifiedAtUtc: new Date().toISOString(),
    replayWindow: { startedAtUtc: replayStarted.toISOString(), finishedAtUtc: replayFinished.toISOString() },
    expectedRagCount,
    expectedAgentCells,
    evidence: Object.fromEntries(Object.entries(evidence).map(([key, value]) => [key, {
      path: evidencePaths[key as keyof typeof evidencePaths], receiptSha256: value.receiptSha256,
    }])),
    checks,
    observed: row,
    powerPairs: { path: pairsPath, rows: pairRows.length, groupCount: new Set(pairRows.map((pair) => pair.scenario_group_id)).size, receiptSha256: pairsReceipt.receiptSha256 },
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const receiptPath = `${runDirectory}/metrics/qwen-smoke-e2e.json`;
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  await persistEvidence(receipt);
  await appendExperimentLog(`Real qwen-smoke end-to-end replay gate passed ${expectedAgentCells} dev agent cells plus 60 B1 rows with complete raw/SQL/trace/service/GPU provenance; empirical power pairs ${pairsReceipt.receiptSha256}, gate ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({ runId: run.runId, profileKey, expectedAgentCells, expectedRagCount, checks, powerPairsPath: pairsPath, powerPairsReceiptSha256: pairsReceipt.receiptSha256, receiptPath, receiptSha256: receipt.receiptSha256, disposition: "PASS" }, null, 2));
} finally {
  await pool.close();
}

async function persistEvidence(detail: unknown): Promise<void> {
  const detailJson = canonicalJson(detail);
  await pool.request().input("key", sql.VarChar(120), `qwen-smoke-e2e-${String((detail as { receiptSha256: string }).receiptSha256).slice(0, 24)}`)
    .input("run", sql.VarChar(120), run.runId).input("detail", sql.NVarChar(sql.MAX), detailJson)
    .input("hash", sql.Char(64), hashJson(JSON.parse(detailJson))).query(`
      INSERT control.evidence_events(event_key,run_id,stage,scientific_tier,disposition,detail_json,detail_sha256)
      VALUES(@key,@run,'qwen_smoke_e2e','tier1','PASS',@detail,@hash);
    `);
}

async function validatedReceipt(path: string, disposition: string): Promise<Record<string, unknown>> {
  const parsed = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
  const receiptSha256 = parsed.receiptSha256;
  if (typeof receiptSha256 !== "string") throw new Error(`Receipt has no hash: ${path}`);
  const { receiptSha256: _ignored, ...body } = parsed;
  if (hashJson(body) !== receiptSha256 || parsed.disposition !== disposition) throw new Error(`Receipt failed validation: ${path}`);
  return parsed;
}

function dateValue(value: unknown, label: string): Date {
  if (typeof value !== "string") throw new Error(`${label} is missing`);
  const date = new Date(value);
  if (!Number.isFinite(date.valueOf())) throw new Error(`${label} is invalid`);
  return date;
}
