import { readFile, readdir, stat } from "node:fs/promises";
import { relative, resolve } from "node:path";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashFile, hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";

interface QueryDefinition {
  name: string;
  sql: string;
}

interface Artifact {
  path: string;
  kind: "query" | "evidence";
  rows: number | null;
  bytes: number;
  sha256: string;
  querySha256?: string;
  sourcePath?: string;
}

const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const requestedOutput = valueAfter("--out") ?? `${runDirectory}/repro/rows`;
const outputDirectory = resolve(requestedOutput);
const outputRelative = relative(resolve(`${runDirectory}/repro`), outputDirectory);
if (outputRelative.startsWith("..") || outputRelative.startsWith("/")) {
  throw new Error(`Report-row output must stay inside ${runDirectory}/repro: ${outputDirectory}`);
}
const config = loadConfig();
const controlDatabase = valueAfter("--control-database") ?? config.databases.controlName;
const queries = queryDefinitions();
const artifacts: Artifact[] = [];
const pool = await connect(config.databases.admin, controlDatabase, 600_000, 4);
try {
  for (const definition of queries) {
    const result = await pool.request().input("run", sql.VarChar(120), run.runId).query<Record<string, unknown>>(definition.sql);
    const body = result.recordset.map((row) => canonicalJson(normalize(row))).join("\n") + (result.recordset.length === 0 ? "" : "\n");
    const path = `${outputDirectory}/${definition.name}.jsonl`;
    await atomicWrite(path, body, 0o600);
    artifacts.push(await artifact(path, "query", result.recordset.length, { querySha256: sha256(definition.sql) }));
  }
} finally {
  await pool.close();
}

for (const sourcePath of await evidencePaths()) {
  const source = await readFile(`${runDirectory}/${sourcePath}`);
  const path = `${outputDirectory}/evidence/${sourcePath.replaceAll("/", "__")}`;
  await atomicWrite(path, source, 0o600);
  artifacts.push(await artifact(path, "evidence", null, { sourcePath }));
}

artifacts.sort((left, right) => left.path.localeCompare(right.path));
const manifestBody = {
  schemaVersion: 1,
  runId: run.runId,
  controlDatabaseRole: "LogWardenControl",
  queryBundleSha256: hashJson(queries.map(({ name, sql: query }) => ({ name, querySha256: sha256(query) }))),
  artifacts,
};
const manifest = { ...manifestBody, receiptSha256: hashJson(manifestBody) };
await atomicWrite(`${outputDirectory}/manifest.json`, `${JSON.stringify(manifest, null, 2)}\n`, 0o600);
console.log(JSON.stringify({
  runId: run.runId,
  controlDatabase,
  outputDirectory,
  queryCount: queries.length,
  artifactCount: artifacts.length,
  rowCount: artifacts.reduce((total, value) => total + (value.rows ?? 0), 0),
  queryBundleSha256: manifest.queryBundleSha256,
  receiptSha256: manifest.receiptSha256,
  disposition: "PASS",
}, null, 2));

async function artifact(
  path: string,
  kind: Artifact["kind"],
  rows: number | null,
  extra: Pick<Artifact, "querySha256" | "sourcePath">,
): Promise<Artifact> {
  const information = await stat(path);
  return {
    path: relative(outputDirectory, path), kind, rows, bytes: information.size, sha256: await hashFile(path),
    ...(extra.querySha256 === undefined ? {} : { querySha256: extra.querySha256 }),
    ...(extra.sourcePath === undefined ? {} : { sourcePath: extra.sourcePath }),
  };
}

async function evidencePaths(): Promise<string[]> {
  const fixed = [
    "manifests/freeze.json",
    "manifests/control-subset.json",
    "manifests/search-freeze.json",
    "packets/build.json",
    "packets/leakage-audit.json",
    "capture/verification-standard-v1.json",
    "knowledge/retrieval-evaluation-dev-calibration-test_id-test_variant_holdout-test_unknown.json",
    "security/tool-security-gate.json",
    "telemetry/reconciliation.json",
    "metrics/campaign-analysis.json",
    "metrics/model-dispositions.json",
    "metrics/power-analysis.json",
  ];
  const metricEntries = await readdir(`${runDirectory}/metrics`, { withFileTypes: true });
  const patterns = [
    /^calibration-(muse-glimmer-30b|gemma-4-31b|qwen-3\.8-27b)\.json$/,
    /^control-comparison-(muse-glimmer-30b|gemma-4-31b|qwen-3\.8-27b)-.+\.json$/,
    /^batching-diagnostic-(muse-glimmer-30b|gemma-4-31b|qwen-3\.8-27b)\.json$/,
    /^chat-port-gate-(muse-glimmer-30b|gemma-4-31b|qwen-3\.8-27b)\.json$/,
    /^chat-port-gate-guided-json-olmo-3\.1-32b-instruct\.json$/,
  ];
  for (const entry of metricEntries) if (entry.isFile() && patterns.some((pattern) => pattern.test(entry.name))) {
    fixed.push(`metrics/${entry.name}`);
  }
  const olmoRoot = `${runDirectory}/raw/chat-port-gate/olmo-3.1-32b-instruct`;
  for (const entry of await readdir(olmoRoot, { withFileTypes: true })) if (entry.isDirectory()) {
    const path = `raw/chat-port-gate/olmo-3.1-32b-instruct/${entry.name}/failure-receipt.json`;
    if (await readFile(`${runDirectory}/${path}`).then(() => true).catch(() => false)) fixed.push(path);
  }
  return [...new Set(fixed)].sort();
}

function normalize(value: unknown): unknown {
  if (value instanceof Date) return value.toISOString();
  if (Array.isArray(value)) return value.map(normalize);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([key, child]) => [key, normalize(child)]));
  }
  return value;
}

function queryDefinitions(): QueryDefinition[] {
  return [
    {
      name: "primary_predictions",
      sql: `
        WITH current_campaign AS (SELECT campaign_id FROM control.runs WHERE run_id=@run)
        SELECT CONVERT(bigint,p.prediction_id) prediction_id,p.episode_id,p.model_profile_id,p.agent_arm_id,
          p.prediction_source,p.outcome,p.failure_stage,p.failure_class,CONVERT(float,p.confidence) confidence,
          CONVERT(bit,p.abstained) abstained,CONVERT(bit,p.complete_case_eligible) complete_case_eligible,
          t.split_role,t.family,t.regime,t.scenario_group_id,t.expected_class,t.expected_severity,t.expected_action,
          CONVERT(bit,t.should_abstain) should_abstain,p.predicted_class,p.predicted_severity,p.predicted_action,
          CONVERT(bit,s.end_to_end_success) end_to_end_success,CONVERT(bit,s.class_correct) class_correct,
          CONVERT(bit,s.severity_correct) severity_correct,CONVERT(float,s.severity_ordinal_cost) severity_ordinal_cost,
          CONVERT(bit,s.action_correct) action_correct,CONVERT(bit,s.abstention_correct) abstention_correct,
          CONVERT(bit,s.contract_success) contract_success,CONVERT(bit,s.required_tools_satisfied) required_tools_satisfied,
          CONVERT(bit,s.forbidden_tools_avoided) forbidden_tools_avoided,
          CONVERT(bit,s.citation_policy_satisfied) citation_policy_satisfied,CONVERT(float,s.composite_score) composite_score,
          TRY_CONVERT(float,JSON_VALUE(s.detail_json,'$.score.costWeightedLoss')) cost_weighted_loss,
          p.prediction_json,s.detail_json score_detail_json
        FROM current_campaign c INNER JOIN control.jobs j ON j.campaign_id=c.campaign_id
        INNER JOIN eval.predictions p ON p.job_id=j.job_id
        INNER JOIN eval.ground_truth_episodes t ON t.episode_id=p.episode_id
        INNER JOIN eval.decision_scores s ON s.prediction_id=p.prediction_id
        WHERE p.control_id IS NULL AND t.split_role IN ('test_id','test_variant_holdout','test_unknown')
        ORDER BY COALESCE(p.model_profile_id,''),p.agent_arm_id,t.split_role,p.episode_id;`,
    },
    {
      name: "control_predictions",
      sql: `
        WITH current_campaign AS (SELECT campaign_id FROM control.runs WHERE run_id=@run)
        SELECT CONVERT(bigint,p.prediction_id) prediction_id,CONVERT(bigint,p.source_prediction_id) source_prediction_id,
          p.episode_id,p.model_profile_id,p.agent_arm_id,p.control_id,p.outcome,p.failure_stage,p.failure_class,
          CONVERT(float,p.confidence) confidence,CONVERT(bit,p.abstained) abstained,t.split_role,t.family,t.regime,
          CONVERT(bit,s.end_to_end_success) end_to_end_success,CONVERT(bit,s.action_correct) action_correct,
          CONVERT(bit,s.contract_success) contract_success,CONVERT(float,s.composite_score) composite_score,
          TRY_CONVERT(float,JSON_VALUE(s.detail_json,'$.score.costWeightedLoss')) cost_weighted_loss
        FROM current_campaign c INNER JOIN control.jobs j ON j.campaign_id=c.campaign_id
        INNER JOIN eval.predictions p ON p.job_id=j.job_id
        INNER JOIN eval.ground_truth_episodes t ON t.episode_id=p.episode_id
        INNER JOIN eval.decision_scores s ON s.prediction_id=p.prediction_id
        WHERE p.control_id IS NOT NULL
        ORDER BY p.model_profile_id,p.control_id,t.split_role,p.episode_id;`,
    },
    {
      name: "tool_scores",
      sql: `
        SELECT CONVERT(bigint,p.prediction_id) prediction_id,p.model_profile_id,p.agent_arm_id,t.split_role,t.family,t.regime,
          score.tool_name,score.expectation,score.called_count,score.valid_call_count,score.snapshot_miss_count,
          score.tool_error_count,CONVERT(bit,score.correct_use) correct_use,CONVERT(float,score.score) tool_score
        FROM eval.tool_scores score INNER JOIN eval.predictions p ON p.prediction_id=score.prediction_id
        INNER JOIN eval.ground_truth_episodes t ON t.episode_id=p.episode_id
        WHERE p.control_id IS NULL AND t.split_role IN ('test_id','test_variant_holdout','test_unknown')
        ORDER BY COALESCE(p.model_profile_id,''),p.agent_arm_id,t.split_role,p.episode_id,score.tool_name;`,
    },
    {
      name: "retrieval_scores",
      sql: `
        SELECT CONVERT(bigint,p.prediction_id) prediction_id,p.model_profile_id,p.agent_arm_id,t.split_role,t.family,t.regime,
          CONVERT(bit,s.answerable) answerable,CONVERT(bit,s.retrieved_any) retrieved_any,CONVERT(float,s.recall_at_k) recall_at_k,
          CONVERT(float,s.reciprocal_rank) reciprocal_rank,CONVERT(float,s.ndcg_at_k) ndcg_at_k,
          CONVERT(float,s.no_answer_correct) no_answer_correct,CONVERT(float,s.citation_precision) citation_precision,
          CONVERT(float,s.citation_recall) citation_recall,CONVERT(bit,s.grounded) grounded
        FROM eval.retrieval_scores s INNER JOIN eval.predictions p ON p.prediction_id=s.prediction_id
        INNER JOIN eval.ground_truth_episodes t ON t.episode_id=p.episode_id
        WHERE p.control_id IS NULL AND t.split_role IN ('test_id','test_variant_holdout','test_unknown')
        ORDER BY COALESCE(p.model_profile_id,''),p.agent_arm_id,t.split_role,p.episode_id;`,
    },
    {
      name: "metric_results",
      sql: `
        SELECT metric_name,model_profile_id,agent_arm_id,split_role,family,regime,analysis_set,
          CONVERT(float,numerator) numerator,CONVERT(bigint,denominator) denominator,
          CONVERT(bigint,eligible_denominator) eligible_denominator,CONVERT(bigint,complete_case_denominator) complete_case_denominator,
          CONVERT(float,metric_value) metric_value,CONVERT(float,standard_error) standard_error,detail_json
        FROM eval.metric_results WHERE run_id=@run
        ORDER BY COALESCE(model_profile_id,''),COALESCE(agent_arm_id,''),COALESCE(split_role,''),COALESCE(family,''),COALESCE(regime,''),metric_name,analysis_set;`,
    },
    {
      name: "paired_contrasts",
      sql: `
        SELECT b.contrast_id,b.metric_name,b.left_cell_json,b.right_cell_json,b.grouped_by,CONVERT(bit,b.paired) paired,
          b.repetitions bootstrap_repetitions,CONVERT(bigint,b.seed) bootstrap_seed,CONVERT(float,b.observed_difference) observed_difference,
          CONVERT(float,b.confidence_level) confidence_level,CONVERT(float,b.ci_low) ci_low,CONVERT(float,b.ci_high) ci_high,
          b.sample_count,p.permutations,CONVERT(bigint,p.seed) permutation_seed,CONVERT(float,p.null_mean) null_mean,
          CONVERT(float,p.null_sd) null_sd,CONVERT(float,p.p_value) permutation_p_value,
          TRY_CONVERT(float,JSON_VALUE(p.detail_json,'$.holmAdjustedPValue')) holm_adjusted_p_value,
          JSON_VALUE(p.detail_json,'$.hypothesisClass') hypothesis_class,b.result_sha256
        FROM eval.bootstrap_results b INNER JOIN eval.permutation_results p
          ON p.run_id=b.run_id AND p.permutation_id=b.contrast_id AND p.metric_name=b.metric_name
        WHERE b.run_id=@run ORDER BY b.contrast_id,b.metric_name;`,
    },
    {
      name: "control_comparisons",
      sql: `
        SELECT control.model_profile_id,control.control_id,control.episode_id,t.split_role,t.family,t.regime,
          CONVERT(bigint,c.control_prediction_id) control_prediction_id,CONVERT(bigint,c.primary_prediction_id) primary_prediction_id,
          CONVERT(bit,c.raw_response_agreement) raw_response_agreement,CONVERT(bit,c.decision_agreement) decision_agreement,
          CONVERT(bit,c.ordered_tool_call_agreement) ordered_tool_call_agreement,
          CONVERT(float,c.action_score_delta) action_score_delta,CONVERT(float,c.confidence_delta) confidence_delta,c.detail_json
        FROM eval.control_comparisons c INNER JOIN eval.predictions control ON control.prediction_id=c.control_prediction_id
        INNER JOIN eval.ground_truth_episodes t ON t.episode_id=control.episode_id
        ORDER BY control.model_profile_id,control.control_id,t.split_role,control.episode_id;`,
    },
    {
      name: "calibration_models",
      sql: `
        SELECT calibration_model_id,model_profile_id,agent_arm_id,fit_role,method,feature_manifest_json,
          coefficients_json,training_row_count,model_sha256
        FROM eval.calibration_models ORDER BY model_profile_id,agent_arm_id,calibration_model_id;`,
    },
    {
      name: "agent_performance",
      sql: `
        SELECT CONVERT(bigint,p.prediction_id) prediction_id,p.model_profile_id,p.agent_arm_id,p.prediction_source,p.outcome,
          t.split_role,t.family,t.regime,CONVERT(float,ar.elapsed_ms) agent_elapsed_ms,ar.status agent_status,
          COALESCE(model.request_count,0) model_request_count,COALESCE(model.response_count,0) model_response_count,
          CONVERT(float,model.client_elapsed_ms) model_client_elapsed_ms,CONVERT(float,model.headers_wait_ms) model_headers_wait_ms,
          CONVERT(float,model.body_read_ms) model_body_read_ms,CONVERT(float,model.parse_ms) parse_ms,
          COALESCE(model.prompt_tokens,0) prompt_tokens,COALESCE(model.completion_tokens,0) completion_tokens,
          COALESCE(model.error_count,0) model_error_count,COALESCE(model.length_finish_count,0) length_finish_count,
          COALESCE(model.repaired_response_count,0) repaired_response_count,COALESCE(model.rejected_response_count,0) rejected_response_count,
          COALESCE(tools.tool_call_count,0) tool_call_count,CONVERT(float,tools.tool_latency_ms) tool_latency_ms,
          COALESCE(tools.snapshot_miss_count,0) snapshot_miss_count,COALESCE(tools.tool_error_count,0) tool_error_count,
          COALESCE(validation.validation_failure_count,0) validation_failure_count
        FROM eval.predictions p INNER JOIN eval.ground_truth_episodes t ON t.episode_id=p.episode_id
        LEFT JOIN agent.agent_runs ar ON ar.agent_run_id=p.agent_run_id
        OUTER APPLY
        (
          SELECT COUNT_BIG(req.model_request_id) request_count,COUNT_BIG(resp.model_response_id) response_count,
            SUM(CONVERT(float,req.client_elapsed_ms)) client_elapsed_ms,SUM(CONVERT(float,req.headers_wait_ms)) headers_wait_ms,
            SUM(CONVERT(float,req.body_read_ms)) body_read_ms,SUM(CONVERT(float,resp.parse_ms)) parse_ms,
            SUM(COALESCE(resp.prompt_tokens,0)) prompt_tokens,SUM(COALESCE(resp.completion_tokens,0)) completion_tokens,
            SUM(CASE WHEN req.status<>'success' OR resp.error_class IS NOT NULL THEN 1 ELSE 0 END) error_count,
            SUM(CASE WHEN resp.finish_reason='length' THEN 1 ELSE 0 END) length_finish_count,
            SUM(CASE WHEN resp.repair_kind IN ('strip_code_fence','strip_leading_text') THEN 1 ELSE 0 END) repaired_response_count,
            SUM(CASE WHEN resp.repair_kind='rejected' OR resp.parse_status<>'success' THEN 1 ELSE 0 END) rejected_response_count
          FROM agent.turns turn_row INNER JOIN agent.model_requests req ON req.turn_id=turn_row.turn_id
          LEFT JOIN agent.model_responses resp ON resp.model_request_id=req.model_request_id
          WHERE turn_row.agent_run_id=ar.agent_run_id
        ) model
        OUTER APPLY
        (
          SELECT COUNT_BIG(*) tool_call_count,SUM(CONVERT(float,tool.latency_ms)) tool_latency_ms,
            SUM(CONVERT(int,tool.snapshot_miss)) snapshot_miss_count,
            SUM(CASE WHEN tool.status<>'success' THEN 1 ELSE 0 END) tool_error_count
          FROM agent.tool_invocations tool WHERE tool.agent_run_id=ar.agent_run_id
        ) tools
        OUTER APPLY
        (
          SELECT SUM(CASE WHEN validation.outcome='fail' THEN 1 ELSE 0 END) validation_failure_count
          FROM agent.validation_events validation WHERE validation.agent_run_id=ar.agent_run_id
        ) validation
        WHERE p.control_id IS NULL AND t.split_role IN ('test_id','test_variant_holdout','test_unknown')
        ORDER BY COALESCE(p.model_profile_id,''),p.agent_arm_id,t.split_role,p.episode_id;`,
    },
    {
      name: "service_performance",
      sql: `
        SELECT sample.model_profile_id,sample.phase,COUNT_BIG(*) sample_count,MIN(sample.sampled_at_utc) first_sample_at_utc,
          MAX(sample.sampled_at_utc) last_sample_at_utc,AVG(CONVERT(float,sample.running_requests)) mean_running_requests,
          MAX(CONVERT(float,sample.running_requests)) max_running_requests,MAX(CONVERT(float,sample.waiting_requests)) max_waiting_requests,
          MAX(CONVERT(float,sample.kv_cache_usage_ratio)) max_kv_cache_usage_ratio,
          MAX(CONVERT(float,sample.preemptions_total))-MIN(CONVERT(float,sample.preemptions_total)) preemption_delta,
          MAX(CONVERT(float,sample.request_errors_total))-MIN(CONVERT(float,sample.request_errors_total)) request_error_delta,
          MAX(CONVERT(float,sample.cancellations_total))-MIN(CONVERT(float,sample.cancellations_total)) cancellation_delta,
          MAX(CONVERT(float,sample.prompt_tokens_total))-MIN(CONVERT(float,sample.prompt_tokens_total)) prompt_token_delta,
          MAX(CONVERT(float,sample.generation_tokens_total))-MIN(CONVERT(float,sample.generation_tokens_total)) generation_token_delta,
          AVG(CONVERT(float,gpu.utilization_gpu_pct)) mean_gpu_utilization_pct,MAX(CONVERT(float,gpu.utilization_gpu_pct)) max_gpu_utilization_pct,
          MAX(CONVERT(float,gpu.memory_used_mib)) max_gpu_memory_used_mib,MAX(CONVERT(float,gpu.power_draw_w)) max_gpu_power_draw_w,
          MAX(CONVERT(float,gpu.temperature_c)) max_gpu_temperature_c
        FROM telemetry.model_service_samples sample
        LEFT JOIN telemetry.gpu_samples gpu ON gpu.run_id=sample.run_id AND gpu.sampled_at_utc=sample.sampled_at_utc
        WHERE sample.run_id=@run AND sample.source_instance='chat-primary' AND sample.model_profile_id IS NOT NULL
        GROUP BY sample.model_profile_id,sample.phase ORDER BY MIN(sample.sampled_at_utc);`,
    },
    {
      name: "retrieval_benchmarks",
      sql: `
        SELECT episode_id,split_role,retrieval_mode,requested_k,CONVERT(float,recall_at_k) recall_at_k,
          CONVERT(float,reciprocal_rank) reciprocal_rank,CONVERT(float,ndcg_at_k) ndcg_at_k,
          CONVERT(float,no_answer_correct) no_answer_correct,result_sha256
        FROM eval.retrieval_benchmark_results WHERE run_id=@run
          AND split_role IN ('test_id','test_variant_holdout','test_unknown')
        ORDER BY retrieval_mode,split_role,episode_id;`,
    },
    {
      name: "safety_audit",
      sql: `SELECT principal_name,check_name,observed_value,expected_value,CONVERT(bit,passed) passed
        FROM reporting.v_safety_audit ORDER BY principal_name,check_name;`,
    },
    {
      name: "run_completeness",
      sql: `
        WITH current_campaign AS (SELECT campaign_id FROM control.runs WHERE run_id=@run)
        SELECT j.run_kind,j.model_profile_id,j.agent_arm_id,j.status,COUNT_BIG(*) job_count
        FROM current_campaign c INNER JOIN control.jobs j ON j.campaign_id=c.campaign_id
        GROUP BY j.run_kind,j.model_profile_id,j.agent_arm_id,j.status
        UNION ALL
        SELECT 'active_queue',NULL,NULL,'nonterminal',COUNT_BIG(*) FROM ops.work_items
        WHERE status NOT IN ('complete','failed','stopped')
        ORDER BY run_kind,model_profile_id,agent_arm_id,status;`,
    },
  ];
}
