import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

interface RunManifest { runId: string }
interface ExecutionRow {
  injection_execution_id: string;
  schedule_item_id: string;
  episode_id: string;
  started_at_utc: Date;
  finished_at_utc: Date;
  injector_request_json: string;
  sql_session_ids_json: string;
  cleanup_verified: boolean;
  return_code: number | null;
  scenario_variant_id: string;
  scenario_id: string;
  config_json: string;
}
interface ExpectedRow {
  expected_evidence_id: string;
  scenario_variant_id: string;
  source_kind: string;
  event_name: string | null;
  error_number: number | null;
  minimum_count: number;
  maximum_count: number | null;
  required: boolean;
}
interface CandidateRow {
  canonical_event_id: string;
  source_kind: string;
  source_event_name: string;
  error_number: number | null;
  occurred_at_utc: Date;
  message_raw: string | null;
  client_app_name: string | null;
  session_id: number | null;
  raw_token_match: boolean;
}
interface SnapshotRow {
  tool_id: string;
  canonical_args_sha256: string;
  result_sha256: string;
  status: string;
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const scheduleName = argument("--schedule") ?? "smoke-v1";
const allowPartial = process.argv.includes("--allow-partial");
const pool = await connect(config.databases.lab, config.databases.controlName);
const executions = await pool.request()
  .input("run", sql.VarChar(120), run.runId)
  .input("schedule", sql.VarChar(80), scheduleName)
  .query<ExecutionRow>(`
    SELECT x.injection_execution_id,x.schedule_item_id,x.episode_id,x.started_at_utc,
           x.finished_at_utc,x.injector_request_json,x.sql_session_ids_json,x.cleanup_verified,
           x.return_code,v.scenario_variant_id,s.scenario_id,s.config_json
    FROM workload.injection_executions AS x
    INNER JOIN workload.schedule_items AS i ON i.schedule_item_id=x.schedule_item_id
    INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=i.schedule_id
    INNER JOIN workload.scenario_variants AS v ON v.scenario_variant_id=i.scenario_variant_id
    INNER JOIN workload.scenario_definitions AS s ON s.scenario_id=v.scenario_id
    WHERE x.run_id=@run AND schedule.schedule_name=@schedule
    ORDER BY i.ordinal;
  `);
const expected = await pool.request()
  .input("schedule", sql.VarChar(80), scheduleName)
  .query<ExpectedRow>(`
    SELECT DISTINCT e.*
    FROM workload.expected_evidence AS e
    INNER JOIN workload.schedule_items AS i ON i.scenario_variant_id=e.scenario_variant_id
    INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=i.schedule_id
    WHERE schedule.schedule_name=@schedule
    ORDER BY e.scenario_variant_id,e.expected_evidence_id;
  `);
const scheduleCount = await pool.request().input("schedule", sql.VarChar(80), scheduleName)
  .query<{ item_count: number }>(`
    SELECT COUNT(*) AS item_count FROM workload.schedule_items AS item
    INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=item.schedule_id
    WHERE schedule.schedule_name=@schedule;
  `);
const expectedExecutionCount = Number(scheduleCount.recordset[0]?.item_count ?? 0);
if (!allowPartial && executions.recordset.length !== expectedExecutionCount)
  throw new Error(`Schedule ${scheduleName} has ${executions.recordset.length}/${expectedExecutionCount} injection executions`);

const episodeResults: Array<Record<string, unknown>> = [];
try {
  for (const execution of executions.recordset) {
    const manifest = JSON.parse(execution.injector_request_json) as {
      correlationToken: string;
      sessionIds: number[];
      snapshots?: Array<{ tool: string; argumentsSha256: string; resultSha256: string }>;
    };
    const scenarioConfig = JSON.parse(execution.config_json) as { snapshotPlan?: unknown[] };
    const plannedSnapshotCount = scenarioConfig.snapshotPlan?.length ?? 0;
    const snapshotRows = await pool.request().input("episode", sql.VarChar(120), execution.episode_id)
      .query<SnapshotRow>(`
        SELECT tool_id,canonical_args_sha256,result_sha256,status
        FROM ingest.context_snapshots WHERE episode_id=@episode
        ORDER BY tool_id,canonical_args_sha256;
      `);
    const manifestedSnapshots = manifest.snapshots ?? [];
    const snapshotPassed = snapshotRows.recordset.length === plannedSnapshotCount &&
      manifestedSnapshots.length === plannedSnapshotCount &&
      snapshotRows.recordset.every((row) => row.status === "complete" && manifestedSnapshots.some((item) =>
        item.tool === row.tool_id && item.argumentsSha256 === row.canonical_args_sha256 && item.resultSha256 === row.result_sha256));
    const rules = expected.recordset.filter((row) => row.scenario_variant_id === execution.scenario_variant_id);
    const ruleResults: Array<Record<string, unknown>> = [];
    const linked = new Set<string>();
    let anchorAssigned = false;
    await pool.request()
      .input("injection", sql.BigInt, execution.injection_execution_id)
      .query("DELETE FROM ingest.injection_event_links WHERE injection_execution_id=@injection;");
    for (const rule of rules) {
      const candidates = await pool.request()
        .input("start", sql.DateTime2(7), new Date(execution.started_at_utc.getTime() - 1_000))
        .input("finish", sql.DateTime2(7), new Date(execution.finished_at_utc.getTime() + 3_000))
        .input("source", sql.VarChar(40), rule.source_kind)
        .input("event", sql.VarChar(120), rule.event_name)
        .input("error", sql.Int, rule.error_number)
        .input("token", sql.VarChar(32), manifest.correlationToken)
        .query<CandidateRow>(`
          SELECT canonical_event_id,source_kind,source_event_name,error_number,occurred_at_utc,
                 message_raw,client_app_name,session_id,
                 CONVERT(bit,CASE WHEN
                   CHARINDEX(@token,COALESCE(CONVERT(nvarchar(max),raw.raw_payload_xml),raw.raw_payload_text,raw.raw_payload_json,N'')) > 0
                   THEN 1 ELSE 0 END) AS raw_token_match
          FROM ingest.canonical_events AS event
          INNER JOIN ingest.raw_events AS raw ON raw.raw_event_id=event.raw_event_id
          WHERE occurred_at_utc BETWEEN @start AND @finish
            AND source_kind=@source
            AND (@event IS NULL OR source_event_name=@event)
            AND (@error IS NULL OR error_number=@error)
          ORDER BY occurred_at_utc,canonical_event_id;
        `);
      const matched = candidates.recordset.filter((candidate) => {
        if (rule.event_name === "rpc_completed" || rule.event_name === "sql_batch_completed")
          return candidate.raw_token_match;
        const tokenMatch = candidate.message_raw?.includes(manifest.correlationToken.slice(0, 12)) === true ||
          candidate.client_app_name?.includes(manifest.correlationToken.slice(0, 12)) === true;
        return tokenMatch || candidate.raw_token_match;
      });
      for (const candidate of matched) {
        if (linked.has(candidate.canonical_event_id)) continue;
        await pool.request()
          .input("injection", sql.BigInt, execution.injection_execution_id)
          .input("event", sql.BigInt, candidate.canonical_event_id)
          .input("role", sql.VarChar(40), anchorAssigned ? "supporting" : "anchor")
          .input("rule", sql.VarChar(80), `expected-${rule.expected_evidence_id}`)
          .query(`
            IF NOT EXISTS
              (SELECT 1 FROM ingest.injection_event_links WHERE injection_execution_id=@injection AND canonical_event_id=@event)
              INSERT ingest.injection_event_links
                (injection_execution_id,canonical_event_id,match_role,match_score,match_rule_id,audited)
              VALUES(@injection,@event,@role,1.0,@rule,0);
          `);
        linked.add(candidate.canonical_event_id);
        anchorAssigned = true;
      }
      ruleResults.push({
        expectedEvidenceId: rule.expected_evidence_id,
        source: rule.source_kind,
        event: rule.event_name,
        errorNumber: rule.error_number,
        required: rule.required,
        minimumCount: rule.minimum_count,
        maximumCount: rule.maximum_count,
        matchedCount: matched.length,
        passed: (!rule.required || matched.length >= rule.minimum_count) &&
          (rule.maximum_count === null || matched.length <= rule.maximum_count),
      });
    }
    const passed = execution.return_code === 0 && execution.cleanup_verified && snapshotPassed &&
      ruleResults.every((row) => row.passed === true);
    await pool.request()
      .input("id", sql.BigInt, execution.injection_execution_id)
      .input("verified", sql.Bit, passed)
      .input("item", sql.BigInt, execution.schedule_item_id)
      .query(`
        UPDATE workload.injection_executions SET verified=@verified WHERE injection_execution_id=@id;
        UPDATE workload.schedule_items SET status=CASE WHEN @verified=1 THEN 'complete' ELSE 'STOP_DATA' END
        WHERE schedule_item_id=@item;
      `).catch(async (error) => {
        // schedule_items has no STOP_DATA enum constraint, but retain a clear
        // fallback if a future migration tightens it.
        if ((error as { number?: number }).number !== 547) throw error;
        await pool.request().input("id", sql.BigInt, execution.injection_execution_id).input("verified", sql.Bit, passed)
          .query("UPDATE workload.injection_executions SET verified=@verified WHERE injection_execution_id=@id;");
      });
    episodeResults.push({
      injectionExecutionId: execution.injection_execution_id,
      episodeId: execution.episode_id,
      scenarioId: execution.scenario_id,
      returnCode: execution.return_code,
      cleanupVerified: execution.cleanup_verified,
      plannedSnapshotCount,
      capturedSnapshotCount: snapshotRows.recordset.length,
      snapshotPassed,
      linkedEventCount: linked.size,
      rules: ruleResults,
      passed,
    });
  }
} finally {
  await pool.close();
}

const passedEpisodes = episodeResults.filter((row) => row.passed === true).length;
const receiptBody = {
  schemaVersion: 1,
  runId: run.runId,
  scheduleName,
  allowPartial,
  expectedExecutionCount,
  episodeCount: episodeResults.length,
  passedEpisodes,
  failedEpisodes: episodeResults.length - passedEpisodes,
  expectedCaptureRate: episodeResults.length === 0 ? null : passedEpisodes / episodeResults.length,
  episodes: episodeResults,
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
const receiptName = scheduleName === "smoke-v1" ? "verification.json" : `verification-${safeName(scheduleName)}.json`;
await atomicWrite(`${runDirectory}/capture/${receiptName}`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({
  runId: run.runId,
  scheduleName,
  episodeCount: receipt.episodeCount,
  passedEpisodes: receipt.passedEpisodes,
  failedEpisodes: receipt.failedEpisodes,
  expectedCaptureRate: receipt.expectedCaptureRate,
  receiptPath: `capture/${receiptName}`,
  receiptSha256: receipt.receiptSha256,
}, null, 2));
if (receipt.failedEpisodes > 0) throw new Error(`${receipt.failedEpisodes} capture episode(s) failed verification`);

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}

function safeName(value: string): string {
  if (!/^[a-z0-9-]{1,80}$/i.test(value)) throw new Error(`Unsafe schedule name: ${value}`);
  return value.toLowerCase();
}
