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
  scenario_variant_id: string;
  scenario_id: string;
}
interface ExpectedRow {
  expected_evidence_id: string;
  scenario_variant_id: string;
  source_kind: string;
  event_name: string | null;
  error_number: number | null;
  minimum_count: number;
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
  paired_token_match: boolean;
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const scheduleName = argument("--schedule") ?? "smoke-v1";
const pool = await connect(config.databases.lab, config.databases.controlName);
const executions = await pool.request()
  .input("run", sql.VarChar(120), run.runId)
  .input("schedule", sql.VarChar(80), scheduleName)
  .query<ExecutionRow>(`
    SELECT x.injection_execution_id,x.schedule_item_id,x.episode_id,x.started_at_utc,
           x.finished_at_utc,x.injector_request_json,x.sql_session_ids_json,x.cleanup_verified,
           v.scenario_variant_id,s.scenario_id
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

const episodeResults: Array<Record<string, unknown>> = [];
try {
  for (const execution of executions.recordset) {
    const manifest = JSON.parse(execution.injector_request_json) as { correlationToken: string; sessionIds: number[] };
    const sessionIds = JSON.parse(execution.sql_session_ids_json) as number[];
    const rules = expected.recordset.filter((row) => row.scenario_variant_id === execution.scenario_variant_id);
    const ruleResults: Array<Record<string, unknown>> = [];
    const linked = new Set<string>();
    let anchorAssigned = false;
    for (const rule of rules) {
      const candidates = await pool.request()
        .input("start", sql.DateTime2(7), new Date(execution.started_at_utc.getTime() - 1_000))
        .input("finish", sql.DateTime2(7), new Date(execution.finished_at_utc.getTime() + 3_000))
        .input("source", sql.VarChar(40), rule.source_kind)
        .input("event", sql.VarChar(120), rule.event_name)
        .input("error", sql.Int, rule.error_number)
        .input("token", sql.VarChar(32), manifest.correlationToken.slice(0, 12))
        .query<CandidateRow>(`
          SELECT canonical_event_id,source_kind,source_event_name,error_number,occurred_at_utc,
                 message_raw,client_app_name,session_id,
                 CONVERT(bit,CASE WHEN EXISTS
                 (
                   SELECT 1 FROM ingest.canonical_events AS paired
                   WHERE paired.source_kind=event.source_kind
                     AND ABS(DATEDIFF_BIG(MILLISECOND,paired.occurred_at_utc,event.occurred_at_utc)) <= 100
                     AND CHARINDEX(@token,COALESCE(paired.message_raw,N'')) > 0
                 ) THEN 1 ELSE 0 END) AS paired_token_match
          FROM ingest.canonical_events AS event
          WHERE occurred_at_utc BETWEEN @start AND @finish
            AND source_kind=@source
            AND (@event IS NULL OR source_event_name=@event)
            AND (@error IS NULL OR error_number=@error)
          ORDER BY occurred_at_utc,canonical_event_id;
        `);
      const matched = candidates.recordset.filter((candidate) => {
        const tokenMatch = candidate.message_raw?.includes(manifest.correlationToken.slice(0, 12)) === true ||
          candidate.client_app_name?.includes(manifest.correlationToken.slice(0, 8)) === true;
        const sessionMatch = candidate.session_id !== null && sessionIds.includes(candidate.session_id);
        return tokenMatch || sessionMatch || candidate.paired_token_match;
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
        matchedCount: matched.length,
        passed: !rule.required || matched.length >= rule.minimum_count,
      });
    }
    const passed = execution.cleanup_verified && ruleResults.every((row) => row.passed === true);
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
      cleanupVerified: execution.cleanup_verified,
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
  episodeCount: episodeResults.length,
  passedEpisodes,
  failedEpisodes: episodeResults.length - passedEpisodes,
  expectedCaptureRate: episodeResults.length === 0 ? null : passedEpisodes / episodeResults.length,
  episodes: episodeResults,
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/capture/verification.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({
  runId: run.runId,
  scheduleName,
  episodeCount: receipt.episodeCount,
  passedEpisodes: receipt.passedEpisodes,
  failedEpisodes: receipt.failedEpisodes,
  expectedCaptureRate: receipt.expectedCaptureRate,
  receiptSha256: receipt.receiptSha256,
}, null, 2));
if (receipt.failedEpisodes > 0) throw new Error(`${receipt.failedEpisodes} capture episode(s) failed verification`);

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
