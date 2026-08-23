import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { scenarioCatalogManifest, scenarioCatalogSchema } from "../src/scenarios.js";

interface ExecutionRow {
  injection_execution_id: string;
  episode_id: string;
  injector: string;
  injector_request_json: string;
  started_at_utc: Date;
  finished_at_utc: Date;
  return_code: number;
  verified: boolean;
  cleanup_verified: boolean;
  database_name: string | null;
  cleanup_status: string | null;
  dropped_at_utc: Date | null;
  live_database_id: number | null;
  config_json: string;
}
interface SnapshotRow {
  episode_id: string;
  tool_id: string;
  canonical_args_json: string;
  canonical_args_sha256: string;
  result_json: string;
  result_sha256: string;
  row_count: number;
  status: string;
}
interface LinkRow {
  source_event_name: string;
  error_number: number | null;
  raw_sha256: string;
  raw_payload: string;
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const scheduleName = argument("--schedule") ?? "injector-gate-v2";
const safeSchedule = safeName(scheduleName);
const injectionReceipt = await verifiedReceipt(`${runDirectory}/capture/injection-summary-${safeSchedule}.json`);
const verificationReceipt = await verifiedReceipt(`${runDirectory}/capture/verification-${safeSchedule}.json`);
const standardCatalog = scenarioCatalogSchema.parse(JSON.parse(await readFile("config/scenarios/standard-v1.json", "utf8")));
const standardManifestHash = hashJson(scenarioCatalogManifest(standardCatalog, "dev"));

const pool = await connect(config.databases.lab, config.databases.controlName, 30_000);
const checks: Array<Record<string, unknown>> = [];
try {
  const executions = await pool.request()
    .input("run", sql.VarChar(120), run.runId)
    .input("schedule", sql.VarChar(80), scheduleName)
    .query<ExecutionRow>(`
      SELECT execution.injection_execution_id,execution.episode_id,
        REPLACE(scenario.injector_procedure,'driver:','') AS injector,
        execution.injector_request_json,execution.started_at_utc,execution.finished_at_utc,
        execution.return_code,execution.verified,execution.cleanup_verified,scenario.config_json,
        disposable.database_name,disposable.cleanup_status,disposable.dropped_at_utc,
        DB_ID(disposable.database_name) AS live_database_id
      FROM workload.injection_executions AS execution
      INNER JOIN workload.schedule_items AS item ON item.schedule_item_id=execution.schedule_item_id
      INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=item.schedule_id
      INNER JOIN workload.scenario_variants AS variant ON variant.scenario_variant_id=item.scenario_variant_id
      INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
      LEFT JOIN workload.disposable_databases AS disposable
        ON disposable.disposable_database_id=execution.disposable_database_id
      WHERE execution.run_id=@run AND schedule.schedule_name=@schedule
      ORDER BY item.ordinal;
    `);
  const scheduled = await pool.request().input("schedule", sql.VarChar(80), scheduleName)
    .query<{ injector: string }>(`
      SELECT REPLACE(scenario.injector_procedure,'driver:','') AS injector
      FROM workload.schedule_items AS item
      INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=item.schedule_id
      INNER JOIN workload.scenario_variants AS variant ON variant.scenario_variant_id=item.scenario_variant_id
      INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
      WHERE schedule.schedule_name=@schedule ORDER BY item.ordinal;
    `);
  const expectedInjectors = scheduled.recordset.map((row) => row.injector).sort();
  const actualInjectors = executions.recordset.map((row) => row.injector).sort();
  pass("terminal_execution_matrix",
    executions.recordset.length === scheduled.recordset.length && canonicalJson(actualInjectors) === canonicalJson(expectedInjectors) &&
      executions.recordset.every((row) => row.return_code === 0 && row.verified && row.cleanup_verified),
    { episodeCount: executions.recordset.length, injectors: actualInjectors });

  const campaign = await pool.request().input("run", sql.VarChar(120), run.runId)
    .query<{ scenario_manifest_hash: string; status: string; freeze_count: number }>(`
      SELECT campaign.scenario_manifest_hash,campaign.status,
        (SELECT COUNT(*) FROM control.campaign_freezes AS freeze WHERE freeze.campaign_id=campaign.campaign_id) AS freeze_count
      FROM control.campaigns AS campaign
      INNER JOIN control.runs AS run ON run.campaign_id=campaign.campaign_id
      WHERE run.run_id=@run;
    `);
  const campaignRow = campaign.recordset[0]!;
  pass("development_schedule_manifest_isolation",
    campaignRow.status === "building" && Number(campaignRow.freeze_count) === 0 &&
      campaignRow.scenario_manifest_hash === standardManifestHash,
    { campaignStatus: campaignRow.status, freezeCount: Number(campaignRow.freeze_count), manifestHash: campaignRow.scenario_manifest_hash });

  const xe = await pool.request().query<Record<string, unknown>>(`
    SELECT session.event_retention_mode_desc,session.max_dispatch_latency,
      runtime.dropped_event_count,runtime.dropped_buffer_count,target.failed_buffer_count,
      (SELECT TRY_CONVERT(int,field.value) FROM sys.server_event_session_targets AS configured
       INNER JOIN sys.server_event_session_fields AS field
         ON field.event_session_id=configured.event_session_id AND field.object_id=configured.target_id
       WHERE configured.event_session_id=session.event_session_id AND configured.name=N'event_file'
         AND field.name=N'max_file_size') AS max_file_size_mb,
      (SELECT TRY_CONVERT(int,field.value) FROM sys.server_event_session_targets AS configured
       INNER JOIN sys.server_event_session_fields AS field
         ON field.event_session_id=configured.event_session_id AND field.object_id=configured.target_id
       WHERE configured.event_session_id=session.event_session_id AND configured.name=N'event_file'
         AND field.name=N'max_rollover_files') AS max_rollover_files
    FROM sys.server_event_sessions AS session
    INNER JOIN sys.dm_xe_sessions AS runtime ON runtime.name=session.name
    INNER JOIN sys.dm_xe_session_targets AS target
      ON target.event_session_address=runtime.address AND target.target_name=N'event_file'
    WHERE session.name=N'logwarden_capture';
  `);
  const xeRow = xe.recordset[0] ?? {};
  pass("xe_capture_contract_and_loss_counters",
    xe.recordset.length === 1 && xeRow.event_retention_mode_desc === "ALLOW_SINGLE_EVENT_LOSS" &&
      Number(xeRow.max_dispatch_latency) === 1_000 && Number(xeRow.max_file_size_mb) === 16 &&
      Number(xeRow.max_rollover_files) === 20 && Number(xeRow.dropped_event_count) === 0 &&
      Number(xeRow.dropped_buffer_count) === 0 && Number(xeRow.failed_buffer_count) === 0,
    {
      retentionMode: xeRow.event_retention_mode_desc,
      dispatchLatencyMs: xeRow.max_dispatch_latency,
      maxFileSizeMb: xeRow.max_file_size_mb,
      maxRolloverFiles: xeRow.max_rollover_files,
      droppedEvents: xeRow.dropped_event_count,
      droppedBuffers: xeRow.dropped_buffer_count,
      failedTargetBuffers: xeRow.failed_buffer_count,
    });

  const sharedLinks = await pool.request().input("schedule", sql.VarChar(80), scheduleName)
    .query<{ shared_count: number }>(`
      SELECT COUNT(*) AS shared_count FROM
      (
        SELECT link.canonical_event_id
        FROM ingest.injection_event_links AS link
        INNER JOIN workload.injection_executions AS execution
          ON execution.injection_execution_id=link.injection_execution_id
        INNER JOIN workload.schedule_items AS item ON item.schedule_item_id=execution.schedule_item_id
        INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=item.schedule_id
        WHERE schedule.schedule_name=@schedule
        GROUP BY link.canonical_event_id
        HAVING COUNT(DISTINCT execution.injection_execution_id)>1
      ) AS shared;
    `);
  pass("exclusive_event_links", Number(sharedLinks.recordset[0]?.shared_count ?? -1) === 0,
    { sharedCanonicalEvents: Number(sharedLinks.recordset[0]?.shared_count ?? -1) });

  for (const execution of executions.recordset) {
    const manifest = JSON.parse(execution.injector_request_json) as {
      correlationToken: string;
      sessionIds: number[];
      expectedSignalCount: number;
      observedSignalCount: number;
    };
    const links = await pool.request().input("execution", sql.BigInt, execution.injection_execution_id)
      .query<LinkRow>(`
        SELECT event.source_event_name,event.error_number,raw.raw_sha256,
          COALESCE(CONVERT(nvarchar(max),raw.raw_payload_xml),raw.raw_payload_text,raw.raw_payload_json,N'') AS raw_payload
        FROM ingest.injection_event_links AS link
        INNER JOIN ingest.canonical_events AS event ON event.canonical_event_id=link.canonical_event_id
        INNER JOIN ingest.raw_events AS raw ON raw.raw_event_id=event.raw_event_id
        WHERE link.injection_execution_id=@execution
        ORDER BY event.occurred_at_utc,event.canonical_event_id;
      `);
    const correlatedLinks = links.recordset.filter((row) =>
      row.raw_payload.includes(manifest.correlationToken) || row.raw_payload.includes(manifest.correlationToken.slice(0, 12)));
    pass(`raw_token_correlation:${execution.injector}`,
      links.recordset.length > 0 && correlatedLinks.length === links.recordset.length,
      { linkedEvents: links.recordset.length, rawTokenMatches: correlatedLinks.length });
    pass(`driver_signal_count:${execution.injector}`,
      manifest.observedSignalCount === manifest.expectedSignalCount,
      { expected: manifest.expectedSignalCount, observed: manifest.observedSignalCount, sessionCount: manifest.sessionIds.length });

    const snapshots = await pool.request().input("episode", sql.VarChar(120), execution.episode_id)
      .query<SnapshotRow>(`
        SELECT episode_id,tool_id,canonical_args_json,canonical_args_sha256,
          result_json,result_sha256,row_count,status
        FROM ingest.context_snapshots WHERE episode_id=@episode
        ORDER BY tool_id,canonical_args_sha256;
      `);
    const scenarioConfig = JSON.parse(execution.config_json) as { snapshotPlan?: unknown[] };
    const expectedSnapshotCount = scenarioConfig.snapshotPlan?.length ?? 0;
    pass(`snapshot_hashes:${execution.injector}`,
      snapshots.recordset.length === expectedSnapshotCount && snapshots.recordset.every((snapshot) =>
        snapshot.status === "complete" &&
        hashJson(JSON.parse(snapshot.canonical_args_json)) === snapshot.canonical_args_sha256 &&
        hashJson(JSON.parse(snapshot.result_json)) === snapshot.result_sha256),
      { expectedCount: expectedSnapshotCount, capturedCount: snapshots.recordset.length });

    if (execution.injector === "deadlock") {
      const graphSnapshot = snapshotResult(snapshots.recordset, "get_deadlock_graph");
      const graphRows = graphSnapshot.rows as Array<Record<string, unknown>>;
      const linkedGraphHashes = new Set(links.recordset
        .filter((row) => row.source_event_name === "xml_deadlock_report")
        .map((row) => row.raw_sha256));
      const matching = graphRows.find((row) => linkedGraphHashes.has(String(row.graph_sha256)));
      const processes = matching === undefined ? [] : JSON.parse(String(matching.processes_json)) as Array<Record<string, unknown>>;
      const resources = matching === undefined ? [] : JSON.parse(String(matching.resources_json)) as Array<Record<string, unknown>>;
      pass("current_deadlock_summary",
        matching !== undefined && Number(matching.process_count) === 2 && Number(matching.resource_count) >= 2 &&
          processes.length === 2 && processes.every((row) => /^[0-9a-f]{64}$/.test(String(row.statement_sha256))) &&
          resources.length >= 2 && resources.every((row) => /^[0-9a-f]{64}$/.test(String(row.resource_sha256))),
        { graphRows: graphRows.length, matchedCurrentGraph: matching !== undefined, processCount: processes.length, resourceCount: resources.length });
    }
    if (execution.injector === "blocking") {
      const blockingRows = snapshotResult(snapshots.recordset, "get_blocking_snapshot").rows as Array<Record<string, unknown>>;
      const transactionRows = snapshotResult(snapshots.recordset, "get_active_transactions").rows as Array<Record<string, unknown>>;
      pass("blocking_inflight_state",
        blockingRows.length >= 1 && transactionRows.length >= 1 && blockingRows.some((row) =>
          Number(row.wait_time_ms) >= 2_000 && manifest.sessionIds.includes(Number(row.session_id)) &&
          manifest.sessionIds.includes(Number(row.blocking_session_id))),
        { blockingRows: blockingRows.length, transactionRows: transactionRows.length });
    }
    if (execution.injector === "transaction_log_full") {
      const logRows = snapshotResult(snapshots.recordset, "get_log_space").rows as Array<Record<string, unknown>>;
      const transactionRows = snapshotResult(snapshots.recordset, "get_active_transactions").rows as Array<Record<string, unknown>>;
      pass("log_full_inflight_state",
        logRows.length === 1 && Number(logRows[0]!.active_log_percent) >= 90 &&
          logRows[0]!.log_truncation_holdup_reason === "ACTIVE_TRANSACTION" && transactionRows.length >= 1,
        { activeLogPercent: logRows[0]?.active_log_percent ?? null, holdupReason: logRows[0]?.log_truncation_holdup_reason ?? null, transactionRows: transactionRows.length });
      pass("disposable_database_cleanup",
        execution.database_name?.startsWith("LW_") === true && execution.cleanup_status === "dropped" &&
          execution.dropped_at_utc !== null && execution.live_database_id === null,
        { databaseNameSha256: execution.database_name === null ? null : hashJson(execution.database_name), cleanupStatus: execution.cleanup_status, liveDatabaseId: execution.live_database_id });
    }
  }
} finally {
  await pool.close();
}

const receiptBody = {
  schemaVersion: 1,
  runId: run.runId,
  scheduleName,
  injectionReceiptSha256: injectionReceipt.receiptSha256,
  verificationReceiptSha256: verificationReceipt.receiptSha256,
  standardManifestHash,
  checks,
  disposition: "PASS",
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/capture/injector-gate-${safeSchedule}.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({
  runId: run.runId,
  scheduleName,
  checks: checks.length,
  disposition: receipt.disposition,
  receiptSha256: receipt.receiptSha256,
}, null, 2));

function snapshotResult(rows: SnapshotRow[], tool: string): { rows: unknown[] } {
  const snapshot = rows.find((row) => row.tool_id === tool);
  if (snapshot === undefined) throw new Error(`Missing ${tool} snapshot`);
  const parsed = JSON.parse(snapshot.result_json) as { rows?: unknown[] };
  if (!Array.isArray(parsed.rows)) throw new Error(`${tool} snapshot result has no rows`);
  return { rows: parsed.rows };
}

function pass(name: string, condition: boolean, observed: Record<string, unknown>): void {
  if (!condition) throw new Error(`Injector gate failed: ${name}: ${JSON.stringify(observed)}`);
  checks.push({ name, disposition: "PASS", observed });
}

async function verifiedReceipt(path: string): Promise<Record<string, unknown> & { receiptSha256: string }> {
  const receipt = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown> & { receiptSha256: string };
  const { receiptSha256, ...body } = receipt;
  if (hashJson(body) !== receiptSha256) throw new Error(`Receipt hash mismatch: ${path}`);
  return receipt;
}

function safeName(value: string): string {
  if (!/^[a-z0-9-]{1,80}$/i.test(value)) throw new Error(`Unsafe schedule name: ${value}`);
  return value.toLowerCase();
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
