import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { migrationFiles } from "../src/migrations.js";
import { connect } from "../src/repository.js";
import { LAB_ROOT, atomicWrite, resolveRunDirectory } from "../src/run.js";

interface RunManifest {
  runId: string;
  runManifestHash: string;
}

interface Probe {
  name: string;
  required: boolean;
  ok: boolean;
  value?: unknown;
  error?: { number?: number; message: string };
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const probes: Probe[] = [];

const controlAdmin = await connect(config.databases.admin, config.databases.controlName);
const workloadAdmin = await connect(config.databases.admin, config.databases.workloadName);
const masterLab = await connect(config.databases.lab, "master");
const controlAgent = await connect(config.databases.agent, config.databases.controlName);

try {
  const server = await requiredQuery("sql_server_identity", controlAdmin, `
    SELECT
      CONVERT(nvarchar(80), SERVERPROPERTY('ProductVersion')) AS product_version,
      CONVERT(nvarchar(160), SERVERPROPERTY('Edition')) AS edition,
      CONVERT(int, SERVERPROPERTY('ProductMajorVersion')) AS major_version,
      CONVERT(nvarchar(160), SERVERPROPERTY('ProductLevel')) AS product_level;
  `);

  const controlSettings = await requiredQuery("control_database_settings", controlAdmin, databaseSettingsQuery());
  const workloadSettings = await requiredQuery("workload_database_settings", workloadAdmin, databaseSettingsQuery());
  const databaseRows = [...controlSettings, ...workloadSettings].sort((a, b) => String(a.name).localeCompare(String(b.name)));
  record(
    "database_compatibility_170",
    true,
    databaseRows.length === 2 && databaseRows.every((row) => row.compatibility_level === 170),
    databaseRows,
  );
  record(
    "query_store_read_write",
    true,
    databaseRows.length === 2 && databaseRows.every((row) => row.query_store_state === "READ_WRITE"),
    databaseRows,
  );
  record(
    "query_store_frozen_settings",
    true,
    databaseRows.length === 2 && databaseRows.every((row) =>
      row.query_capture_mode_desc === "ALL" &&
      Number(row.interval_length_minutes) === 1 &&
      Number(row.flush_interval_seconds) === 60 &&
      Number(row.max_storage_size_mb) === 1024 &&
      row.wait_stats_capture_mode_desc === "ON"),
    databaseRows,
  );

  const exactVector = await tryQuery(controlAdmin, `
    SELECT CONVERT(float, VECTOR_DISTANCE('cosine',
      CAST('[1,0,0]' AS vector(3)), CAST('[1,0,0]' AS vector(3)))) AS distance;
  `);
  record("vector_exact", true, exactVector.ok && Number(exactVector.rows[0]?.distance) === 0, exactVector.rows, exactVector.error);

  const featureMetadata = await requiredQuery("feature_metadata", controlAdmin, `
    SELECT
      CONVERT(bit, CASE WHEN TYPE_ID(N'json') IS NOT NULL THEN 1 ELSE 0 END) AS json_type_supported,
      CONVERT(bit, CASE WHEN OBJECT_ID(N'sys.json_indexes') IS NOT NULL THEN 1 ELSE 0 END) AS json_index_supported,
      CONVERT(bit, CASE WHEN OBJECT_ID(N'sys.vector_indexes') IS NOT NULL THEN 1 ELSE 0 END) AS vector_index_metadata_supported,
      CONVERT(bit, CASE WHEN OBJECT_ID(N'sys.fn_ai_generate_chunks') IS NOT NULL OR OBJECT_ID(N'sys.ai_generate_chunks') IS NOT NULL THEN 1 ELSE 0 END) AS ai_chunks_metadata_supported,
      CONVERT(bit, CASE WHEN OBJECT_ID(N'sys.fn_ai_generate_embeddings') IS NOT NULL OR OBJECT_ID(N'sys.ai_generate_embeddings') IS NOT NULL THEN 1 ELSE 0 END) AS ai_embeddings_metadata_supported,
      CONVERT(bit, CASE WHEN EXISTS
        (SELECT 1 FROM sys.database_scoped_configurations WHERE name = N'PREVIEW_FEATURES' AND value = 1)
        THEN 1 ELSE 0 END) AS preview_features_enabled;
  `);
  const features = first(featureMetadata);
  record("json_native_type", false, features.json_type_supported === true, features.json_type_supported);
  record("json_native_index", false, features.json_index_supported === true, features.json_index_supported);
  record("vector_index_metadata", false, features.vector_index_metadata_supported === true, features.vector_index_metadata_supported);
  record("sql_ai_chunks", false, features.ai_chunks_metadata_supported === true, features.ai_chunks_metadata_supported);
  record("sql_ai_embeddings", false, features.ai_embeddings_metadata_supported === true, features.ai_embeddings_metadata_supported);

  const fulltext = await requiredQuery("fulltext_query", controlAdmin, `
    SELECT
      CONVERT(int, FULLTEXTSERVICEPROPERTY('IsFullTextInstalled')) AS installed,
      CONVERT(int, CASE WHEN EXISTS (SELECT 1 FROM sys.fulltext_catalogs WHERE name = N'logwarden_runbooks_fts') THEN 1 ELSE 0 END) AS catalog_present,
      CONVERT(int, CASE WHEN EXISTS (SELECT 1 FROM sys.fulltext_indexes WHERE object_id = OBJECT_ID(N'kb.runbook_chunks')) THEN 1 ELSE 0 END) AS index_present,
      CONVERT(int, COALESCE(FULLTEXTCATALOGPROPERTY(N'logwarden_runbooks_fts', 'PopulateStatus'), -1)) AS population_status;
  `);
  const fulltextRow = first(fulltext);
  record("fulltext", true, fulltextRow.installed === 1 && fulltextRow.catalog_present === 1 && fulltextRow.index_present === 1, fulltextRow);

  const xe = await tryQuery(masterLab, `
    SELECT s.name, s.startup_state, s.event_retention_mode_desc,
           s.max_dispatch_latency, s.track_causality,
           CONVERT(bit, CASE WHEN r.name IS NULL THEN 0 ELSE 1 END) AS running,
           (SELECT COUNT(*) FROM sys.server_event_session_events AS e WHERE e.event_session_id = s.event_session_id) AS event_count,
           (SELECT COUNT(*) FROM sys.server_event_session_targets AS t WHERE t.event_session_id = s.event_session_id AND t.name = N'event_file') AS file_target_count
    FROM sys.server_event_sessions AS s
    LEFT JOIN sys.dm_xe_sessions AS r ON r.name = s.name
    WHERE s.name = N'logwarden_capture';
  `);
  const xeRow = xe.rows[0] ?? {};
  record("xe_session", true,
    xe.ok && xe.rows.length === 1 && xeRow.running === true && xeRow.startup_state === true &&
      xeRow.max_dispatch_latency === 2000 && Number(xeRow.event_count) >= 5 && xeRow.file_target_count === 1,
    xe.rows, xe.error);

  const blockedThreshold = await requiredQuery("blocked_process_threshold_query", masterLab, `
    SELECT CONVERT(int, value_in_use) AS seconds
    FROM sys.configurations WHERE name = N'blocked process threshold (s)';
  `);
  record("blocked_process_threshold", true, first(blockedThreshold).seconds === 5, first(blockedThreshold));

  const labPermissions = await requiredQuery("lab_permissions", masterLab, `
    SELECT
      CONVERT(int, HAS_PERMS_BY_NAME(NULL, NULL, 'VIEW SERVER STATE')) AS view_server_state,
      CONVERT(int, HAS_PERMS_BY_NAME(NULL, NULL, 'VIEW SERVER PERFORMANCE STATE')) AS view_server_performance_state,
      CONVERT(int, HAS_PERMS_BY_NAME(NULL, NULL, 'ALTER ANY EVENT SESSION')) AS alter_any_event_session,
      CONVERT(int, IS_SRVROLEMEMBER(N'sysadmin')) AS sysadmin;
  `);
  const labPermissionRow = first(labPermissions);
  record("lab_server_permissions", true,
    labPermissionRow.view_server_state === 1 && labPermissionRow.view_server_performance_state === 1 &&
      labPermissionRow.alter_any_event_session === 1 && labPermissionRow.sysadmin === 0,
    labPermissionRow);

  const agentPositive = await tryQuery(controlAgent, `
    SELECT TOP (0) work_item_id FROM ops.work_items;
    SELECT
      CONVERT(int, HAS_PERMS_BY_NAME(N'ops.usp_claim_work_item', N'OBJECT', N'EXECUTE')) AS claim_execute,
      CONVERT(int, HAS_PERMS_BY_NAME(N'ops.usp_transition_work_item', N'OBJECT', N'EXECUTE')) AS transition_execute,
      CONVERT(int, HAS_PERMS_BY_NAME(N'kb.search_runbook_exact', N'OBJECT', N'EXECUTE')) AS search_execute,
      CONVERT(int, IS_SRVROLEMEMBER(N'sysadmin')) AS sysadmin;
  `);
  const agentPositiveRow = agentPositive.rows[0] ?? {};
  record("agent_positive_permissions", true,
    agentPositive.ok && agentPositiveRow.claim_execute === 1 && agentPositiveRow.transition_execute === 1 &&
      agentPositiveRow.search_execute === 1 && agentPositiveRow.sysadmin === 0,
    agentPositive.rows, agentPositive.error);

  await deniedProbe("agent_denied_ground_truth", controlAgent, "SELECT TOP (1) ground_truth_json FROM workload.scenario_definitions;");
  await deniedProbe("agent_denied_eval", controlAgent, "SELECT TOP (1) * FROM eval.ground_truth_episodes;");
  await deniedProbe("agent_denied_jobs", controlAgent, "SELECT TOP (1) * FROM control.jobs;");
  await deniedProbe("agent_denied_direct_kb_write", controlAgent,
    "UPDATE kb.runbook_chunks SET content = content WHERE 1 = 0;");
  await deniedProbe("agent_denied_ddl", controlAgent, "CREATE TABLE dbo.__lw_forbidden(id int); DROP TABLE dbo.__lw_forbidden;", [262]);

  const migrationRows = await controlAdmin.request().query<{ migration_id: string; migration_sha256: string }>(
    "SELECT migration_id, migration_sha256 FROM control.schema_migrations ORDER BY migration_id",
  );
  const migrationFilesOnDisk = await migrationFiles(`${LAB_ROOT}/db/migrations`);
  const diskMigrations = await Promise.all(migrationFilesOnDisk.map(async (path) => ({
    migration_id: path.split("/").at(-1)!.replace(/\.sql$/, ""),
    migration_sha256: sha256(await readFile(path, "utf8")),
  })));
  record("migration_integrity", true,
    canonicalJson(migrationRows.recordset) === canonicalJson(diskMigrations),
    { database: migrationRows.recordset, disk: diskMigrations });

  const schemaCounts = await requiredQuery("schema_counts", controlAdmin, `
    SELECT s.name AS schema_name, COUNT(*) AS table_count
    FROM sys.tables AS t INNER JOIN sys.schemas AS s ON s.schema_id = t.schema_id
    WHERE s.name IN ('control','workload','ingest','ops','kb','agent','telemetry','eval','reporting')
    GROUP BY s.name ORDER BY s.name;
  `);
  record("tier1_schema_present", true, asRecords(schemaCounts).length === 9, asRecords(schemaCounts));

  const snapshotBody = {
    schemaVersion: 1,
    runId: run.runId,
    runManifestHash: run.runManifestHash,
    server: first(server),
    databases: databaseRows,
    features,
    probes: [...probes].sort((a, b) => a.name.localeCompare(b.name)),
  };
  const snapshotSha256 = hashJson(snapshotBody);
  const requiredFailures = probes.filter((probe) => probe.required && !probe.ok);

  await controlAdmin.request()
    .input("run_id", sql.VarChar(120), run.runId)
    .input("version", sql.VarChar(80), String(first(server).product_version))
    .input("edition", sql.NVarChar(160), String(first(server).edition))
    .input("compatibility", sql.Int, Number(databaseRows.find((row) => row.name === config.databases.controlName)?.compatibility_level ?? 0))
    .input("preview", sql.Bit, features.preview_features_enabled === true)
    .input("vector", sql.Bit, probeOk("vector_exact"))
    .input("vector_mode", sql.VarChar(32), probeOk("vector_index_metadata") ? "ann_available" : "exact_only")
    .input("json_type", sql.Bit, probeOk("json_native_type"))
    .input("json_index", sql.Bit, probeOk("json_native_index"))
    .input("ai_chunks", sql.Bit, probeOk("sql_ai_chunks"))
    .input("ai_embeddings", sql.Bit, probeOk("sql_ai_embeddings"))
    .input("fulltext", sql.Bit, probeOk("fulltext"))
    .input("query_store", sql.VarChar(32), probeOk("query_store_read_write") ? "READ_WRITE" : "UNAVAILABLE")
    .input("xe", sql.Bit, probeOk("xe_session"))
    .input("json", sql.NVarChar(sql.MAX), canonicalJson(snapshotBody))
    .input("hash", sql.Char(64), snapshotSha256)
    .query(`
      IF NOT EXISTS (SELECT 1 FROM control.capability_snapshots WHERE snapshot_sha256 = @hash)
        INSERT control.capability_snapshots
        (
          run_id, sql_product_version, sql_edition, database_compatibility_level,
          preview_features_enabled, vector_supported, vector_index_mode,
          json_type_supported, json_index_supported, ai_chunks_supported,
          ai_embeddings_supported, fulltext_supported, query_store_state,
          xe_permissions_ok, snapshot_json, snapshot_sha256
        )
        VALUES
        (
          @run_id, @version, @edition, @compatibility, @preview, @vector, @vector_mode,
          @json_type, @json_index, @ai_chunks, @ai_embeddings, @fulltext,
          @query_store, @xe, @json, @hash
        );
    `);

  const output = {
    ...snapshotBody,
    snapshotSha256,
    disposition: requiredFailures.length === 0 ? "PASS" : "STOP_CAPABILITY",
    requiredFailures: requiredFailures.map((probe) => probe.name),
  };
  await atomicWrite(`${runDirectory}/environment/capabilities.json`, `${JSON.stringify(output, null, 2)}\n`);
  await atomicWrite(`${runDirectory}/environment/sql-settings.json`, `${JSON.stringify({
    schemaVersion: 1,
    runId: run.runId,
    server: first(server),
    databases: databaseRows,
    blockedProcessThresholdSeconds: first(blockedThreshold).seconds,
    xe: xe.rows,
    snapshotSha256,
  }, null, 2)}\n`);

  const evidence = {
    capabilitySnapshotSha256: snapshotSha256,
    disposition: output.disposition,
    requiredFailures: output.requiredFailures,
  };
  await controlAdmin.request()
    .input("key", sql.VarChar(120), `lw-foundation-v1:${run.runId}:${snapshotSha256.slice(0, 16)}`)
    .input("run_id", sql.VarChar(120), run.runId)
    .input("json", sql.NVarChar(sql.MAX), canonicalJson(evidence))
    .input("hash", sql.Char(64), hashJson(evidence))
    .query(`
      IF NOT EXISTS (SELECT 1 FROM control.evidence_events WHERE event_key = @key)
        INSERT control.evidence_events(event_key, run_id, stage, scientific_tier, disposition, detail_json, detail_sha256)
        VALUES(@key, @run_id, 'LW-0', 'foundation', '${output.disposition}', @json, @hash);
    `);

  console.log(JSON.stringify({
    runId: run.runId,
    disposition: output.disposition,
    snapshotSha256,
    passedRequired: probes.filter((probe) => probe.required && probe.ok).map((probe) => probe.name),
    failedRequired: output.requiredFailures,
    unavailableOptional: probes.filter((probe) => !probe.required && !probe.ok).map((probe) => probe.name),
  }, null, 2));
  if (requiredFailures.length > 0) process.exitCode = 2;
} finally {
  await Promise.allSettled([
    controlAdmin.close(),
    workloadAdmin.close(),
    masterLab.close(),
    controlAgent.close(),
  ]);
}

function record(name: string, required: boolean, ok: boolean, value?: unknown, error?: Probe["error"]): void {
  probes.push({ name, required, ok, ...(value === undefined ? {} : { value }), ...(error === undefined ? {} : { error }) });
}

function probeOk(name: string): boolean {
  return probes.find((probe) => probe.name === name)?.ok === true;
}

async function requiredQuery(name: string, pool: sql.ConnectionPool, query: string): Promise<Record<string, unknown>[]> {
  const result = await tryQuery(pool, query);
  record(name, true, result.ok, result.rows, result.error);
  return result.rows;
}

async function deniedProbe(name: string, pool: sql.ConnectionPool, query: string, acceptedNumbers: number[] = [229]): Promise<void> {
  const result = await tryQuery(pool, query);
  record(name, true, !result.ok && result.error?.number !== undefined && acceptedNumbers.includes(result.error.number),
    result.ok ? { unexpectedlyAllowed: true } : result.error);
}

async function tryQuery(pool: sql.ConnectionPool, query: string): Promise<{
  ok: boolean;
  rows: Record<string, unknown>[];
  error?: Probe["error"];
}> {
  try {
    const result = await pool.request().batch(query);
    const recordsets = result.recordsets as unknown as Array<Record<string, unknown>[]>;
    return { ok: true, rows: recordsets[recordsets.length - 1] ?? [] };
  } catch (error) {
    const requestError = error as { number?: number; message?: string };
    return {
      ok: false,
      rows: [],
      error: {
        ...(requestError.number === undefined ? {} : { number: requestError.number }),
        message: requestError.message ?? String(error),
      },
    };
  }
}

function first(rows: Record<string, unknown>[]): Record<string, unknown> {
  return rows[0] ?? {};
}

function asRecords(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  return rows;
}

function databaseSettingsQuery(): string {
  return `
    SELECT DB_NAME() AS name, d.compatibility_level, d.is_read_committed_snapshot_on,
           q.actual_state_desc AS query_store_state,
           q.desired_state_desc AS query_store_desired_state,
           q.query_capture_mode_desc,
           q.interval_length_minutes,
           q.flush_interval_seconds,
           q.max_storage_size_mb,
           q.wait_stats_capture_mode_desc
    FROM sys.databases AS d
    CROSS JOIN sys.database_query_store_options AS q
    WHERE d.database_id = DB_ID();
  `;
}
