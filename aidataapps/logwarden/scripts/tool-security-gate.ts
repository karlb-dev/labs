import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { canonicalToolArguments, canonicalToolArgumentsSha256, loadToolRegistry, toolRegistrySha256 } from "../src/tools.js";

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const gateId = randomUUID().replaceAll("-", "");
const corpusId = `gate-tools-${gateId.slice(0, 24)}`;
const runbookId = `rb-${gateId.slice(0, 24)}`;
const chunkId = `chunk-${gateId.slice(0, 24)}`;
const episodeId = `episode-tools-${gateId.slice(0, 24)}`;
const registry = loadToolRegistry();
const registrySha256 = toolRegistrySha256(registry);
const snapshotArgs = canonicalToolArguments("get_log_space", { databaseName: "LogWardenWorkload" });
const snapshotArgsSha256 = canonicalToolArgumentsSha256("get_log_space", { databaseName: "LogWardenWorkload" });
const snapshotResult = { status: "ok", databaseName: "logwardenworkload", activeLogPercent: 2.5, logTruncationHoldupReason: "NOTHING" };

const lab = await connect(config.databases.lab, config.databases.controlName);
try {
  await lab.request()
    .input("corpus", sql.VarChar(80), corpusId)
    .input("manifest", sql.NVarChar(sql.MAX), canonicalJson({ gateId, source: "synthetic tool gate" }))
    .input("manifest_hash", sql.Char(64), hashJson({ gateId, source: "synthetic tool gate" }))
    .input("runbook", sql.VarChar(100), runbookId)
    .input("source_hash", sql.Char(64), hashJson({ gateId, kind: "source" }))
    .input("body_hash", sql.Char(64), hashJson({ gateId, kind: "body" }))
    .input("chunk", sql.VarChar(120), chunkId)
    .input("content_hash", sql.Char(64), hashJson({ gateId, kind: "chunk" }))
    .input("metadata", sql.NVarChar(sql.MAX), canonicalJson({ gate: true }))
    .query(`
      INSERT kb.search_corpora(corpus_id,corpus_version,corpus_kind,source_manifest_json,source_manifest_sha256)
      VALUES(@corpus,'gate-v1','primary',@manifest,@manifest_hash);
      INSERT kb.runbooks(runbook_id,corpus_id,title,incident_class,severity_floor,source_kind,
        source_sha256,body_markdown,body_sha256,metadata_json)
      VALUES(@runbook,@corpus,N'Bounded transaction log diagnostics','transaction_log_full','medium',
        'synthetic_gate',@source_hash,N'Inspect log space and backup history before escalation.',@body_hash,@metadata);
      INSERT kb.runbook_chunks(chunk_id,runbook_id,corpus_id,ordinal,heading_path,content,
        token_count,chunker_version,content_sha256,metadata_json)
      VALUES(@chunk,@runbook,@corpus,0,N'Log space',
        N'A bounded transaction log incident should inspect log space and backup history before escalation.',
        14,'gate-v1',@content_hash,@metadata);
    `);
  await lab.request()
    .input("episode", sql.VarChar(120), episodeId)
    .input("args", sql.NVarChar(sql.MAX), canonicalJson(snapshotArgs))
    .input("args_hash", sql.Char(64), snapshotArgsSha256)
    .input("result", sql.NVarChar(sql.MAX), canonicalJson(snapshotResult))
    .input("result_hash", sql.Char(64), hashJson(snapshotResult))
    .query(`
      INSERT ingest.context_snapshots
        (episode_id,snapshot_kind,tool_id,canonical_args_sha256,canonical_args_json,
         procedure_version,requested_at_utc,captured_at_utc,result_json,result_sha256,
         row_count,latency_ms,status)
      VALUES(@episode,'log_space','get_log_space',@args_hash,@args,'gate-v1',
        SYSUTCDATETIME(),SYSUTCDATETIME(),@result,@result_hash,1,0.1,'complete');
    `);
} finally {
  await lab.close();
}

const positives: Array<Record<string, unknown>> = [];
const negatives: Array<Record<string, unknown>> = [];
const agent = await connect(config.databases.agent, config.databases.controlName, 30_000);
try {
  await waitForFulltext(agent);
  positives.push(await positive("runbook_search", () => agent.request()
    .input("query", sql.NVarChar(1000), "transaction log backup history")
    .input("top_k", sql.Int, 5).input("corpus_id", sql.VarChar(80), corpusId)
    .execute("kb.usp_search_runbooks"), 1));
  positives.push(await positive("get_recent_incident_counts", () => agent.request()
    .input("incident_class", sql.VarChar(80), null).input("window_minutes", sql.Int, 60)
    .execute("ops.usp_get_recent_incident_counts"), 1));
  positives.push(await positive("get_blocking_snapshot", () => agent.request()
    .input("database_name", sql.NVarChar(128), "LogWardenWorkload").input("max_rows", sql.Int, 5)
    .execute("agent.usp_tool_get_blocking_snapshot")));
  positives.push(await positive("get_log_space", () => agent.request()
    .input("database_name", sql.NVarChar(128), "LogWardenWorkload")
    .execute("agent.usp_tool_get_log_space"), 1));
  positives.push(await positive("get_active_transactions", () => agent.request()
    .input("database_name", sql.NVarChar(128), "LogWardenWorkload").input("max_rows", sql.Int, 5)
    .execute("agent.usp_tool_get_active_transactions")));
  positives.push(await positive("get_backup_history", () => agent.request()
    .input("database_name", sql.NVarChar(128), "LogWardenWorkload").input("max_rows", sql.Int, 5)
    .execute("agent.usp_tool_get_backup_history"), 1));
  positives.push(await positive("get_deadlock_graph", () => agent.request()
    .input("max_rows", sql.Int, 5).execute("agent.usp_tool_get_deadlock_graph")));
  positives.push(await positive("snapshot_hit", () => agent.request()
    .input("episode_id", sql.VarChar(120), episodeId)
    .input("tool_id", sql.VarChar(80), "get_log_space")
    .input("canonical_args_sha256", sql.Char(64), snapshotArgsSha256)
    .execute("agent.usp_tool_resolve_context_snapshot"), 1));
  const miss = await agent.request()
    .input("episode_id", sql.VarChar(120), episodeId)
    .input("tool_id", sql.VarChar(80), "get_log_space")
    .input("canonical_args_sha256", sql.Char(64), "0".repeat(64))
    .execute("agent.usp_tool_resolve_context_snapshot");
  if (miss.recordset[0]?.snapshot_miss !== true || miss.recordset[0]?.status !== "no_data") throw new Error("Snapshot miss was not deterministic");
  positives.push(summary("snapshot_miss", miss.recordset));

  for (const [name, operation] of [
    ["protected_truth", () => agent.request().query("SELECT TOP (1) * FROM eval.ground_truth_episodes")],
    ["protected_snapshot", () => agent.request().query("SELECT TOP (1) * FROM ingest.context_snapshots")],
    ["runbook_table_bypass", () => agent.request().query("SELECT TOP (1) * FROM kb.runbook_chunks")],
    ["control_job_bypass", () => agent.request().query("SELECT TOP (1) * FROM control.jobs")],
    ["msdb_backup_bypass", () => agent.request().query("SELECT TOP (1) * FROM msdb.dbo.backupset")],
    ["msdb_proxy_bypass", () => agent.request()
      .input("database_name",sql.NVarChar(128),"LogWardenWorkload").input("max_rows",sql.Int,5)
      .execute("msdb.dbo.usp_logwarden_backup_history")],
    ["arbitrary_ingest_execute", () => agent.request().input("run_id",sql.VarChar(120),run.runId).input("worker_id",sql.VarChar(120),"forbidden").execute("ingest.usp_ingest_xe")],
    ["ddl_create_table", () => agent.request().query("CREATE TABLE dbo.forbidden_tool_gate(id int)")],
    ["database_outside_allowlist", () => agent.request().input("database_name",sql.NVarChar(128),"master").execute("agent.usp_tool_get_log_space")],
    ["database_argument_injection", () => agent.request().input("database_name",sql.NVarChar(128),"LogWardenWorkload]; DROP DATABASE x;--").execute("agent.usp_tool_get_log_space")],
  ] as Array<[string, () => Promise<unknown>]>) negatives.push(await denied(name, operation));

  const directDmv = await agent.request().query<{ view_server_state: number; view_server_performance_state: number; visible_requests: number }>(`
    SELECT
      HAS_PERMS_BY_NAME(NULL,NULL,'VIEW SERVER STATE') AS view_server_state,
      HAS_PERMS_BY_NAME(NULL,NULL,'VIEW SERVER PERFORMANCE STATE') AS view_server_performance_state,
      (SELECT COUNT(*) FROM sys.dm_exec_requests) AS visible_requests;
  `);
  const direct = directDmv.recordset[0]!;
  if (direct.view_server_state !== 0 || direct.view_server_performance_state !== 0 || Number(direct.visible_requests) > 1) {
    throw new Error(`Certificate permissions leaked to the direct agent token: ${JSON.stringify(direct)}`);
  }
  negatives.push({ name: "server_dmv_permission_not_leaked", disposition: "PASS", observed: direct });
} finally {
  await agent.close();
}

const administrative = await administrativeEvidence();
const receiptBody = {
  schemaVersion: 1,
  runId: run.runId,
  gateId,
  registryVersion: registry.registryVersion,
  registrySha256,
  registryFrozen: registry.frozen,
  toolCount: Object.keys(registry.tools).length,
  positives,
  negatives,
  canonicalSnapshot: { episodeId, args: snapshotArgs, argsSha256: snapshotArgsSha256 },
  administrative,
  disposition: "PASS",
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/security/tool-security-gate.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({ runId: run.runId, toolCount: receiptBody.toolCount, positiveCases: positives.length, negativeCases: negatives.length, registrySha256, disposition: "PASS", receiptSha256: receipt.receiptSha256 }, null, 2));

async function positive(name: string, operation: () => Promise<{ recordset: unknown[] }>, minimumRows = 0): Promise<Record<string, unknown>> {
  const result = await operation();
  if (result.recordset.length < minimumRows) throw new Error(`${name} returned ${result.recordset.length}; expected at least ${minimumRows}`);
  return summary(name, result.recordset);
}

function summary(name: string, rows: unknown[]): Record<string, unknown> {
  const normalized = JSON.parse(JSON.stringify(rows)) as unknown[];
  return { name, rowCount: normalized.length, resultSha256: hashJson(normalized), disposition: "PASS" };
}

async function denied(name: string, operation: () => Promise<unknown>): Promise<Record<string, unknown>> {
  try {
    await operation();
  } catch (error) {
    const value = error as { number?: number; code?: string };
    return { name, disposition: "PASS", errorNumber: value.number ?? null, errorCode: value.code ?? null };
  }
  throw new Error(`Negative tool-security case was unexpectedly allowed: ${name}`);
}

async function waitForFulltext(pool: sql.ConnectionPool): Promise<void> {
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    try {
      const result = await pool.request()
        .input("query", sql.NVarChar(1000), "transaction log backup history")
        .input("top_k", sql.Int, 5).input("corpus_id", sql.VarChar(80), corpusId)
        .execute("kb.usp_search_runbooks");
      if (result.recordset.some((row) => row.chunk_id === chunkId)) return;
    } catch {
      // Full-text change tracking can briefly report the row as unavailable.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("Full-text tool gate row did not become searchable");
}

async function administrativeEvidence(): Promise<Record<string, unknown>> {
  const control = await connect(config.databases.admin, config.databases.controlName);
  const master = await connect(config.databases.admin, "master");
  const msdb = await connect(config.databases.admin, "msdb");
  try {
    const modules = await control.request().query<{ module_count: number; signed_server_modules: number; unsafe_parameter_count: number }>(`
      SELECT
        (SELECT COUNT(*) FROM sys.procedures WHERE object_id IN
          (OBJECT_ID('ops.usp_get_recent_incident_counts'),OBJECT_ID('kb.usp_search_runbooks'),
           OBJECT_ID('agent.usp_tool_get_blocking_snapshot'),OBJECT_ID('agent.usp_tool_get_log_space'),
           OBJECT_ID('agent.usp_tool_get_active_transactions'),OBJECT_ID('agent.usp_tool_get_backup_history'),
           OBJECT_ID('agent.usp_tool_get_deadlock_graph'),OBJECT_ID('agent.usp_tool_resolve_context_snapshot'))) AS module_count,
        (SELECT COUNT(DISTINCT major_id) FROM sys.crypt_properties
         WHERE major_id IN (OBJECT_ID('agent.usp_tool_get_blocking_snapshot'),OBJECT_ID('agent.usp_tool_get_log_space'),
           OBJECT_ID('agent.usp_tool_get_active_transactions'),OBJECT_ID('agent.usp_tool_get_backup_history'),
           OBJECT_ID('agent.usp_tool_get_deadlock_graph'))
           AND thumbprint=(SELECT thumbprint FROM sys.certificates WHERE name='LogWardenToolCertificate')) AS signed_server_modules,
        (SELECT COUNT(*) FROM sys.parameters
         WHERE object_id IN (OBJECT_ID('ops.usp_get_recent_incident_counts'),OBJECT_ID('kb.usp_search_runbooks'),
           OBJECT_ID('agent.usp_tool_get_blocking_snapshot'),OBJECT_ID('agent.usp_tool_get_log_space'),
           OBJECT_ID('agent.usp_tool_get_active_transactions'),OBJECT_ID('agent.usp_tool_get_backup_history'),
           OBJECT_ID('agent.usp_tool_get_deadlock_graph'),OBJECT_ID('agent.usp_tool_resolve_context_snapshot'))
           AND (name IN ('@sql','@statement','@command_text') OR max_length=-1)) AS unsafe_parameter_count;
    `);
    const server = await master.request().query<{ permission_name: string }>(`
      SELECT permission_name FROM sys.server_permissions AS permission
      INNER JOIN sys.server_principals AS principal ON principal.principal_id=permission.grantee_principal_id
      WHERE principal.name='LogWardenToolCertificateLogin' ORDER BY permission_name;
    `);
    const msdbEvidence = await msdb.request().query<{ certificate_present: number; user_present: number; select_grants: number; execute_grants: number; agent_execute_grants: number }>(`
      SELECT
        CONVERT(int,CASE WHEN CERT_ID('LogWardenToolCertificate') IS NULL THEN 0 ELSE 1 END) AS certificate_present,
        CONVERT(int,CASE WHEN USER_ID('LogWardenToolCertificateUser') IS NULL THEN 0 ELSE 1 END) AS user_present,
        (SELECT COUNT(*) FROM sys.database_permissions AS p INNER JOIN sys.database_principals AS u ON u.principal_id=p.grantee_principal_id WHERE u.name='LogWardenToolCertificateUser' AND p.permission_name='SELECT' AND p.state='G') AS select_grants,
        (SELECT COUNT(*) FROM sys.database_permissions AS p INNER JOIN sys.database_principals AS u ON u.principal_id=p.grantee_principal_id WHERE u.name='LogWardenToolCertificateUser' AND p.permission_name='EXECUTE' AND p.state='G' AND p.major_id=OBJECT_ID('dbo.usp_logwarden_backup_history')) AS execute_grants,
        (SELECT COUNT(*) FROM sys.database_permissions AS p INNER JOIN sys.database_principals AS u ON u.principal_id=p.grantee_principal_id WHERE u.name='lw_agent' AND p.permission_name='EXECUTE' AND p.state='G' AND p.major_id=OBJECT_ID('dbo.usp_logwarden_backup_history')) AS agent_execute_grants;
    `);
    const moduleRow = modules.recordset[0]!;
    const permissionNames = server.recordset.map((row) => row.permission_name);
    const msdbRow = msdbEvidence.recordset[0]!;
    if (Number(moduleRow.module_count) !== 8 || Number(moduleRow.signed_server_modules) !== 5 || Number(moduleRow.unsafe_parameter_count) !== 0 ||
        !["VIEW ANY DATABASE","VIEW SERVER PERFORMANCE STATE","VIEW SERVER STATE"].every((permission) => permissionNames.includes(permission)) ||
        msdbRow.certificate_present !== 1 || msdbRow.user_present !== 1 || Number(msdbRow.select_grants) !== 0 ||
        Number(msdbRow.execute_grants) !== 1 || Number(msdbRow.agent_execute_grants) !== 0) {
      throw new Error(`Administrative tool evidence failed: ${JSON.stringify({ moduleRow, permissionNames, msdbRow })}`);
    }
    return { modules: moduleRow, certificateServerPermissions: permissionNames, msdb: msdbRow };
  } finally {
    await Promise.all([control.close(), master.close(), msdb.close()]);
  }
}
