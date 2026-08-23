import { basename } from "node:path";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashFile } from "../src/hash.js";
import { resolveRunDirectory } from "../src/run.js";

type MigrationRow = { migration_id: string; migration_sha256: string };
type ReconciliationRow = { migration_id: string; recorded_sha256: string; source_sha256: string; rationale: string };

const runDirectory = resolveRunDirectory();
const runId = basename(runDirectory);
const gitHead = process.env.MODELPRINT_GIT_HEAD ?? (() => {
  try { return execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(); }
  catch { return null; }
})();
const migrationsDirectory = fileURLToPath(new URL("../db/migrations/", import.meta.url));
const migrationNames = (await readdir(migrationsDirectory)).filter((name) => name.endsWith(".sql")).sort();
const sourceMigrations = new Map<string, string>();
for (const name of migrationNames) sourceMigrations.set(name, await hashFile(`${migrationsDirectory}/${name}`));

const primary = JSON.parse(await readFile(`${runDirectory}/manifests/campaign-freeze.json`, "utf8")) as { campaignId: string | number; expectedJobs: number };
const robustness = JSON.parse(await readFile(`${runDirectory}/manifests/robustness-freeze.json`, "utf8")) as { campaignId: string | number; expectedJobs: number };
const expectedCampaigns = [primary, robustness].map((value) => ({ campaignId: Number(value.campaignId), expectedJobs: Number(value.expectedJobs) }));

const config = loadConfig().database;
const pool = await new sql.ConnectionPool({ server: config.server, port: config.port, user: config.user, password: config.password,
  database: config.database, options: { encrypt: false, trustServerCertificate: true }, requestTimeout: 600_000 }).connect();

const failures: string[] = [];
try {
  const server = await pool.request().query(`SELECT DB_NAME() database_name,CONVERT(varchar(64),SERVERPROPERTY('ProductVersion')) product_version,
    CONVERT(varchar(128),SERVERPROPERTY('Edition')) edition,d.compatibility_level,
    (SELECT value FROM sys.database_scoped_configurations WHERE name='PREVIEW_FEATURES') preview_features
    FROM sys.databases d WHERE d.name=DB_NAME()`);
  const serverState = server.recordset[0] as { product_version: string; compatibility_level: number; preview_features: boolean };
  const serverMajor = Number(serverState.product_version.split(".")[0]);
  if (!Number.isInteger(serverMajor) || serverMajor < 17) failures.push(`SQL Server 2025 or newer is required; found ${serverState.product_version}`);
  if (Number(serverState.compatibility_level) !== 170) failures.push(`database compatibility level must be 170; found ${serverState.compatibility_level}`);
  if (serverState.preview_features !== true) failures.push("database PREVIEW_FEATURES must be enabled for the governed ANN continuation");
  const recorded = await pool.request().query<MigrationRow>("SELECT migration_id,migration_sha256 FROM dbo.schema_migrations ORDER BY migration_id");
  const hasReconciliations = await pool.request().query<{ present: number }>("SELECT CASE WHEN OBJECT_ID(N'dbo.schema_migration_reconciliations',N'U') IS NULL THEN 0 ELSE 1 END present");
  const reconciliations = hasReconciliations.recordset[0]?.present
    ? (await pool.request().query<ReconciliationRow>("SELECT migration_id,recorded_sha256,source_sha256,rationale FROM dbo.schema_migration_reconciliations")).recordset
    : [];
  const recordedMap = new Map(recorded.recordset.map((row) => [row.migration_id, row.migration_sha256]));
  const migrationAudit = migrationNames.map((name) => {
    const sourceSha256 = sourceMigrations.get(name)!;
    const recordedSha256 = recordedMap.get(name);
    const reconciliation = reconciliations.find((row) => row.migration_id === name && row.recorded_sha256 === recordedSha256 && row.source_sha256 === sourceSha256);
    const disposition = recordedSha256 === sourceSha256 ? "EXACT" : reconciliation ? "EXPLICITLY_RECONCILED" : recordedSha256 ? "UNRECONCILED_DRIFT" : "MISSING";
    if (disposition === "UNRECONCILED_DRIFT" || disposition === "MISSING") failures.push(`migration ${name}: ${disposition}`);
    return { migrationId: name, sourceSha256, recordedSha256: recordedSha256 ?? null, disposition, rationale: reconciliation?.rationale ?? null };
  });
  for (const name of recordedMap.keys()) if (!sourceMigrations.has(name)) failures.push(`database records migration absent from source: ${name}`);

  const campaigns = [];
  for (const expected of expectedCampaigns) {
    const result = await pool.request().input("campaign", sql.BigInt, expected.campaignId).query(`SELECT
      COUNT_BIG(*) jobs,
      SUM(CASE WHEN status='complete' THEN CONVERT(bigint,1) ELSE CONVERT(bigint,0) END) complete_jobs,
      SUM(CASE WHEN status='failed' THEN CONVERT(bigint,1) ELSE CONVERT(bigint,0) END) failed_jobs,
      (SELECT COUNT_BIG(*) FROM dbo.generations WHERE campaign_id=@campaign) generations
      FROM dbo.generation_jobs WHERE campaign_id=@campaign`);
    const row = result.recordset[0] as Record<string, string | number>;
    const summary = { campaignId: expected.campaignId, expectedJobs: expected.expectedJobs, jobs: Number(row.jobs),
      completeJobs: Number(row.complete_jobs), failedJobs: Number(row.failed_jobs), generations: Number(row.generations) };
    if (summary.jobs !== expected.expectedJobs || summary.completeJobs !== expected.expectedJobs || summary.generations !== expected.expectedJobs || summary.failedJobs !== 0)
      failures.push(`campaign ${expected.campaignId} is incomplete or inconsistent`);
    campaigns.push(summary);
  }

  const state = await pool.request().input("runId", sql.VarChar(120), runId).query(`SELECT
    (SELECT COUNT_BIG(*) FROM dbo.metric_results WHERE run_id=@runId) metric_results,
    (SELECT COUNT_BIG(*) FROM dbo.search_runs WHERE run_id=@runId) search_runs,
    (SELECT COUNT_BIG(*) FROM dbo.search_runs WHERE run_id=@runId AND finished_at IS NULL) active_search_runs,
    (SELECT COUNT_BIG(*) FROM dbo.prediction_runs WHERE run_id=@runId) prediction_runs,
    (SELECT COUNT_BIG(*) FROM dbo.attribution_models WHERE run_id=@runId) attribution_models,
    (SELECT COUNT_BIG(*) FROM dbo.ann_benchmark_runs WHERE run_id=@runId) ann_benchmark_runs,
    (SELECT COUNT_BIG(*) FROM dbo.likelihood_scores) likelihood_scores,
    (SELECT COUNT_BIG(*) FROM dbo.semantic_vectors) semantic_vectors,
    (SELECT COUNT_BIG(*) FROM dbo.style_vectors) style_vectors`);
  const stateRow = state.recordset[0] as Record<string, string | number>;
  const runState = Object.fromEntries(Object.entries(stateRow).map(([key, value]) => [key, Number(value)]));
  if (Number(runState.active_search_runs) !== 0) failures.push(`${runState.active_search_runs} search run(s) are still active`);

  const result = { schemaVersion: 1, checkedAt: new Date().toISOString(), status: failures.length ? "FAIL" : "PASS", runId,
    gitHead, server: server.recordset[0], migrations: migrationAudit, campaigns, runState, failures };
  await writeFile(`${runDirectory}/environment/database-handoff-validation.json`, `${JSON.stringify(result, null, 2)}\n`);
  console.log(JSON.stringify(result, null, 2));
  if (failures.length) process.exitCode = 2;
} finally {
  await pool.close();
}
