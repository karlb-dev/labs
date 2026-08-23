import { readFile, readdir, stat } from "node:fs/promises";
import { basename, relative, resolve } from "node:path";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashFile, hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";

interface TaxonomyRow {
  runId: string;
  taxonomyCode: string;
  stage: string;
  severity: string;
  evidence: Record<string, unknown>;
}

interface ClaimRow {
  claimId: string;
  runId: string;
  claimText: string;
  evidenceTag: string;
  status: string;
  supportQuery: string;
  supportArtifacts: string[];
  limitations: string[];
}

interface ManifestArtifact {
  path: string;
  kind: "query" | "evidence";
  rows: number | null;
}

const allowedEvidenceTags = new Set(["OBS", "PAIRED", "CAUSAL-APPLICATION", "SYSTEMS", "AUDIT", "HARNESS", "DERIVED"]);
const allowedClaimStatuses = new Set(["planned", "supported", "qualified", "rejected", "unavailable"]);
const allowedSeverities = new Set(["info", "warning", "error", "critical"]);
const dryRun = process.argv.includes("--dry-run");
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const taxonomy = await readJsonl<TaxonomyRow>(`${runDirectory}/tables/taxonomy.jsonl`);
const claims = await readJsonl<ClaimRow>(`${runDirectory}/tables/claims.jsonl`);
const rowManifest = JSON.parse(await readFile(`${runDirectory}/repro/rows/manifest.json`, "utf8")) as {
  runId: string;
  artifacts: ManifestArtifact[];
  queryBundleSha256: string;
  receiptSha256: string;
};
const expected = JSON.parse(await readFile(`${runDirectory}/repro/expected-report-outputs.json`, "utf8")) as {
  receiptSha256: string;
  outputs: Array<{ path: string; bytes: number; sha256: string }>;
};

validateInputs();
await validateExpectedOutputs();
const reports = (await readdir(`${runDirectory}/reports`, { withFileTypes: true }))
  .filter((entry) => entry.isFile() && entry.name.endsWith(".md"))
  .map((entry) => entry.name)
  .sort();
const reportSnapshots = await Promise.all(reports.map(async (name) => ({
  reportName: name,
  reportVersion: "tier1-v1",
  inputSetSha256: rowManifest.receiptSha256,
  contentSha256: await hashFile(`${runDirectory}/reports/${name}`),
  relativePath: `reports/${name}`,
})));
const claimRows = claims.map((claim) => ({ ...claim, claimSha256: hashJson(claim) }));
const rowCounts = Object.fromEntries(rowManifest.artifacts.filter((item) => item.kind === "query").map((item) => [item.path, item.rows]));
const snapshotRowCounts = canonicalJson({ inputQueries: rowCounts, taxonomyRows: taxonomy.length, claimRows: claims.length });

const config = loadConfig();
const pool = await connect(config.databases.lab, config.databases.controlName, 600_000, 4);
let insertedTaxonomy = 0;
let insertedClaims = 0;
let insertedSnapshots = 0;
try {
  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    const existingTaxonomy = await new sql.Request(transaction).input("run", sql.VarChar(120), run.runId).query<{
      taxonomy_code: string; stage: string; severity: string; evidence_json: string;
    }>(`SELECT taxonomy_code,stage,severity,evidence_json FROM eval.taxonomy_assignments WHERE run_id=@run;`);
    if (existingTaxonomy.recordset.length > 0) {
      const expectedRows = taxonomy.map(taxonomyIdentity).sort();
      const observedRows = existingTaxonomy.recordset.map((row) => canonicalJson({
        taxonomyCode: row.taxonomy_code, stage: row.stage, severity: row.severity, evidence: JSON.parse(row.evidence_json),
      })).sort();
      assertEqual("persisted taxonomy", observedRows, expectedRows);
    } else if (!dryRun) {
      for (const row of taxonomy) {
        await new sql.Request(transaction)
          .input("run", sql.VarChar(120), row.runId)
          .input("code", sql.VarChar(100), row.taxonomyCode)
          .input("stage", sql.VarChar(80), row.stage)
          .input("severity", sql.VarChar(24), row.severity)
          .input("evidence", sql.NVarChar(sql.MAX), canonicalJson(row.evidence))
          .query(`INSERT eval.taxonomy_assignments(run_id,taxonomy_code,stage,severity,evidence_json)
                  VALUES(@run,@code,@stage,@severity,@evidence);`);
        insertedTaxonomy += 1;
      }
    }

    const existingClaims = await new sql.Request(transaction).input("run", sql.VarChar(120), run.runId).query<{
      claim_id: string; claim_text: string; evidence_tag: string; status: string; support_query: string;
      support_artifacts_json: string; limitations_json: string; claim_sha256: string;
    }>(`SELECT claim_id,claim_text,evidence_tag,status,support_query,support_artifacts_json,limitations_json,claim_sha256
        FROM eval.claims WHERE run_id=@run;`);
    if (existingClaims.recordset.length > 0) {
      const expectedRows = claimRows.map(claimIdentity).sort();
      const observedRows = existingClaims.recordset.map((row) => canonicalJson({
        claimId: row.claim_id, claimText: row.claim_text, evidenceTag: row.evidence_tag, status: row.status,
        supportQuery: row.support_query, supportArtifacts: JSON.parse(row.support_artifacts_json),
        limitations: JSON.parse(row.limitations_json), claimSha256: row.claim_sha256,
      })).sort();
      assertEqual("persisted claims", observedRows, expectedRows);
    } else if (!dryRun) {
      for (const row of claimRows) {
        await new sql.Request(transaction)
          .input("id", sql.VarChar(100), row.claimId)
          .input("run", sql.VarChar(120), row.runId)
          .input("text", sql.NVarChar(2000), row.claimText)
          .input("tag", sql.VarChar(40), row.evidenceTag)
          .input("status", sql.VarChar(32), row.status)
          .input("query", sql.NVarChar(sql.MAX), row.supportQuery)
          .input("artifacts", sql.NVarChar(sql.MAX), canonicalJson(row.supportArtifacts))
          .input("limitations", sql.NVarChar(sql.MAX), canonicalJson(row.limitations))
          .input("hash", sql.Char(64), row.claimSha256)
          .query(`INSERT eval.claims(claim_id,run_id,claim_text,evidence_tag,status,support_query,support_artifacts_json,limitations_json,claim_sha256)
                  VALUES(@id,@run,@text,@tag,@status,@query,@artifacts,@limitations,@hash);`);
        insertedClaims += 1;
      }
    }

    const existingSnapshots = await new sql.Request(transaction).input("run", sql.VarChar(120), run.runId).query<{
      report_name: string; report_version: string; input_set_sha256: string; content_sha256: string; relative_path: string; row_counts_json: string;
    }>(`SELECT report_name,report_version,input_set_sha256,content_sha256,relative_path,row_counts_json
        FROM reporting.report_snapshots WHERE run_id=@run;`);
    if (existingSnapshots.recordset.length > 0) {
      const expectedRows = reportSnapshots.map((row) => snapshotIdentity(row, snapshotRowCounts)).sort();
      const observedRows = existingSnapshots.recordset.map((row) => canonicalJson({
        reportName: row.report_name, reportVersion: row.report_version, inputSetSha256: row.input_set_sha256,
        contentSha256: row.content_sha256, relativePath: row.relative_path, rowCounts: JSON.parse(row.row_counts_json),
      })).sort();
      assertEqual("persisted report snapshots", observedRows, expectedRows);
    } else if (!dryRun) {
      for (const row of reportSnapshots) {
        await new sql.Request(transaction)
          .input("run", sql.VarChar(120), run.runId)
          .input("name", sql.VarChar(160), row.reportName)
          .input("version", sql.VarChar(40), row.reportVersion)
          .input("input", sql.Char(64), row.inputSetSha256)
          .input("content", sql.Char(64), row.contentSha256)
          .input("path", sql.NVarChar(500), row.relativePath)
          .input("counts", sql.NVarChar(sql.MAX), snapshotRowCounts)
          .query(`INSERT reporting.report_snapshots(run_id,report_name,report_version,input_set_sha256,content_sha256,relative_path,row_counts_json)
                  VALUES(@run,@name,@version,@input,@content,@path,@counts);`);
        insertedSnapshots += 1;
      }
    }

    if (dryRun) await transaction.rollback();
    else await transaction.commit();
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
} finally {
  await pool.close();
}

const body = {
  schemaVersion: 1,
  runId: run.runId,
  mode: dryRun ? "dry-run" : "persist",
  inputManifestReceiptSha256: rowManifest.receiptSha256,
  queryBundleSha256: rowManifest.queryBundleSha256,
  expectedOutputsReceiptSha256: expected.receiptSha256,
  taxonomyRows: taxonomy.length,
  taxonomyBundleSha256: hashJson(taxonomy),
  claimRows: claims.length,
  claimBundleSha256: hashJson(claimRows),
  reportSnapshots: reportSnapshots.length,
  reportSnapshotBundleSha256: hashJson(reportSnapshots),
  inserted: { taxonomy: insertedTaxonomy, claims: insertedClaims, reportSnapshots: insertedSnapshots },
  disposition: "PASS",
};
const receipt = { ...body, receiptSha256: hashJson(body) };
if (!dryRun) {
  await atomicWrite(`${runDirectory}/metrics/results-finalization.json`, `${JSON.stringify(receipt, null, 2)}\n`);
  await appendExperimentLog(`Persisted and hash-locked ${taxonomy.length} taxonomy rows, ${claims.length} claims, and ${reportSnapshots.length} report snapshots; receipt ${receipt.receiptSha256}.`);
}
console.log(JSON.stringify(receipt, null, 2));

function validateInputs(): void {
  if (run.runId !== rowManifest.runId) throw new Error("Row-manifest run identity mismatch");
  if (taxonomy.length === 0 || claims.length === 0) throw new Error("Result tables are empty");
  for (const row of taxonomy) {
    if (row.runId !== run.runId || !/^[A-Z][A-Z0-9_]{1,99}$/.test(row.taxonomyCode) || !allowedSeverities.has(row.severity)) {
      throw new Error(`Invalid taxonomy row: ${canonicalJson(row)}`);
    }
  }
  const identifiers = new Set<string>();
  for (const row of claims) {
    if (row.runId !== run.runId || row.claimId.length > 100 || identifiers.has(row.claimId) ||
        !allowedEvidenceTags.has(row.evidenceTag) || !allowedClaimStatuses.has(row.status)) {
      throw new Error(`Invalid claim row: ${row.claimId}`);
    }
    identifiers.add(row.claimId);
  }
  const expectedBody = { ...expected } as Record<string, unknown>;
  delete expectedBody.receiptSha256;
  if (hashJson(expectedBody) !== expected.receiptSha256) throw new Error("Expected-output receipt hash drift");
}

async function validateExpectedOutputs(): Promise<void> {
  for (const output of expected.outputs) {
    const root = output.path.startsWith("repo/") ? resolve(runDirectory, "../..") : resolve(runDirectory);
    const path = output.path.startsWith("repo/") ? resolve(root, output.path.slice("repo/".length)) : resolve(root, output.path);
    if (!path.startsWith(`${root}/`) || relative(root, path).startsWith("..")) throw new Error(`Unsafe expected output: ${output.path}`);
    const information = await stat(path);
    if (information.size !== output.bytes || await hashFile(path) !== output.sha256) throw new Error(`Expected output drift: ${output.path}`);
  }
}

function taxonomyIdentity(row: TaxonomyRow): string {
  return canonicalJson({ taxonomyCode: row.taxonomyCode, stage: row.stage, severity: row.severity, evidence: row.evidence });
}

function claimIdentity(row: ClaimRow & { claimSha256: string }): string {
  return canonicalJson({
    claimId: row.claimId, claimText: row.claimText, evidenceTag: row.evidenceTag, status: row.status,
    supportQuery: row.supportQuery, supportArtifacts: row.supportArtifacts, limitations: row.limitations, claimSha256: row.claimSha256,
  });
}

function snapshotIdentity(row: typeof reportSnapshots[number], counts: string): string {
  return canonicalJson({ ...row, rowCounts: JSON.parse(counts) });
}

function assertEqual(label: string, observed: string[], wanted: string[]): void {
  if (canonicalJson(observed) !== canonicalJson(wanted)) throw new Error(`${label} differs from generated state`);
}

async function readJsonl<T>(path: string): Promise<T[]> {
  return (await readFile(path, "utf8")).split("\n").filter(Boolean).map((line) => JSON.parse(line) as T);
}
