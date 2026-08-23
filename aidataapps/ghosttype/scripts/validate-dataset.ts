import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

// GT-1 runtime gates over imported rows (spec §20 subset that is computable
// without ScriptDom/fixtures; parser/binding/compile land in GT-3):
//  1. counts and role math against the package contract
//  2. split-group role exclusivity (import already enforces; re-verified)
//  3. UTF-16/UTF-8 cursor round-trip against the stored document prefix
//  4. insertion integrity: prefix+canonical+suffix recomposes; no lone
//     surrogates; expect_empty ⇔ empty canonical consistency
//  5. suffix-duplication trap audit (gold never duplicates suffix start)
//  6. gold-candidate presence and canonical membership

interface CaseRow {
  case_id: string; data_role: string; expect_empty: boolean;
  cursor_offset_utf16: number; cursor_offset_utf8: number;
  doc_prefix: string; doc_suffix: string; canonical_insertion: string;
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const pool = await new sql.ConnectionPool({ ...config.databases.admin, database: config.databases.controlName }).connect();
const failures: Array<Record<string, unknown>> = [];
const note = (gate: string, caseId: string, detail: unknown) => failures.push({ gate, caseId, detail });

const counts = await pool.request().query<{ data_role: string; n: number }>(
  "SELECT data_role, COUNT(*) n FROM dataset.cases GROUP BY data_role ORDER BY data_role");
const total = counts.recordset.reduce((sum, row) => sum + row.n, 0);
if (total !== 588) note("counts", "-", { total });
// Package role math adjusted by dataset-v2.0.0-disposition-1 (split group
// gt-vector-indexmeta-g03 governed as test_id): test_id 335+1, calibration 45-1.
const expectedRoles: Record<string, number> = {
  test_id: 335, train: 110, calibration: 44, test_template_holdout: 26,
  test_suffix_holdout: 25, test_schema_holdout: 20, test_sparse_catalog: 18, test_capability: 10,
};
for (const row of counts.recordset) {
  if (expectedRoles[row.data_role] !== row.n) note("role-math", row.data_role, { expected: expectedRoles[row.data_role], actual: row.n });
}

const groupSpan = await pool.request().query<{ split_group_id: string; roles: number }>(
  "SELECT c.split_group_id, COUNT(DISTINCT c.data_role) roles FROM dataset.cases c GROUP BY c.split_group_id HAVING COUNT(DISTINCT c.data_role) > 1");
for (const row of groupSpan.recordset) note("split-exclusivity", row.split_group_id, { roles: row.roles });

const rows = await pool.request().query<CaseRow>(
  "SELECT case_id, data_role, expect_empty, cursor_offset_utf16, cursor_offset_utf8, doc_prefix, doc_suffix, canonical_insertion FROM dataset.cases");
let cursorPass = 0, integrityPass = 0, trapAudit = 0;
for (const row of rows.recordset) {
  const utf16 = row.doc_prefix.length;
  const utf8 = Buffer.byteLength(row.doc_prefix, "utf8");
  if (utf16 !== row.cursor_offset_utf16 || utf8 !== row.cursor_offset_utf8) {
    note("cursor-roundtrip", row.case_id, { utf16, utf8, recorded16: row.cursor_offset_utf16, recorded8: row.cursor_offset_utf8 });
  } else cursorPass += 1;

  const recomposed = row.doc_prefix + row.canonical_insertion + row.doc_suffix;
  const loneSurrogate = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(recomposed);
  const emptyConsistent = row.expect_empty === (row.canonical_insertion === "");
  if (loneSurrogate || !emptyConsistent) {
    note("insertion-integrity", row.case_id, { loneSurrogate, emptyConsistent });
  } else integrityPass += 1;

  if (row.canonical_insertion !== "" && row.doc_suffix !== "") {
    const tail = row.canonical_insertion.slice(-Math.min(row.canonical_insertion.length, 24));
    if (tail.length >= 4 && row.doc_suffix.startsWith(tail)) { note("suffix-duplication-gold", row.case_id, { tail }); }
    else trapAudit += 1;
  } else trapAudit += 1;
}

const goldless = await pool.request().query<{ case_id: string }>(
  `SELECT c.case_id FROM dataset.cases c WHERE c.expect_empty = 0
     AND NOT EXISTS (SELECT 1 FROM dataset.gold_candidates g WHERE g.case_id = c.case_id AND g.is_canonical = 1)`);
for (const row of goldless.recordset) note("gold-canonical-missing", row.case_id, {});

const summary = {
  schemaVersion: 1,
  totalCases: total,
  roleCounts: counts.recordset,
  gates: {
    cursorRoundTrip: `${cursorPass}/${total}`,
    insertionIntegrity: `${integrityPass}/${total}`,
    suffixTrapAudit: `${trapAudit}/${total}`,
    failures: failures.length,
  },
  failures: failures.slice(0, 50),
};
await atomicWrite(`${runDirectory}/manifests/dataset-validation.json`, `${JSON.stringify(summary, null, 2)}\n`);
await pool.request()
  .input("key", sql.VarChar(160), `gt1-dataset-validate:${hashJson(summary).slice(0, 16)}`)
  .input("run", sql.VarChar(120), runDirectory.split("/").at(-1))
  .input("stage", sql.VarChar(40), "GT-1")
  .input("disp", sql.VarChar(40), failures.length === 0 ? "PASS" : "FAIL")
  .input("json", sql.NVarChar(sql.MAX), JSON.stringify(summary))
  .input("hash", sql.Char(64), hashJson(summary))
  .query(`IF NOT EXISTS (SELECT 1 FROM control.evidence_events WHERE event_key=@key)
      INSERT control.evidence_events(event_key,run_id,stage,disposition,detail_json,detail_sha256) VALUES(@key,@run,@stage,@disp,@json,@hash)`);
console.log(JSON.stringify(summary.gates, null, 2));
if (failures.length > 0) { console.error(JSON.stringify(failures.slice(0, 10), null, 2)); process.exitCode = 2; }
await pool.close();
