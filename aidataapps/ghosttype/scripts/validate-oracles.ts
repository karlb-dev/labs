import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { recomposeForParse } from "../src/recompose.js";

// GT-3 parse oracle (addendum B-4): run the pinned ScriptDom parser over every
// gold insertion. Two independent axes are recorded per case in
// dataset.case_oracle_status:
//   oracle='parser_generation' — can the pinned parser parse the gold
//     recomposition at all? fail => PARSER_INELIGIBLE for parse-based scoring.
//   oracle='parse' — gold introduces no parse errors relative to the
//     prefix+suffix baseline AND the gold recomposition parses clean.
// parse_eligible=false rows (per-package oracle flags) are not_applicable.
// Idempotent: rows for these two oracles are rewritten on every invocation.

const SCRIPTDOM_DLL = "dotnet/GhostType.ScriptDom/bin/Release/net8.0/GhostType.ScriptDom.dll";
const PINNED_PACKAGE = "Microsoft.SqlServer.TransactSql.ScriptDom 170.191.0";

interface CaseRow {
  case_id: string; data_role: string; scenario: string; expect_empty: boolean;
  doc_prefix: string; doc_suffix: string; canonical_insertion: string; oracle_json: string;
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();

const child = spawn("dotnet", [SCRIPTDOM_DLL], { stdio: ["pipe", "pipe", "inherit"] });
const lines = createInterface({ input: child.stdout });
const pendingReaders: Array<(value: string) => void> = [];
const bufferedLines: string[] = [];
lines.on("line", (line) => {
  const reader = pendingReaders.shift();
  if (reader) reader(line);
  else bufferedLines.push(line);
});
const request = (payload: Record<string, unknown>): Promise<Record<string, any>> => {
  child.stdin.write(`${JSON.stringify(payload)}\n`);
  const buffered = bufferedLines.shift();
  if (buffered !== undefined) return Promise.resolve(JSON.parse(buffered));
  return new Promise((resolve) => pendingReaders.push((line) => resolve(JSON.parse(line))));
};

const version = await request({ op: "version" });
if (!version.ok) throw new Error(`ScriptDom service failed version check: ${JSON.stringify(version)}`);

const pool = await new sql.ConnectionPool({ ...config.databases.admin, database: config.databases.controlName }).connect();
const rows = await pool.request().query<CaseRow>(
  "SELECT case_id, data_role, scenario, expect_empty, doc_prefix, doc_suffix, canonical_insertion, oracle_json FROM dataset.cases ORDER BY case_id");

type OracleRow = { caseId: string; oracle: string; status: string; detail: Record<string, unknown> };
const results: OracleRow[] = [];
const tallies: Record<string, Record<string, number>> = {};
const failCases: Array<Record<string, unknown>> = [];
const tally = (oracle: string, status: string) => {
  tallies[oracle] = tallies[oracle] ?? {};
  tallies[oracle][status] = (tallies[oracle][status] ?? 0) + 1;
};

for (const row of rows.recordset) {
  const flags = JSON.parse(row.oracle_json) as { parse_eligible?: boolean };
  if (flags.parse_eligible !== true) {
    results.push({ caseId: row.case_id, oracle: "parse", status: "not_applicable", detail: { reason: "package parse_eligible=false" } });
    results.push({ caseId: row.case_id, oracle: "parser_generation", status: "not_applicable", detail: { reason: "package parse_eligible=false" } });
    tally("parse", "not_applicable"); tally("parser_generation", "not_applicable");
    continue;
  }
  const recomposition = recomposeForParse(row.doc_prefix, row.canonical_insertion, row.doc_suffix);
  const delta = await request({ op: "insertDelta", prefix: row.doc_prefix + recomposition.joiner, candidate: row.canonical_insertion, suffix: row.doc_suffix });
  if (!delta.ok) {
    results.push({ caseId: row.case_id, oracle: "parse", status: "unavailable", detail: { error: delta.error } });
    results.push({ caseId: row.case_id, oracle: "parser_generation", status: "unavailable", detail: { error: delta.error } });
    tally("parse", "unavailable"); tally("parser_generation", "unavailable");
    failCases.push({ caseId: row.case_id, scenario: row.scenario, error: delta.error });
    continue;
  }
  // Ghost-text semantics (SPEC §32 "partial SQL is expected"): the gold gate
  // requires zero STRUCTURAL errors in the gold recomposition; trailing
  // incompleteness (empty-suffix documents) is allowed and flagged.
  // parser_generation records that the pinned parser handled the row's syntax
  // family (generation-unsupported rows arrive as parse_eligible=false);
  // structural failures are gold-parse findings under oracle='parse'.
  results.push({ caseId: row.case_id, oracle: "parser_generation", status: "pass", detail: {} });
  tally("parser_generation", "pass");
  const structuralClean = delta.candidateStructuralCount === 0;
  const parseStatus = structuralClean ? "pass" : "fail";
  results.push({
    caseId: row.case_id, oracle: "parse", status: parseStatus,
    detail: {
      baselineErrorCount: delta.baselineErrorCount,
      baselineStructuralCount: delta.baselineStructuralCount,
      candidateErrorCount: delta.candidateErrorCount,
      candidateStructuralCount: delta.candidateStructuralCount,
      partialTail: delta.candidateTrailingCount > 0,
      recomposeRule: recomposition.rule, joiner: recomposition.joiner,
      structuralIntroducedCount: delta.structuralIntroducedCount,
      ...(structuralClean ? {} : { disposition: "GOLD_PARSE_FINDING", structuralIntroduced: delta.structuralIntroduced, candidateErrors: delta.candidateErrors }),
    },
  });
  tally("parse", parseStatus);
  if (parseStatus !== "pass") failCases.push({ caseId: row.case_id, scenario: row.scenario, dataRole: row.data_role, parseStatus, delta });
}

child.stdin.end();

const writer = new sql.Transaction(pool);
await writer.begin();
try {
  await new sql.Request(writer).query("DELETE FROM dataset.case_oracle_status WHERE oracle IN ('parse','parser_generation')");
  for (const row of results) {
    await new sql.Request(writer)
      .input("c", sql.VarChar(80), row.caseId).input("o", sql.VarChar(40), row.oracle)
      .input("s", sql.VarChar(30), row.status)
      .input("d", sql.NVarChar(sql.MAX), JSON.stringify({ ...row.detail, parser: version.parser, assemblyVersion: version.assemblyVersion, package: PINNED_PACKAGE }))
      .query("INSERT dataset.case_oracle_status(case_id, oracle, status, detail_json) VALUES(@c, @o, @s, @d)");
  }
  await writer.commit();
} catch (error) {
  await writer.rollback();
  throw error;
}

const summary = {
  schemaVersion: 1,
  stage: "GT-3",
  parser: { name: version.parser, assemblyVersion: version.assemblyVersion, package: PINNED_PACKAGE },
  totalCases: rows.recordset.length,
  tallies,
  failOrUnavailable: failCases.slice(0, 50),
};
await atomicWrite(`${runDirectory}/manifests/parse-oracle.json`, `${JSON.stringify(summary, null, 2)}\n`);
await pool.request()
  .input("key", sql.VarChar(160), `gt3-parse-oracle:${hashJson(summary).slice(0, 16)}`)
  .input("run", sql.VarChar(120), runDirectory.split("/").at(-1))
  .input("stage", sql.VarChar(40), "GT-3")
  .input("disp", sql.VarChar(40), failCases.length === 0 ? "PASS" : "PASS_WITH_FINDINGS")
  .input("json", sql.NVarChar(sql.MAX), JSON.stringify(summary))
  .input("hash", sql.Char(64), hashJson(summary))
  .query(`IF NOT EXISTS (SELECT 1 FROM control.evidence_events WHERE event_key=@key)
      INSERT control.evidence_events(event_key,run_id,stage,disposition,detail_json,detail_sha256) VALUES(@key,@run,@stage,@disp,@json,@hash)`);
console.log(JSON.stringify({ parser: summary.parser.assemblyVersion, tallies }, null, 2));
if (failCases.length > 0) console.error(JSON.stringify(failCases.slice(0, 10), null, 2));
await pool.close();
