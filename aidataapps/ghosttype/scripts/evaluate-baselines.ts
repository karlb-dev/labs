import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { recomposeForParse } from "../src/recompose.js";
import { normalizeV1, scoreCandidate, type EvalContract } from "../src/scoring.js";

// GT-5 deterministic baselines (spec §35 B0–B4; addendum K-1 grammar-aware
// B2). All arms run over all 588 cases; parse-eligible candidates get the
// pinned ScriptDom introduced-error check (recompose-v1); everything scores
// through the shared normalize-v1 scorer that model replay will reuse.
// Leakage contract: B3/B4 consult TRAIN-role rows only.
// Idempotent: baseline rows for the current run are rewritten per invocation.

const SCRIPTDOM_DLL = "dotnet/GhostType.ScriptDom/bin/Release/net8.0/GhostType.ScriptDom.dll";
const PROFILE_ID = "baseline-deterministic";

interface CaseRow {
  case_id: string; data_role: string; completion_class: string; catalog_snapshot_id: string;
  expect_empty: boolean; doc_prefix: string; doc_suffix: string; canonical_insertion: string;
  eval_json: string; oracle_json: string;
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const runId = runDirectory.split("/").at(-1)!;

const child = spawn("dotnet", [SCRIPTDOM_DLL], { stdio: ["pipe", "pipe", "inherit"] });
const lines = createInterface({ input: child.stdout });
const pendingReaders: Array<(value: string) => void> = [];
lines.on("line", (line) => pendingReaders.shift()?.(line));
const request = (payload: Record<string, unknown>): Promise<Record<string, any>> => {
  child.stdin.write(`${JSON.stringify(payload)}\n`);
  return new Promise((resolve) => pendingReaders.push((line) => resolve(JSON.parse(line))));
};

const pool = await new sql.ConnectionPool({ ...config.databases.admin, database: config.databases.controlName }).connect();
const cases = (await pool.request().query<CaseRow>(
  `SELECT case_id, data_role, completion_class, catalog_snapshot_id, expect_empty,
          doc_prefix, doc_suffix, canonical_insertion, eval_json, oracle_json
   FROM dataset.cases ORDER BY case_id`)).recordset;
const goldRows = (await pool.request().query<{ case_id: string; insertion: string }>(
  "SELECT case_id, insertion FROM dataset.gold_candidates ORDER BY case_id, ordinal")).recordset;
const goldByCase = new Map<string, string[]>();
for (const row of goldRows) {
  goldByCase.set(row.case_id, [...(goldByCase.get(row.case_id) ?? []), row.insertion]);
}
const catalogRows = (await pool.request().query<{
  snapshot_id: string; object_kind: string; schema_name: string | null; object_name: string;
  unavailable: boolean; column_name: string | null; ordinal: number | null; is_parameter: boolean | null;
}>(`SELECT o.snapshot_id, o.object_kind, o.schema_name, o.object_name, o.unavailable,
           c.column_name, c.ordinal, c.is_parameter
    FROM catalog.objects o LEFT JOIN catalog.columns c ON c.object_pk = o.object_pk
    ORDER BY o.snapshot_id, o.object_pk, c.ordinal`)).recordset;

interface CatalogObject { kind: string; schemaName: string | null; name: string; unavailable: boolean; columns: string[] }
const catalogBySnapshot = new Map<string, CatalogObject[]>();
for (const row of catalogRows) {
  const list = catalogBySnapshot.get(row.snapshot_id) ?? [];
  if (list.length === 0 || list.at(-1)!.name !== row.object_name || list.at(-1)!.schemaName !== row.schema_name || list.at(-1)!.kind !== row.object_kind) {
    list.push({ kind: row.object_kind, schemaName: row.schema_name, name: row.object_name, unavailable: row.unavailable, columns: [] });
  }
  if (row.column_name !== null && row.is_parameter === false) list.at(-1)!.columns.push(row.column_name);
  catalogBySnapshot.set(row.snapshot_id, list);
}

// ---------- B1: frozen keyword-bigram table (pure grammar floor) ----------
const KEYWORD_TABLE: Record<string, string> = {
  ORDER: "BY", GROUP: "BY", PARTITION: "BY", INNER: "JOIN", OUTER: "JOIN", CROSS: "JOIN",
  IS: "NULL", INSERT: "INTO", DELETE: "FROM", UNION: "ALL", PRIMARY: "KEY", FOREIGN: "KEY",
  NOT: "NULL", TOP: "(", BETWEEN: "", EXISTS: "(",
};

// ---------- shared prefix-token context ----------
interface TokenInfo { type: string; text: string; offset: number }
function aliasMap(prefix: string): Map<string, string> {
  const map = new Map<string, string>();
  const pattern = /(?:FROM|JOIN)\s+((?:\[[^\]]+\]|[\w.])+)\s+(?:AS\s+)?(\[[^\]]+\]|\w+)/gi;
  for (const match of prefix.matchAll(pattern)) {
    const alias = match[2].replace(/[\[\]]/g, "");
    if (/^(ON|WHERE|GROUP|ORDER|INNER|LEFT|RIGHT|FULL|CROSS|JOIN|SET|WITH)$/i.test(alias)) continue;
    map.set(alias.toLowerCase(), match[1].replace(/[\[\]]/g, ""));
  }
  return map;
}
const fullName = (object: CatalogObject) => (object.schemaName ? `${object.schemaName}.${object.name}` : object.name);

// ---------- B2: grammar-aware catalog baseline (addendum K-1) ----------
// Category from tail tokens -> candidate list -> frozen ranking:
// prefix-match filter, then mention-count in prefix desc, then catalog
// ordinal (stable insertion order), then name ascending. Detected category
// is recorded per row for the coverage report.
function b2Candidate(row: CaseRow, tokens: TokenInfo[]): { text: string; category: string } {
  const objects = catalogBySnapshot.get(row.catalog_snapshot_id) ?? [];
  const visible = objects.filter((o) => !o.unavailable);
  const tail = tokens.slice(-4);
  const last = tail.at(-1);
  const prev = tail.at(-2);
  const prefixText = row.doc_prefix;
  const endsWithSpace = /\s$/.test(prefixText);
  const partial = !endsWithSpace && last?.type === "Identifier" ? last.text : "";

  const rank = (names: string[]): string[] => {
    const filtered = partial === "" ? names : names.filter((n) => n.toLowerCase().startsWith(partial.toLowerCase()) && n !== partial);
    const mentions = (name: string) => (prefixText.toLowerCase().match(new RegExp(name.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g")) ?? []).length;
    return [...filtered].sort((a, b) => mentions(b) - mentions(a) || names.indexOf(a) - names.indexOf(b) || a.localeCompare(b));
  };

  // alias-dot: "c." or "c.Na"
  const dotIndex = partial !== "" ? -3 : -2;
  if ((last?.text === "." ) || (tail.at(dotIndex + 1)?.text === "." && partial !== "")) {
    const aliasToken = last?.text === "." ? prev : tail.at(dotIndex);
    const alias = aliasToken?.text?.replace(/[\[\]]/g, "").toLowerCase() ?? "";
    const aliases = aliasMap(prefixText);
    const tableName = aliases.get(alias);
    if (tableName) {
      const table = visible.find((o) => fullName(o).toLowerCase() === tableName.toLowerCase() || o.name.toLowerCase() === tableName.toLowerCase());
      if (table && table.columns.length > 0) {
        const ranked = rank(table.columns);
        if (ranked.length > 0) return { text: ranked[0], category: "column_of_alias" };
      }
      return { text: "", category: "column_of_alias_unresolved" };
    }
    // schema-dot: "dbo." -> tables in that schema
    const schemaTables = visible.filter((o) => (o.kind === "table" || o.kind === "view") && o.schemaName?.toLowerCase() === alias);
    if (schemaTables.length > 0) {
      const ranked = rank(schemaTables.map((o) => o.name));
      if (ranked.length > 0) return { text: ranked[0], category: "table_of_schema" };
    }
    return { text: "", category: "dot_unresolved" };
  }

  const lastKeyword = [...tail].reverse().find((t) => t.type !== "Identifier" && t.type !== "Dot")?.text?.toUpperCase();
  if (lastKeyword === "FROM" || lastKeyword === "JOIN" || (prev?.text?.toUpperCase() === "FROM" && partial !== "") || (prev?.text?.toUpperCase() === "JOIN" && partial !== "")) {
    const tables = visible.filter((o) => o.kind === "table" || o.kind === "view" || o.kind === "synonym");
    const ranked = rank(tables.map(fullName));
    if (ranked.length > 0) return { text: ranked[0], category: "table_source" };
    return { text: "", category: "table_source_empty" };
  }
  if (lastKeyword === "EXEC" || lastKeyword === "EXECUTE") {
    const procs = visible.filter((o) => o.kind === "procedure" || o.kind === "function");
    const ranked = rank(procs.map(fullName));
    if (ranked.length > 0) return { text: ranked[0], category: "procedure" };
    return { text: "", category: "procedure_empty" };
  }
  return { text: "", category: "no_category" };
}

// ---------- B3: accepted-history n-gram over TRAIN rows only ----------
const trainRows = cases.filter((row) => row.data_role === "train");
const b3Index = new Map<string, Map<string, number>>();
const b3Key = (row: CaseRow) => {
  const tokens = normalizeV1(row.doc_prefix).split(" ").slice(-3).join(" ").toLowerCase();
  return `${row.completion_class}|${tokens}`;
};
for (const row of trainRows) {
  const key = b3Key(row);
  const counts = b3Index.get(key) ?? new Map<string, number>();
  counts.set(row.canonical_insertion, (counts.get(row.canonical_insertion) ?? 0) + 1);
  b3Index.set(key, counts);
}
function b3Candidate(row: CaseRow): string {
  const counts = b3Index.get(b3Key(row));
  if (!counts) return "";
  // Self-exclusion: a train row never consults its own gold.
  const entries = [...counts.entries()]
    .map(([insertion, count]) => [insertion, row.data_role === "train" && insertion === row.canonical_insertion ? count - 1 : count] as const)
    .filter(([, count]) => count > 0);
  if (entries.length === 0) return "";
  return [...entries].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0][0];
}

// ---------- B4: best TRAIN exemplar, same catalog + class ----------
const tokenSet = (text: string) => new Set(normalizeV1(text).toLowerCase().split(" ").slice(-24));
const b4Pool = trainRows.map((row) => ({ row, tokens: tokenSet(row.doc_prefix) }));
function b4Candidate(row: CaseRow): string {
  let best: { score: number; insertion: string; id: string } | null = null;
  const target = tokenSet(row.doc_prefix);
  for (const exemplar of b4Pool) {
    if (exemplar.row.case_id === row.case_id) continue;
    if (exemplar.row.catalog_snapshot_id !== row.catalog_snapshot_id) continue;
    if (exemplar.row.completion_class !== row.completion_class) continue;
    let overlap = 0;
    for (const token of target) if (exemplar.tokens.has(token)) overlap += 1;
    const score = overlap / (target.size + exemplar.tokens.size - overlap || 1);
    if (!best || score > best.score || (score === best.score && exemplar.row.case_id < best.id)) {
      best = { score, insertion: exemplar.row.canonical_insertion, id: exemplar.row.case_id };
    }
  }
  return best && best.score > 0.2 ? best.insertion : "";
}

// ---------- run all arms ----------
const version = await request({ op: "version" });
if (!version.ok) throw new Error("ScriptDom service failed");
const arms = ["B0-empty", "B1-keyword", "B2-catalog", "B3-history", "B4-template"] as const;
const b2Categories: Record<string, number> = {};
const armTallies: Record<string, Record<string, number>> = {};
const rows: Array<Record<string, unknown>> = [];

for (const row of cases) {
  const contract = JSON.parse(row.eval_json) as EvalContract;
  const flags = JSON.parse(row.oracle_json) as { parse_eligible?: boolean };
  const accepted = goldByCase.get(row.case_id) ?? [];
  let tokens: TokenInfo[] = [];
  const tokenResult = await request({ op: "tokens", sql: row.doc_prefix });
  if (tokenResult.ok) tokens = tokenResult.tokens;

  for (const arm of arms) {
    let candidate = "";
    let armDetail: Record<string, unknown> = {};
    if (arm === "B1-keyword") {
      const lastWord = tokens.at(-1)?.text?.toUpperCase() ?? "";
      candidate = KEYWORD_TABLE[lastWord] ?? "";
    } else if (arm === "B2-catalog") {
      const picked = b2Candidate(row, tokens);
      candidate = picked.text;
      armDetail = { category: picked.category };
      b2Categories[picked.category] = (b2Categories[picked.category] ?? 0) + 1;
    } else if (arm === "B3-history") {
      candidate = b3Candidate(row);
    } else if (arm === "B4-template") {
      candidate = b4Candidate(row);
    }

    let parseIntroduced: number | null = null;
    if (flags.parse_eligible === true && candidate !== "") {
      const joiner = recomposeForParse(row.doc_prefix, candidate, row.doc_suffix).joiner;
      const delta = await request({ op: "insertDelta", prefix: row.doc_prefix + joiner, candidate, suffix: row.doc_suffix });
      if (delta.ok) parseIntroduced = delta.structuralIntroducedCount;
    }

    const score = scoreCandidate({
      candidate, contract, acceptedInsertions: accepted,
      docPrefix: row.doc_prefix, docSuffix: row.doc_suffix, parseIntroduced,
    });
    armTallies[arm] = armTallies[arm] ?? {};
    armTallies[arm][score.outcome] = (armTallies[arm][score.outcome] ?? 0) + 1;
    rows.push({
      caseId: row.case_id, arm, candidate, dataRole: row.data_role,
      score, detail: { ...score.detail, ...armDetail },
    });
  }
}
child.stdin.end();

const writer = new sql.Transaction(pool);
await writer.begin();
try {
  await new sql.Request(writer)
    .input("run", sql.VarChar(120), runId).input("profile", sql.VarChar(80), PROFILE_ID)
    .query("DELETE FROM eval.row_scores WHERE run_id=@run AND profile_id=@profile");
  for (const row of rows) {
    const score = row.score as ReturnType<typeof scoreCandidate>;
    await new sql.Request(writer)
      .input("run", sql.VarChar(120), runId).input("case", sql.VarChar(80), row.caseId as string)
      .input("profile", sql.VarChar(80), PROFILE_ID).input("arm", sql.VarChar(40), row.arm as string)
      .input("eligible", sql.Bit, 1).input("outcome", sql.VarChar(30), score.outcome)
      .input("nx", sql.Bit, score.normalizedExact).input("cp", sql.Bit, score.constraintPass)
      .input("gp", sql.Bit, score.groundingPass).input("ep", sql.Bit, score.emptyPolicyPass)
      .input("nm", sql.Bit, score.noMarkdownPass).input("ii", sql.Bit, score.insertionIntegrityPass)
      .input("pp", sql.Bit, score.parsePass)
      .input("detail", sql.NVarChar(sql.MAX), JSON.stringify({ candidate: row.candidate, ...(row.detail as object) }))
      .query(`INSERT eval.row_scores(run_id,case_id,profile_id,arm,eligible,outcome,normalized_exact,constraint_pass,grounding_pass,empty_policy_pass,no_markdown_pass,insertion_integrity_pass,parse_pass,detail_json)
              VALUES(@run,@case,@profile,@arm,@eligible,@outcome,@nx,@cp,@gp,@ep,@nm,@ii,@pp,@detail)`);
  }
  await writer.commit();
} catch (error) {
  await writer.rollback();
  throw error;
}

// aggregate metric rows per arm x role
await pool.request()
  .input("run", sql.VarChar(120), runId).input("profile", sql.VarChar(80), PROFILE_ID)
  .query(`DELETE FROM eval.metric_results WHERE run_id=@run AND profile_id=@profile;
    INSERT eval.metric_results(run_id,profile_id,arm,suite,metric_name,metric_value,rows_count,detail_json)
    SELECT rs.run_id, rs.profile_id, rs.arm, c.data_role, 'normalized_exact_rate',
           AVG(CAST(CASE WHEN rs.normalized_exact=1 THEN 1.0 ELSE 0.0 END AS float)), COUNT(*), NULL
    FROM eval.row_scores rs JOIN dataset.cases c ON c.case_id = rs.case_id
    WHERE rs.run_id=@run AND rs.profile_id=@profile GROUP BY rs.run_id, rs.profile_id, rs.arm, c.data_role;
    INSERT eval.metric_results(run_id,profile_id,arm,suite,metric_name,metric_value,rows_count,detail_json)
    SELECT rs.run_id, rs.profile_id, rs.arm, 'all', 'success_rate',
           AVG(CAST(CASE WHEN rs.outcome IN ('success','abstain_correct') THEN 1.0 ELSE 0.0 END AS float)), COUNT(*), NULL
    FROM eval.row_scores rs WHERE rs.run_id=@run AND rs.profile_id=@profile GROUP BY rs.run_id, rs.profile_id, rs.arm;`);

const summary = {
  schemaVersion: 1, stage: "GT-5", runId, profileId: PROFILE_ID,
  parser: version.assemblyVersion, cases: cases.length,
  armOutcomes: armTallies, b2CategoryCoverage: b2Categories,
  normalizeRule: "normalize-v1", recomposeRule: "recompose-v1",
};
await atomicWrite(`${runDirectory}/manifests/baselines.json`, `${JSON.stringify(summary, null, 2)}\n`);
await pool.request()
  .input("key", sql.VarChar(160), `gt5-baselines:${hashJson(summary).slice(0, 16)}`)
  .input("run", sql.VarChar(120), runId).input("stage", sql.VarChar(40), "GT-5")
  .input("disp", sql.VarChar(40), "PASS")
  .input("json", sql.NVarChar(sql.MAX), JSON.stringify(summary))
  .input("hash", sql.Char(64), hashJson(summary))
  .query(`IF NOT EXISTS (SELECT 1 FROM control.evidence_events WHERE event_key=@key)
      INSERT control.evidence_events(event_key,run_id,stage,disposition,detail_json,detail_sha256) VALUES(@key,@run,@stage,@disp,@json,@hash)`);
console.log(JSON.stringify(summary, null, 2));
await pool.close();
