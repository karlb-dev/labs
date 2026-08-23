import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { sha256 } from "../src/hash.js";
import { LAB_ROOT, valueAfter } from "../src/run.js";

// GT-1: import the vendored dataset package into immutable SQL rows.
// Re-verifies every record hash against the package manifest during import
// (canonical recipe: sha256(utf8(json ensure_ascii=false sort_keys compact))).

interface RecordRow {
  id: string;
  dataset_version: string;
  completion_category: string;
  prompt: Record<string, string | boolean>;
  messages: Array<{ role: string; content: string }>;
  gold_completion: string;
  eval: Record<string, unknown>;
  source: Record<string, unknown>;
  environment: Record<string, unknown>;
  ghosttype: Record<string, unknown>;
}

function canonicalPython(value: unknown): string {
  // Match the package generator: json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : JSON.stringify(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalPython).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalPython(v)}`).join(",")}}`;
}

const bundle = valueAfter("--bundle") ?? `${LAB_ROOT}/data/ghosttype_dataset_v2`;
const config = loadConfig();
const lines = (await readFile(`${bundle}/records/ghosttype_sql_completions_v2.jsonl`, "utf8")).split("\n").filter(Boolean);
const records = lines.map((line) => JSON.parse(line) as RecordRow);
const hashManifest = JSON.parse(await readFile(`${bundle}/manifests/record-hashes.json`, "utf8")) as Record<string, string>;

let hashMatches = 0;
for (const record of records) {
  if (sha256(Buffer.from(canonicalPython(record), "utf8")) === hashManifest[record.id]) hashMatches += 1;
}
if (hashMatches !== records.length) throw new Error(`STOP_DATA: record hash verification ${hashMatches}/${records.length}`);

const dispositions = JSON.parse(await readFile(`${LAB_ROOT}/config/dataset-dispositions.json`, "utf8")) as {
  splitGroupRoleOverrides: Record<string, string>;
};
const pool = await new sql.ConnectionPool({ ...config.databases.admin, database: config.databases.controlName }).connect();
const already = await pool.request().query<{ n: number }>("SELECT COUNT(*) n FROM dataset.cases");
if (already.recordset[0]!.n === records.length) {
  console.log(JSON.stringify({ status: "already-imported", cases: already.recordset[0]!.n, hashMatches }));
  await pool.close();
  process.exit(0);
}
if (already.recordset[0]!.n > 0) {
  // Partial import from an interrupted run: wipe and reload (rows are immutable inputs, safe to rebuild).
  console.log(JSON.stringify({ status: "partial-detected", wiping: already.recordset[0]!.n }));
  await pool.request().query("DELETE FROM dataset.gold_candidates; DELETE FROM dataset.split_groups; DELETE FROM dataset.cases; DELETE FROM dataset.dataset_versions;");
}

const version = records[0]!.dataset_version;
await pool.request()
  .input("v", sql.VarChar(40), version)
  .input("p", sql.NVarChar(400), bundle.replace(LAB_ROOT, "."))
  .input("n", sql.Int, records.length)
  .input("l", sql.Int, records.filter((r) => r.ghosttype.legacy_id).length)
  .input("h", sql.Char(64), sha256(await readFile(`${bundle}/manifests/record-hashes.json`)))
  .query("INSERT dataset.dataset_versions(dataset_version,package_path,record_count,legacy_count,manifest_sha256) VALUES(@v,@p,@n,@l,@h)");

const groups = new Map<string, { role: string; count: number }>();
let inserted = 0;
for (const record of records) {
  const g = record.ghosttype as Record<string, any>;
  const e = record.eval as Record<string, any>;
  const prompt = record.prompt as Record<string, any>;
  const promptMessage = record.messages.map((m) => m.content).join("\n\n");
  // Package cursor offsets are statement-relative: the insertion base is
  // current_statement_prefix; recent_document_prefix is separate context.
  const docPrefix = `${prompt.current_statement_prefix ?? ""}`;
  const recentPrefix = `${prompt.recent_document_prefix ?? ""}`;
  const dataRole = dispositions.splitGroupRoleOverrides[g.split_group_id as string] ?? g.data_role;
  await pool.request()
    .input("case_id", sql.VarChar(80), record.id)
    .input("v", sql.VarChar(40), version)
    .input("legacy", sql.VarChar(80), g.legacy_id ?? null)
    .input("rhash", sql.Char(64), hashManifest[record.id])
    .input("role", sql.VarChar(40), dataRole)
    .input("group_id", sql.VarChar(120), g.split_group_id)
    .input("tfam", sql.VarChar(120), g.template_family)
    .input("nfam", sql.VarChar(120), g.near_duplicate_family)
    .input("cls", sql.VarChar(40), g.completion_class)
    .input("cat", sql.VarChar(40), record.completion_category)
    .input("dialect", sql.VarChar(20), g.dialect)
    .input("compat", sql.Int, g.compatibility_level)
    .input("fixture", sql.VarChar(80), g.fixture_id)
    .input("snapshot", sql.VarChar(120), g.catalog_snapshot_id)
    .input("perm", sql.VarChar(80), g.permission_profile_id)
    .input("c16", sql.Int, g.cursor.offset_utf16)
    .input("c8", sql.Int, g.cursor.offset_utf8)
    .input("s0", sql.Int, g.cursor.selection_start_utf16)
    .input("s1", sql.Int, g.cursor.selection_end_utf16)
    .input("dp", sql.NVarChar(sql.MAX), docPrefix)
    .input("rp", sql.NVarChar(sql.MAX), recentPrefix)
    .input("ds", sql.NVarChar(sql.MAX), prompt.document_suffix ?? "")
    .input("lp", sql.NVarChar(sql.MAX), prompt.current_line_prefix ?? "")
    .input("ls", sql.NVarChar(sql.MAX), prompt.current_line_suffix ?? "")
    .input("gold", sql.NVarChar(sql.MAX), g.canonical_insertion)
    .input("empty", sql.Bit, e.expect_empty === true)
    .input("ambig", sql.Bit, g.known_ambiguity === true)
    .input("absr", sql.NVarChar(400), g.abstention_reason ?? null)
    .input("diff", sql.VarChar(20), e.difficulty ?? "unknown")
    .input("scen", sql.VarChar(40), e.scenario ?? "unknown")
    .input("caps", sql.NVarChar(sql.MAX), JSON.stringify(g.capability_requirements ?? []))
    .input("mref", sql.NVarChar(sql.MAX), JSON.stringify(g.must_reference_objects ?? []))
    .input("mnot", sql.NVarChar(sql.MAX), JSON.stringify(g.must_not_reference_objects ?? []))
    .input("oracle", sql.NVarChar(sql.MAX), JSON.stringify(g.oracle ?? {}))
    .input("evalj", sql.NVarChar(sql.MAX), JSON.stringify(record.eval))
    .input("pm", sql.NVarChar(sql.MAX), promptMessage)
    .input("ph", sql.Char(64), sha256(promptMessage))
    .input("src", sql.NVarChar(sql.MAX), JSON.stringify(record.source))
    .input("env", sql.NVarChar(sql.MAX), JSON.stringify(record.environment))
    .query(`INSERT dataset.cases(case_id,dataset_version,legacy_id,record_sha256,data_role,split_group_id,template_family,near_duplicate_family,
        completion_class,completion_category,dialect,compatibility_level,fixture_id,catalog_snapshot_id,permission_profile_id,
        cursor_offset_utf16,cursor_offset_utf8,selection_start_utf16,selection_end_utf16,doc_prefix,recent_prefix,doc_suffix,line_prefix,line_suffix,
        canonical_insertion,expect_empty,known_ambiguity,abstention_reason,difficulty,scenario,
        capability_requirements_json,must_reference_json,must_not_reference_json,oracle_json,eval_json,prompt_message,prompt_sha256,source_json,environment_json)
      VALUES(@case_id,@v,@legacy,@rhash,@role,@group_id,@tfam,@nfam,@cls,@cat,@dialect,@compat,@fixture,@snapshot,@perm,
        @c16,@c8,@s0,@s1,@dp,@rp,@ds,@lp,@ls,@gold,@empty,@ambig,@absr,@diff,@scen,@caps,@mref,@mnot,@oracle,@evalj,@pm,@ph,@src,@env)`);
  const accepted = (g.accepted_insertions ?? []) as string[];
  for (let i = 0; i < accepted.length; i += 1) {
    await pool.request()
      .input("case_id", sql.VarChar(80), record.id)
      .input("ord", sql.Int, i)
      .input("text", sql.NVarChar(sql.MAX), accepted[i])
      .input("canon", sql.Bit, accepted[i] === g.canonical_insertion)
      .query("INSERT dataset.gold_candidates(case_id,ordinal,insertion,is_canonical) VALUES(@case_id,@ord,@text,@canon)");
  }
  const entry = groups.get(g.split_group_id);
  if (entry) { entry.count += 1; if (entry.role !== dataRole) throw new Error(`STOP_DATA: split group ${g.split_group_id} spans roles after dispositions`); }
  else groups.set(g.split_group_id, { role: dataRole, count: 1 });
  inserted += 1;
}
for (const [groupId, { role, count }] of groups) {
  await pool.request().input("g", sql.VarChar(120), groupId).input("r", sql.VarChar(40), role).input("n", sql.Int, count)
    .query("INSERT dataset.split_groups(split_group_id,data_role,case_count) VALUES(@g,@r,@n)");
}
console.log(JSON.stringify({ imported: inserted, goldRows: records.length, splitGroups: groups.size, hashMatches }, null, 2));
await pool.close();
