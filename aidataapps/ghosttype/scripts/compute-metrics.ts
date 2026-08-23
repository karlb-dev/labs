import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

// GT-8 aggregation: per (profile, arm) metric rows from eval.row_scores +
// telemetry, and exact paired (McNemar sign-test) comparisons of each model
// profile against the deterministic baselines on normalized_exact.
// Claims (H1) are only emitted for profiles whose campaign is complete
// (a row for every case). Idempotent per run: aggregate rows rewritten.
//
// Power labels are PROVISIONAL until the E-1 power simulation lands:
// discordant-pair count >= 25 -> "provisional-adequate", else "exploratory".

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const runId = runDirectory.split("/").at(-1)!;
const pool = await new sql.ConnectionPool({ ...config.databases.admin, database: config.databases.controlName }).connect();

interface ScoreRow {
  case_id: string; profile_id: string; arm: string; data_role: string; outcome: string;
  normalized_exact: boolean | null; parse_pass: boolean | null; latency_ms: number | null;
  format_fail: boolean; completion_tokens: number | null;
}
const scores = (await pool.request().input("run", sql.VarChar(120), runId).query<ScoreRow>(
  `SELECT rs.case_id, rs.profile_id, rs.arm, c.data_role, rs.outcome, rs.normalized_exact,
          rs.parse_pass, rs.latency_ms, CAST(CASE WHEN rs.outcome='format_fail' THEN 1 ELSE 0 END AS bit) format_fail,
          rs.completion_tokens
   FROM eval.row_scores rs JOIN dataset.cases c ON c.case_id = rs.case_id
   WHERE rs.run_id = @run`)).recordset;
const totalCases = (await pool.request().query<{ n: number }>("SELECT COUNT(*) n FROM dataset.cases")).recordset[0].n;

const byProfileArm = new Map<string, ScoreRow[]>();
for (const row of scores) {
  const key = `${row.profile_id}|${row.arm}`;
  byProfileArm.set(key, [...(byProfileArm.get(key) ?? []), row]);
}

const quantile = (values: number[], q: number): number | null => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
};
const rate = (rows: ScoreRow[], predicate: (r: ScoreRow) => boolean) =>
  rows.length === 0 ? null : rows.filter(predicate).length / rows.length;

// exact two-sided binomial sign test on discordant pairs
function mcnemarP(b: number, c: number): number {
  const n = b + c;
  if (n === 0) return 1;
  const k = Math.min(b, c);
  let cumulative = 0;
  let coefficient = 1; // C(n, 0)
  for (let i = 0; i <= k; i += 1) {
    if (i > 0) coefficient = (coefficient * (n - i + 1)) / i;
    cumulative += coefficient * Math.pow(0.5, n);
  }
  return Math.min(1, 2 * cumulative);
}

await pool.request().input("run", sql.VarChar(120), runId)
  .query(`DELETE FROM eval.metric_results WHERE run_id=@run AND suite LIKE 'agg-%';
          DELETE FROM eval.claims WHERE run_id=@run`);

const manifest: Array<Record<string, unknown>> = [];
for (const [key, rows] of byProfileArm) {
  const [profileId, arm] = key.split("|");
  const complete = rows.length === totalCases;
  const suites: Array<[string, ScoreRow[]]> = [["agg-all", rows]];
  for (const role of new Set(rows.map((r) => r.data_role))) {
    suites.push([`agg-role-${role}`, rows.filter((r) => r.data_role === role)]);
  }
  for (const [suite, subset] of suites) {
    const latencies = subset.map((r) => r.latency_ms).filter((v): v is number => v !== null);
    const metrics: Array<[string, number | null]> = [
      ["normalized_exact_rate", rate(subset, (r) => r.normalized_exact === true)],
      ["success_or_correct_abstain_rate", rate(subset, (r) => r.outcome === "success" || r.outcome === "abstain_correct")],
      ["abstain_wrong_rate", rate(subset, (r) => r.outcome === "abstain_wrong")],
      ["format_fail_rate", rate(subset, (r) => r.format_fail)],
      ["truncation_rate", rate(subset, (r) => r.outcome === "fail" && r.parse_pass === null && r.normalized_exact === false)],
      ["parse_clean_rate_offered", rate(subset.filter((r) => r.parse_pass !== null), (r) => r.parse_pass === true)],
      ["latency_p50_ms", quantile(latencies, 0.5)],
      ["latency_p95_ms", quantile(latencies, 0.95)],
    ];
    for (const [name, value] of metrics) {
      if (value === null) continue;
      await pool.request()
        .input("run", sql.VarChar(120), runId).input("profile", sql.VarChar(80), profileId)
        .input("arm", sql.VarChar(40), arm).input("suite", sql.VarChar(60), suite)
        .input("name", sql.VarChar(80), name).input("value", sql.Float, value)
        .input("n", sql.Int, subset.length)
        .input("detail", sql.NVarChar(sql.MAX), JSON.stringify({ complete }))
        .query(`INSERT eval.metric_results(run_id,profile_id,arm,suite,metric_name,metric_value,rows_count,detail_json)
                VALUES(@run,@profile,@arm,@suite,@name,@value,@n,@detail)`);
    }
  }
  manifest.push({
    profileId, arm, rows: rows.length, complete,
    exactRate: rate(rows, (r) => r.normalized_exact === true),
    successOrAbstain: rate(rows, (r) => r.outcome === "success" || r.outcome === "abstain_correct"),
  });
}

// paired comparisons: every model profile arm vs each baseline arm
const baselineArms = ["B0-empty", "B2-catalog"];
const baselineRows = new Map<string, Map<string, boolean>>();
for (const arm of baselineArms) {
  const map = new Map<string, boolean>();
  for (const row of scores.filter((r) => r.profile_id === "baseline-deterministic" && r.arm === arm)) {
    map.set(row.case_id, row.normalized_exact === true);
  }
  baselineRows.set(arm, map);
}
const comparisons: Array<Record<string, unknown>> = [];
for (const [key, rows] of byProfileArm) {
  const [profileId, arm] = key.split("|");
  if (profileId === "baseline-deterministic") continue;
  const complete = rows.length === totalCases;
  for (const baselineArm of baselineArms) {
    const baseline = baselineRows.get(baselineArm)!;
    if (baseline.size === 0) continue;
    let modelOnly = 0, baselineOnly = 0, both = 0, neither = 0;
    for (const row of rows) {
      const modelExact = row.normalized_exact === true;
      const baseExact = baseline.get(row.case_id) ?? false;
      if (modelExact && !baseExact) modelOnly += 1;
      else if (!modelExact && baseExact) baselineOnly += 1;
      else if (modelExact) both += 1;
      else neither += 1;
    }
    const p = mcnemarP(modelOnly, baselineOnly);
    const discordant = modelOnly + baselineOnly;
    const powerLabel = discordant >= 25 ? "provisional-adequate" : "exploratory";
    const insert = await pool.request()
      .input("run", sql.VarChar(120), runId).input("profile", sql.VarChar(80), profileId)
      .input("arm", sql.VarChar(40), arm).input("suite", sql.VarChar(60), `agg-paired-vs-${baselineArm}`)
      .input("name", sql.VarChar(80), "mcnemar_exact_p").input("value", sql.Float, p)
      .input("n", sql.Int, rows.length)
      .input("detail", sql.NVarChar(sql.MAX), JSON.stringify({ modelOnly, baselineOnly, both, neither, complete, powerLabel }))
      .query(`INSERT eval.metric_results(run_id,profile_id,arm,suite,metric_name,metric_value,rows_count,detail_json)
              OUTPUT INSERTED.metric_id
              VALUES(@run,@profile,@arm,@suite,@name,@value,@n,@detail)`);
    const metricId = insert.recordset[0].metric_id as number;
    comparisons.push({ profileId, arm, baselineArm, modelOnly, baselineOnly, both, neither, p, powerLabel, complete });
    if (complete && baselineArm === "B2-catalog") {
      const supported = modelOnly > baselineOnly && p < 0.05;
      await pool.request()
        .input("run", sql.VarChar(120), runId).input("rq", sql.VarChar(20), "H1")
        .input("tag", sql.VarChar(20), supported ? "SUPPORTED" : "NOT_SUPPORTED")
        .input("tax", sql.VarChar(60), "quality/normalized-exact-vs-B2")
        .input("sup", sql.Bit, supported ? 1 : 0).input("power", sql.VarChar(30), powerLabel)
        .input("rat", sql.NVarChar(1000),
          `${profileId}/${arm} vs B2-catalog on normalized_exact: model-only ${modelOnly}, baseline-only ${baselineOnly}, exact McNemar p=${p.toExponential(2)}; power label provisional until E-1 simulation.`)
        .input("mid", sql.BigInt, metricId)
        .query(`INSERT eval.claims(run_id,research_question,evidence_tag,taxonomy,supported,power_label,rationale,metric_id)
                VALUES(@run,@rq,@tag,@tax,@sup,@power,@rat,@mid)`);
    }
  }
}

const summary = { schemaVersion: 1, stage: "GT-8", runId, profiles: manifest, comparisons };
await atomicWrite(`${runDirectory}/manifests/metrics.json`, `${JSON.stringify(summary, null, 2)}\n`);
await pool.request()
  .input("key", sql.VarChar(160), `gt8-metrics:${hashJson(summary).slice(0, 16)}`)
  .input("run", sql.VarChar(120), runId).input("stage", sql.VarChar(40), "GT-8")
  .input("disp", sql.VarChar(40), "PASS")
  .input("json", sql.NVarChar(sql.MAX), JSON.stringify(summary))
  .input("hash", sql.Char(64), hashJson(summary))
  .query(`IF NOT EXISTS (SELECT 1 FROM control.evidence_events WHERE event_key=@key)
      INSERT control.evidence_events(event_key,run_id,stage,disposition,detail_json,detail_sha256) VALUES(@key,@run,@stage,@disp,@json,@hash)`);
console.log(JSON.stringify(summary, null, 2));
await pool.close();
