import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

// Tier-1 report pack, generated from SQL only (the database is the source
// of record; reports are always regenerable). Markdown lands in reports/.
// Adaptation note (logged in EXPERIMENT_LOG): implemented in TS rather than
// the spec's analysis/build_reports.py so the lab keeps a single stack; the
// Colab lift can keep the same command surface.

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const runId = runDirectory.split("/").at(-1)!;
const pool = await new sql.ConnectionPool({ ...config.databases.admin, database: config.databases.controlName }).connect();
const generatedAt = new Date().toISOString();
const header = (title: string) =>
  `# ${title}\n\nrun: \`${runId}\` · generated ${generatedAt} · lab: aidataapps/ghosttype (Mac local campaign)\n\n> Regenerated from GhostTypeControl by \`npm run reports\`. The database is the source of record.\n\n`;

const table = (headers: string[], rows: Array<Array<string | number | null>>): string => {
  const line = (cells: Array<string | number | null>) => `| ${cells.map((c) => (c === null ? "—" : String(c))).join(" | ")} |`;
  return [line(headers), `|${headers.map(() => "---").join("|")}|`, ...rows.map(line)].join("\n") + "\n";
};
const pct = (value: number | null) => (value === null ? null : `${(value * 100).toFixed(1)}%`);

// ---------------- DATASET_REPORT ----------------
const roleCounts = (await pool.request().query<{ data_role: string; completion_class: string; n: number }>(
  "SELECT data_role, completion_class, COUNT(*) n FROM dataset.cases GROUP BY data_role, completion_class ORDER BY data_role")).recordset;
const catalogCounts = (await pool.request().query<{ catalog_snapshot_id: string; n: number }>(
  "SELECT catalog_snapshot_id, COUNT(*) n FROM dataset.cases GROUP BY catalog_snapshot_id ORDER BY n DESC")).recordset;
const oracleCounts = (await pool.request().query<{ oracle: string; status: string; n: number }>(
  "SELECT oracle, status, COUNT(*) n FROM dataset.case_oracle_status GROUP BY oracle, status ORDER BY oracle, status")).recordset;
const parseFails = (await pool.request().query<{ case_id: string; data_role: string }>(
  "SELECT s.case_id, c.data_role FROM dataset.case_oracle_status s JOIN dataset.cases c ON c.case_id=s.case_id WHERE s.oracle='parse' AND s.status='fail' ORDER BY s.case_id")).recordset;
const b2Coverage = (await pool.request().query<{ category: string; n: number }>(
  `SELECT JSON_VALUE(detail_json, '$.category') category, COUNT(*) n
   FROM eval.row_scores WHERE profile_id='baseline-deterministic' AND arm='B2-catalog'
   GROUP BY JSON_VALUE(detail_json, '$.category') ORDER BY n DESC`)).recordset;

let datasetReport = header("GhostType DATASET_REPORT (Tier 1)");
datasetReport += "## Role × class composition (post-disposition)\n\n";
datasetReport += table(["role", "class", "cases"], roleCounts.map((r) => [r.data_role, r.completion_class, r.n]));
datasetReport += "\n## Catalog composition\n\n";
datasetReport += table(["catalog snapshot", "cases"], catalogCounts.map((r) => [r.catalog_snapshot_id, r.n]));
datasetReport += "\n## Oracle status (GT-3, ScriptDom pinned 170.191.0)\n\n";
datasetReport += table(["oracle", "status", "cases"], oracleCounts.map((r) => [r.oracle, r.status, r.n]));
datasetReport += "\n### Adjudicated package defects (golds untouched; logged for dataset v2.1)\n\n";
datasetReport += table(["case", "role", "finding"], parseFails.map((r) => [r.case_id, r.data_role,
  r.case_id.startsWith("gt-sparse-alias") || r.case_id === "gt-edge-cont-05" ? "cursor placed after statement terminator" : "gold is syntactically invalid T-SQL"]));
datasetReport += "\nPlus two split-group role-governance dispositions (`config/dataset-dispositions.json`).\n";
datasetReport += "\n## B2 category-detection coverage (addendum K-1 requirement)\n\n";
datasetReport += table(["category", "rows"], b2Coverage.map((r) => [r.category ?? "(model rows)", r.n]));
await atomicWrite("reports/DATASET_REPORT.md", datasetReport);

// ---------------- COMPLETION_QUALITY_REPORT ----------------
const metricRows = (await pool.request().input("run", sql.VarChar(120), runId).query<{
  profile_id: string; arm: string; suite: string; metric_name: string; metric_value: number; rows_count: number; detail_json: string | null;
}>("SELECT profile_id, arm, suite, metric_name, metric_value, rows_count, detail_json FROM eval.metric_results WHERE run_id=@run AND suite IN ('agg-all') OR (run_id=@run AND suite LIKE 'agg-paired%') ORDER BY profile_id, arm, suite, metric_name")).recordset;
const outcomeRows = (await pool.request().input("run", sql.VarChar(120), runId).query<{
  profile_id: string; arm: string; outcome: string; n: number; total: number;
}>(`SELECT profile_id, arm, outcome, COUNT(*) n,
      SUM(COUNT(*)) OVER (PARTITION BY profile_id, arm) total
    FROM eval.row_scores WHERE run_id=@run GROUP BY profile_id, arm, outcome ORDER BY profile_id, arm, outcome`)).recordset;

let qualityReport = header("GhostType COMPLETION_QUALITY_REPORT (Tier 1, mac campaign)");
qualityReport += "All rows use the frozen rules: raw-text transport, candidate-extract-v1, normalize-v1 (case-sensitive), recompose-v1, ScriptDom [170.191.0] parse delta, deterministic reference decode (T=0). Campaigns marked incomplete are in progress — numbers move until the profile's 588 rows are terminal.\n\n";
qualityReport += "## Outcomes by profile/arm\n\n";
const outcomeKinds = ["success", "partial", "fail", "format_fail", "abstain_correct", "abstain_wrong"];
const armKeys = [...new Set(outcomeRows.map((r) => `${r.profile_id}|${r.arm}`))];
qualityReport += table(["profile", "arm", "rows", ...outcomeKinds], armKeys.map((key) => {
  const [profile, arm] = key.split("|");
  const rows = outcomeRows.filter((r) => r.profile_id === profile && r.arm === arm);
  const total = rows[0]?.total ?? 0;
  return [profile, arm, `${total}${total < 588 ? " ⏳" : ""}`,
    ...outcomeKinds.map((kind) => rows.find((r) => r.outcome === kind)?.n ?? 0)];
}));
qualityReport += "\n## Aggregate metrics (suite agg-all)\n\n";
const metricNames = ["normalized_exact_rate", "success_or_correct_abstain_rate", "abstain_wrong_rate", "format_fail_rate", "parse_clean_rate_offered", "latency_p50_ms", "latency_p95_ms"];
const metricKeys = [...new Set(metricRows.filter((r) => r.suite === "agg-all").map((r) => `${r.profile_id}|${r.arm}`))];
qualityReport += table(["profile", "arm", ...metricNames.map((n) => n.replace(/_/g, " "))], metricKeys.map((key) => {
  const [profile, arm] = key.split("|");
  const find = (name: string) => metricRows.find((r) => r.profile_id === profile && r.arm === arm && r.suite === "agg-all" && r.metric_name === name)?.metric_value ?? null;
  return [profile, arm,
    pct(find("normalized_exact_rate")), pct(find("success_or_correct_abstain_rate")), pct(find("abstain_wrong_rate")),
    pct(find("format_fail_rate")), pct(find("parse_clean_rate_offered")),
    find("latency_p50_ms")?.toFixed(0) ?? null, find("latency_p95_ms")?.toFixed(0) ?? null];
}));
qualityReport += "\n## Paired comparisons (exact McNemar on normalized_exact)\n\n";
const paired = metricRows.filter((r) => r.suite.startsWith("agg-paired"));
qualityReport += table(["profile", "arm", "vs", "model-only", "baseline-only", "p (exact)", "power label"], paired.map((r) => {
  const detail = JSON.parse(r.detail_json ?? "{}");
  return [r.profile_id, r.arm, r.suite.replace("agg-paired-vs-", ""), detail.modelOnly, detail.baselineOnly,
    r.metric_value.toExponential(2), detail.powerLabel + (detail.complete ? "" : " (incomplete ⏳)")];
}));
await atomicWrite("reports/COMPLETION_QUALITY_REPORT.md", qualityReport);

// ---------------- CLAIMS_TABLE ----------------
const claims = (await pool.request().input("run", sql.VarChar(120), runId).query<{
  research_question: string; evidence_tag: string; taxonomy: string; supported: boolean; power_label: string; rationale: string;
}>("SELECT research_question, evidence_tag, taxonomy, supported, power_label, rationale FROM eval.claims WHERE run_id=@run ORDER BY claim_id")).recordset;
let claimsReport = header("GhostType CLAIMS_TABLE (Tier 1)");
claimsReport += claims.length === 0
  ? "No claims yet: claims are only emitted for profiles with complete (588/588) campaigns.\n"
  : table(["RQ", "tag", "taxonomy", "supported", "power", "rationale"],
      claims.map((c) => [c.research_question, c.evidence_tag, c.taxonomy, c.supported ? "yes" : "no", c.power_label, c.rationale]));
claimsReport += "\nClaim ceiling (addendum §7): simulated acceptance is not observed user acceptance; all quality claims are about replayed frozen episodes on this mac serving stack, DEV-tier SQL timings, and never about live users.\n";
await atomicWrite("reports/CLAIMS_TABLE.md", claimsReport);

// ---------------- STATE_OF_RECORD ----------------
const events = (await pool.request().query<{ stage: string; disposition: string; event_key: string; created_at_utc: Date }>(
  "SELECT stage, disposition, event_key, created_at_utc FROM control.evidence_events ORDER BY event_id")).recordset;
const profiles = (await pool.request().query<{ profile_id: string; port_gate_status: string | null; served_model_id: string }>(
  "SELECT profile_id, port_gate_status, served_model_id FROM control.model_profiles ORDER BY profile_id")).recordset;
let stateReport = header("GhostType STATE_OF_RECORD (Tier 1, mac campaign)");
stateReport += "## Campaign identity\n\n";
stateReport += `- parser: Microsoft.SqlServer.TransactSql.ScriptDom **[170.191.0]** (TSql170Parser)\n- frozen rules: normalize-v1, recompose-v1, candidate-extract-v1\n- transport: raw text, deterministic reference decode (T=0, top_p=1, no stop strings)\n- SQL host: SQL Server 2025 CU8 under Rosetta emulation — every SQL timing is DEV-tier\n- model host: mlx_lm.server :8020 / mlx_vlm.server :8021 (single resident model)\n\n`;
stateReport += "## Model profiles / port gates\n\n";
stateReport += table(["profile", "served model", "port gate"], profiles.map((p) => [p.profile_id, p.served_model_id, p.port_gate_status ?? "not yet run"]));
stateReport += "\n## Evidence event log\n\n";
stateReport += table(["stage", "disposition", "event key", "at (UTC)"],
  events.map((e) => [e.stage, e.disposition, `\`${e.event_key}\``, e.created_at_utc.toISOString()]));
stateReport += "\n## Inherited scars honored\n\nAutocommit DDL for preview features; explicit isolation hygiene; BACPAC restore re-applies preview/compat; unpaired-surrogate policy in insertion integrity; DiskANN INT-key rule reserved for the Tier-2 ANN comparator; statement-relative cursor offsets documented in migration 004.\n";
await atomicWrite("reports/STATE_OF_RECORD.md", stateReport);

// ---------------- SAFETY_REPORT ----------------
const safetyClasses = (await pool.request().query<{ safety_class: string; n: number }>(
  `SELECT JSON_VALUE(oracle_json, '$.safety_class') safety_class, COUNT(*) n
   FROM dataset.cases GROUP BY JSON_VALUE(oracle_json, '$.safety_class') ORDER BY n DESC`)).recordset;
let safetyReport = header("GhostType SAFETY_REPORT (Tier 1)");
safetyReport += "## Safety class routing (package-declared, per case)\n\n";
safetyReport += table(["safety class", "cases"], safetyClasses.map((r) => [r.safety_class, r.n]));
safetyReport += `\n## Execution posture\n\nNo model or baseline candidate has been executed against any database in this campaign. All Tier-1 scoring is static (normalize/constraint/grounding/parse via ScriptDom). Fixture DDL execution and the execution/mutation oracles are dispositioned to a later stage (SOURCE_INTAKE.md); when they run they follow spec §39: rollback-safe transactions, disposable fixture databases, per-fixture throwaway principals, harness-enforced row/time caps, and static adjudication (never execution) for server-level/destructive statements.\n\n## Injection surface\n\nPrompt-injection stress descriptions ship inside package prompts; they are scored like any other rows and can never be cited as ordinary evidence (addendum B-6). No injection-marked content is embedded into retrieval indexes in Tier 1 (no retrieval index has been built from case text).\n`;
await atomicWrite("reports/SAFETY_REPORT.md", safetyReport);

// ---------------- LIMITATIONS ----------------
let limitations = header("GhostType LIMITATIONS (Tier 1)");
limitations += `- **Mac campaign, not the spec's GPU serving plane.** Single-resident MLX serving, sequential replay; no multi-user serving results exist yet (Colab lift-and-shift is the second campaign).
- **SQL timings are DEV-tier** (SQL Server 2025 under Rosetta emulation, outside Microsoft's support boundary).
- **Simulated acceptance is not observed user acceptance** (claim ceiling, addendum §7).
- **Power labels are provisional** until the E-1 power simulation runs; paired-test labels use a discordant-count rule of thumb.
- **8 package defects** are adjudicated findings (3 invalid golds, 5 cursor placements); their rows still score, so exact-match ceilings are slightly depressed for affected families until dataset v2.1.
- **The M-arm matrix is reduced**: the mac campaign replays the package's frozen prompt (arm M1-packaged). M0/M2/R0 prompt constructions and retrieval arms are not yet run.
- **B4's identifier-safe adaptation** is reduced to same-catalog exemplar reuse; cross-catalog adaptation is unimplemented.
- **Execution/compile oracles have not run**; parse and static gates only. compile_eligible/execution_eligible flags are stored and waiting.
`;
await atomicWrite("reports/LIMITATIONS.md", limitations);

// ---------------- REPRODUCIBILITY ----------------
let repro = header("GhostType REPRODUCIBILITY (Tier 1)");
repro += `## Row-only reproduction (Tier-1 closeout gate)

\`\`\`bash
docker compose -f compose.yaml -f compose.mac.yaml up -d   # SQL on :1435
./scripts/repro.sh --mode rows
\`\`\`

Rebuilds every row-level artifact from the vendored, hash-verified package: schema migrations (drift-refusing), dataset import (588 record hashes re-verified in-process), all GT-1 gates, catalog snapshots, the GT-3 parse oracle under the pinned ScriptDom [170.191.0], deterministic baselines B0–B4, aggregate metrics, and the unit suite for the frozen rules.

## Database backup/restore

Stage-boundary backups only (no rolling dumps): \`npm run db:backup\`. Restore re-applies PREVIEW_FEATURES and compatibility level 170 (inherited ModelPrint scar): \`./scripts/export-database.sh restore <bak>\`.

## Model campaign reproduction

Model rows depend on the frozen MLX servings on this machine (registry \`config/models.mac.json\`; per-profile pins recorded by the port gate in \`control.model_profiles\`). Re-running \`npm run quality:run -- --profile <id>\` after wiping that profile's requests reproduces the campaign at temperature 0; exact token-for-token stability is recorded per profile by the port-gate determinism canary.
`;
await atomicWrite("reports/REPRODUCIBILITY.md", repro);

console.log(JSON.stringify({ reports: ["DATASET_REPORT", "COMPLETION_QUALITY_REPORT", "CLAIMS_TABLE", "STATE_OF_RECORD", "SAFETY_REPORT", "LIMITATIONS", "REPRODUCIBILITY"], runId, events: events.length, claims: claims.length }, null, 2));
await pool.close();
