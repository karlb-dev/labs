import { writeFileSync } from "node:fs";
import { readFileSync } from "node:fs";
import sql from "mssql";
import { loadConfig } from "../src/config.js";

// Exports the Tier-1 mac-campaign report dataset to
// reports/tier1-report-data.json. The HTML report generator
// (scripts/build-tier1-html-report.ts) renders this JSON into the frozen
// template — rerunning on Colab with that campaign's database regenerates
// the same report shape from its own data.

const PROFILE_ORDER = [
  "muse-glimmer-30b-mlx", "qwen-3.8-27b-mlx", "gemma-4-26b-a4b-nothink-mlx",
  "gemma-4-e4b-mlx", "gemma-4-26b-a4b-mlx", "olmo-3.1-32b-mlx",
];
const SAMPLES_PER_CELL = 4;
const TRIM = (text: string | null, n: number) => {
  if (text === null || text === undefined) return "";
  return text.length > n ? `${text.slice(0, n)}…` : text;
};
const quantile = (values: number[], q: number): number | null => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return Math.round(sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]);
};

const config = loadConfig();
const pool = await new sql.ConnectionPool({ ...config.databases.admin, database: config.databases.controlName }).connect();
const registry = JSON.parse(readFileSync("config/models.mac.json", "utf8"));

const scoreRows = (await pool.request().query<{
  profile_id: string; case_id: string; arm: string; outcome: string; failure_stage: string | null;
  normalized_exact: boolean | null; constraint_pass: boolean | null; grounding_pass: boolean | null;
  parse_pass: boolean | null; latency_ms: number | null; completion_tokens: number | null;
  completion_class: string; expect_empty: boolean; data_role: string; scenario: string;
}>(`SELECT rs.profile_id, rs.case_id, rs.arm, rs.outcome, rs.failure_stage, rs.normalized_exact,
      rs.constraint_pass, rs.grounding_pass, rs.parse_pass, rs.latency_ms, rs.completion_tokens,
      c.completion_class, c.expect_empty, c.data_role, c.scenario
    FROM eval.row_scores rs JOIN dataset.cases c ON c.case_id = rs.case_id`)).recordset;

const telemetry = (await pool.request().query<{ profile_id: string; tps: number | null; reasoning_chars: number | null; case_id: string }>(
  `SELECT q.profile_id, t.tokens_per_second tps, r.reasoning_chars, q.case_id
   FROM telemetry.model_requests t
   JOIN completion.requests q ON q.request_id = t.request_id
   LEFT JOIN completion.raw_model_responses r ON r.request_id = t.request_id`)).recordset;

const gates = (await pool.request().query<{ stage: string; disposition: string; event_key: string; at: Date }>(
  "SELECT stage, disposition, event_key, created_at_utc at FROM control.evidence_events ORDER BY event_id")).recordset;
const portGates = (await pool.request().query<{ profile_id: string; port_gate_status: string | null; served_model_id: string }>(
  "SELECT profile_id, port_gate_status, served_model_id FROM control.model_profiles")).recordset;
const roleCounts = (await pool.request().query<{ data_role: string; n: number }>(
  "SELECT data_role, COUNT(*) n FROM dataset.cases GROUP BY data_role ORDER BY n DESC")).recordset;
const oracleTallies = (await pool.request().query<{ oracle: string; status: string; n: number }>(
  "SELECT oracle, status, COUNT(*) n FROM dataset.case_oracle_status GROUP BY oracle, status")).recordset;
const paired = (await pool.request().query<{ profile_id: string; suite: string; metric_value: number; detail_json: string }>(
  `SELECT profile_id, suite, metric_value, detail_json FROM eval.metric_results
   WHERE suite LIKE 'agg-paired%' AND metric_name='mcnemar_exact_p'`)).recordset;

// ---------- per-profile aggregates ----------
interface ClassStats {
  rows: number; offered: number; exact: number; exactPerOffer: number | null; partial: number;
  grounded: number | null; parseClean: number | null; parseBad: number;
  latencyP50: number | null; latencyP95: number | null;
}
const profiles: Record<string, any> = {};
const isBaseline = (id: string) => id === "baseline-deterministic";
for (const profileId of [...new Set(scoreRows.map((r) => r.profile_id))]) {
  const rows = scoreRows.filter((r) => r.profile_id === profileId);
  const registryProfile = registry.profiles[profileId] ?? null;
  const outcomes: Record<string, number> = {};
  for (const row of rows) outcomes[row.outcome] = (outcomes[row.outcome] ?? 0) + 1;
  const classes: Record<string, ClassStats> = {};
  for (const cls of ["cursor_fragment", "intent_query"]) {
    const subset = rows.filter((r) => r.completion_class === cls && !r.expect_empty);
    const offered = subset.filter((r) => r.outcome !== "abstain_wrong");
    const withParse = subset.filter((r) => r.parse_pass !== null);
    const latencies = subset.map((r) => r.latency_ms).filter((v): v is number => v !== null);
    const exact = subset.filter((r) => r.normalized_exact === true).length;
    classes[cls] = {
      rows: subset.length, offered: offered.length, exact,
      exactPerOffer: offered.length ? exact / offered.length : null,
      partial: subset.filter((r) => r.outcome === "partial").length,
      grounded: subset.filter((r) => r.grounding_pass === true).length,
      parseClean: withParse.filter((r) => r.parse_pass === true).length,
      parseBad: withParse.filter((r) => r.parse_pass === false).length,
      latencyP50: quantile(latencies, 0.5), latencyP95: quantile(latencies, 0.95),
    };
  }
  const traps = rows.filter((r) => r.expect_empty);
  const tel = telemetry.filter((t) => t.profile_id === profileId);
  const reasoningChars = tel.map((t) => t.reasoning_chars).filter((v): v is number => v !== null && v > 0);
  const arm = rows[0]?.arm ?? "";
  profiles[profileId] = {
    profileId, arm,
    servedModelId: registryProfile?.servedModelId ?? portGates.find((p) => p.profile_id === profileId)?.served_model_id ?? profileId,
    quantization: registryProfile?.quantization ?? null,
    portGate: portGates.find((p) => p.profile_id === profileId)?.port_gate_status ?? null,
    decodeLabel: registryProfile?.decodeLabel ?? (isBaseline(profileId) ? "deterministic" : "deterministic-reference"),
    reasoningMode: registryProfile?.reasoningPolicy?.chatTemplateKwargs ? "disabled-by-template"
      : registryProfile?.reasoningAllowanceTokens > 0 ? `on (allowance ${registryProfile.reasoningAllowanceTokens})` : "none",
    totalRows: rows.length, outcomes, classes,
    traps: { total: traps.length, correct: traps.filter((r) => r.outcome === "abstain_correct").length },
    truncations: rows.filter((r) => r.failure_stage === "truncation" || r.failure_stage === "timeout").length,
    latencyP50: quantile(rows.map((r) => r.latency_ms).filter((v): v is number => v !== null), 0.5),
    latencyP95: quantile(rows.map((r) => r.latency_ms).filter((v): v is number => v !== null), 0.95),
    tokensPerSecondMedian: quantile(tel.map((t) => t.tps).filter((v): v is number => v !== null), 0.5),
    reasoningCharsMedian: quantile(reasoningChars, 0.5),
  };
}

// ---------- curated samples with real inputs/outputs ----------
const caseText = new Map((await pool.request().query<{
  case_id: string; doc_prefix: string; doc_suffix: string; canonical_insertion: string;
}>("SELECT case_id, doc_prefix, doc_suffix, canonical_insertion FROM dataset.cases")).recordset
  .map((r) => [r.case_id, r]));
const candidates = new Map((await pool.request().query<{
  request_id: string; extracted_text: string; raw_text: string; format_failure: string | null; suffix_overlap_trimmed: number;
}>("SELECT request_id, extracted_text, raw_text, format_failure, suffix_overlap_trimmed FROM completion.candidates")).recordset
  .map((r) => [r.request_id, r]));
const requestIds = new Map((await pool.request().query<{ request_id: string; case_id: string; profile_id: string }>(
  "SELECT request_id, case_id, profile_id FROM completion.requests")).recordset
  .map((r) => [`${r.profile_id}|${r.case_id}`, r.request_id]));
const reasoningByRequest = new Map(telemetry.filter((t) => t.reasoning_chars !== null)
  .map((t) => [`${t.profile_id}|${t.case_id}`, t.reasoning_chars]));

const samples: any[] = [];
for (const profileId of Object.keys(profiles)) {
  if (isBaseline(profileId)) continue;
  const rows = scoreRows.filter((r) => r.profile_id === profileId);
  for (const outcome of ["success", "partial", "fail", "format_fail", "abstain_wrong", "abstain_correct"]) {
    const pool_ = rows.filter((r) => r.outcome === outcome).sort((a, b) => a.case_id.localeCompare(b.case_id));
    // deterministic spread: take evenly spaced picks across the sorted list
    const picks: typeof pool_ = [];
    for (let i = 0; i < Math.min(SAMPLES_PER_CELL, pool_.length); i += 1) {
      picks.push(pool_[Math.floor((i * pool_.length) / Math.min(SAMPLES_PER_CELL, pool_.length))]);
    }
    for (const row of picks) {
      const text = caseText.get(row.case_id)!;
      const requestId = requestIds.get(`${profileId}|${row.case_id}`);
      const candidate = requestId ? candidates.get(requestId) : undefined;
      samples.push({
        profileId, caseId: row.case_id, outcome, failureStage: row.failure_stage,
        completionClass: row.completion_class, expectEmpty: row.expect_empty,
        scenario: row.scenario, dataRole: row.data_role,
        prefix: TRIM(text.doc_prefix, 360), suffix: TRIM(text.doc_suffix, 140),
        gold: TRIM(text.canonical_insertion, 360),
        model: TRIM(candidate?.extracted_text ?? "", 360),
        rawExcerpt: candidate?.format_failure ? TRIM(candidate.raw_text, 200) : null,
        formatFailure: candidate?.format_failure ?? null,
        suffixTrimmed: candidate?.suffix_overlap_trimmed ?? 0,
        exact: row.normalized_exact === true, grounded: row.grounding_pass,
        parsePass: row.parse_pass, constraintPass: row.constraint_pass,
        latencyMs: row.latency_ms === null ? null : Math.round(row.latency_ms),
        completionTokens: row.completion_tokens,
        reasoningChars: reasoningByRequest.get(`${profileId}|${row.case_id}`) ?? null,
      });
    }
  }
}

const data = {
  schemaVersion: 1,
  generatedAtUtc: new Date().toISOString(),
  lab: "aidataapps/ghosttype",
  campaign: {
    runId: readFileSync(".current-run", "utf8").trim().split("/").at(-1),
    platform: "Karl's MacBook Pro (M4 Max, 48 GB) — mlx_lm.server :8020 / mlx_vlm.server :8021, single resident model",
    sqlHost: "SQL Server 2025 CU8 under Rosetta emulation (all SQL timings DEV-tier)",
    parser: "Microsoft.SqlServer.TransactSql.ScriptDom [170.191.0] (TSql170Parser)",
    frozenRules: ["raw-text transport", "candidate-extract-v1", "normalize-v1 (case-sensitive)", "recompose-v1", "deterministic reference decode T=0 (Muse: DFlash drafter, non-deterministic label)"],
    dataset: "ghosttype_dataset_v2 — 588 cases (430 intent_query, 158 cursor_fragment; 101 expect-empty traps), 14 catalogs, hash-verified",
  },
  profileOrder: [...PROFILE_ORDER.filter((p) => profiles[p]), ...Object.keys(profiles).filter((p) => !PROFILE_ORDER.includes(p) && !isBaseline(p))],
  profiles,
  baselines: profiles["baseline-deterministic"] ? (() => {
    const rows = scoreRows.filter((r) => r.profile_id === "baseline-deterministic");
    const byArm: Record<string, Record<string, number>> = {};
    for (const row of rows) {
      byArm[row.arm] = byArm[row.arm] ?? {};
      byArm[row.arm][row.outcome] = (byArm[row.arm][row.outcome] ?? 0) + 1;
    }
    return byArm;
  })() : {},
  paired: paired.map((r) => ({ profileId: r.profile_id, versus: r.suite.replace("agg-paired-vs-", ""), p: r.metric_value, ...JSON.parse(r.detail_json ?? "{}") })),
  dataset: {
    roles: roleCounts, oracles: oracleTallies,
    defects: [
      "2 split groups spanned roles (governed to most-held-out role; dispositions in config/dataset-dispositions.json)",
      "3 golds are syntactically invalid T-SQL (gt-cur-case-03/-04 simple-vs-searched CASE; gt-edge-proc-05 EXEC expression argument)",
      "5 cursor-placement defects: whole statement emitted as prefix, gold lands after the terminator (gt-sparse-alias-01..04, gt-edge-cont-05)",
    ],
  },
  gates: gates.map((g) => ({ stage: g.stage, disposition: g.disposition, key: g.event_key, at: g.at.toISOString() })),
  notes: [
    "H1 (beat the grammar-aware deterministic baseline on overall normalized-exact) is unsupported for every profile: the headline metric bundles abstention discipline with generation quality — per-offer and per-class suites carry the real signal.",
    "Gemma 26B measured in BOTH modes: thinking bought no answer quality (32 vs 35 cursor exact) at 20-30x latency and an 18% unbounded-reasoning pathology, but did buy abstention discipline (54 vs 13 traps correct).",
    "Muse: quality/trust leader (42.9% cursor exact-per-offer, 92% traps, 0 parse errors) at 6x median latency; over-abstains ~30% of answerable rows. No working thinking-off switch in mlx_vlm 0.6.15.",
    "Gemma 26B nothink is the only profile in plausible ghost-text latency range on this hardware (p50 ~1.5s cursor / ~2.5s intent).",
  ],
  samples,
};
writeFileSync("reports/tier1-report-data.json", `${JSON.stringify(data, null, 1)}\n`);
console.log(JSON.stringify({ profiles: Object.keys(profiles).length, samples: samples.length, bytes: JSON.stringify(data).length }));
await pool.close();
