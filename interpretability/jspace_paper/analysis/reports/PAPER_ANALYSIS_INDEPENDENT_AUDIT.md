# PAPER_ANALYSIS_INDEPENDENT_AUDIT.md — P9 independent audit

Auditor: fresh, independent session (not involved in producing any audited
document). Date: 2026-08-05. Branch `interp_jspace_paper_analysis` at the
paper-analysis working tree; campaign registries and Drive run roots
accessed **read-only**. Scope per `protocol/PAPER_ANALYSIS_PROTOCOL.md`
P9 and the plan §12: claim audit, contradiction audit, reviewer-objection
matrix, and (separately reported) the reproduction audit
(`PAPER_REPRODUCTION_AUDIT.md`).

Method: every quantitative/causal sentence in the two route claim tables,
two outlines, six abstract candidates, gemma-handout update plan, OLMo
disposition, the ten analysis reports, and the four updated TeX sources
was traced to (a) a live evidence id in `data/master_evidence_live.parquet`
(324 ids, 260 live, re-parsed this audit), (b) the frozen states of record
(`jspace_phase4/reports/PHASE4_STATE_OF_RECORD.md`, `release/PHASE4_CLAIM_LEDGER.md`,
`jspace_phase3/reports/PHASE3_STATE_OF_RECORD.md` + `REPORT_PHASE3.md`,
`jspace_gemma/release/GEMMA_TRANSPORT_STATE_OF_RECORD_V2.md` + claim ledgers,
`jspace_olmo_lineage/reports/*`), and (c) the verified tables
`analysis/tables/recon_*.csv` and the five synthesis tables. Frozen Drive
payloads (`span_audit_cross_model_stats.json`, `prose_grid_figure_stats.json`,
`confirmatory_analysis.json`, `replication_analysis.json`,
`n8_level1_comparison.json`, per-item parquets) were opened read-only to
resolve ambiguous ranges.

## Verdicts

| Audit section | Verdict |
|---|---|
| 1. Claim audit | **PASS-with-findings** (F1, F2, F4–F6, F8–F10) |
| 2. Contradiction audit | **PASS-with-findings** (F3, F7) |
| 3. Reviewer-objection matrix | **PASS-with-findings** (7 of 8 closed with evidence ids; O4 open-as-limitation, correctly disclosed) |
| 4. Reproduction audit | **PASS** (see `PAPER_REPRODUCTION_AUDIT.md`; 8/8 scripts green, zero real diffs) |

What was checked and found exact (non-exhaustive): P-HP1 −0.5045
[−0.7195, −0.2949] p_holm=0.0005 and non-replication +0.1036 [−1.6813,
+1.8892] p=0.7075 on the disclosed 322/32 fallback; P-HP3 +0.2788
[0.2048, 0.3608] p_one_sided=0.00025 (Holm 0.0005) → +0.2966 [0.2071,
0.3824]; P3-P2 +0.095833 (188/26) and +0.102083 (190/28), plus-one
p=1/100001 each; P3-P3 +0.431367 [0.132018, 0.763437] p=0.009180
(Holm 0.018 per the frozen Holm-family table), residual +0.403816
p=0.01854, cross-fit R²=−0.0947; substitution +8.582031 p=0.000488 and
+1.342254 [−1.593275, +4.482051] p=0.419; P3-P1 −0.271183 p=0.057892
(sha256-v1) / −0.260954 p=0.062332 (historical), randomization CI
[−0.535135, +0.014565], percentile-t [−0.537109, +0.015201], normal
[−0.515271, −0.006637]; span audit 0.89/0.94/1.37, 37/42/28%,
18/21/26%, r 0.1951/0.2915/0.7452; mediation −0.89/−4.05/−0.04 (Qwen)
and −0.09 / −0.20 [−0.38, −0.04] (Think); overlap mining −0.30/−0.31/+0.24,
blocked rates 1.33/1.22/0.10 ("12× model property" is verbatim frozen),
975 items; prose 0.0018–0.0206 vs 0.1754–0.9452, std −2.11/−2.14/−1.51
vs −0.55/−0.64/−0.26, reductions 49.3/71.6/77.8%; G4 0.76/0.76/0.78;
lens-independence r 0.9882/0.99, tail-Jaccard 0.878/0.883; occupancy
Qwen 5.0–6.1% vs OLMo 0.44–1.23% (occ_med 2/2 vs 3/3/4); Qwen ladder
q50 0.99522→0.99771→0.99870, q05 0.99308→0.99684→0.99812, Jaccard
exactly 7/13 ×3, projector 0.67479→0.70302→0.70982, rescue −0.2147 /
+0.5589 / −0.2940, margins 17,381 = 15,536+1,845+0, materiality closest
3,856.83×, norms 181.776618/181.777423 and 181.826310/181.785516 vs
173.345, prompt-112 55.544060/55.587600 vs 160.070954; OLMo lineage
effects table (all eight estimates + CIs), movement 0.32556/0.00604,
to-Base q50 0.9614/0.9548/0.9523, mapped 0.6744, raw unembedding 0.9853,
selected-ID 0.3333, projector 0.334/0.322/0.265, excess-equivalence
1.54e-4; wedge 972-row, 0.617%/0.309%, Bank-S 0.833%/0.278%, zero
capable facts, route null_or_unresolved; Bank-W 16 shared capable,
0.7788@16, 18 first passing, SESOI 0.15849625 nats (=0.10 nat/doubling
× log2 3); H6 Base 9/336 (L56@0.10 = 9/12), Think one licensed cell
L56@0.10 12/12, 672 rows under ceiling (max 0.022162/0.025295); Gemma
0/12 ×5 layers, TRE 1.3569/1.5730/3.5250/5.3647/4.3856, cosines
0.0389/−0.0622, gains 1.0–5.5, ceiling 0.07870368901355948 =
3·q99(0.026234563…), all-slot 0.002458111383020878, route
benign_scheduling_floor, OLMo control 0.111; durability 521→520
(515+3+2), deficit sha 361bda08…, suites 302/71/90, review 13/14 PASS +
1 PASS-with-limitation, PI 13 items accepted; payload SHAs 5df4ace5…/
f5367e5a… confirmed in the frozen envelopes; release-manifest pin
7cf71215… confirmed. Tier assignments in both claim tables match the
ledger and the frozen tier rules; **no tier promotion was found
anywhere** (one narrowing: the seed-sentence "lesion/preference"
confirmatory scope is carried at development tier in C3/Paper A — a
permitted narrowing). U1–U4 register corrections are applied in
`kburtram_jspace.tex`. The forbidden-wording sweep (near-miss,
convergence-rate, canonical-lens, "Bank W negative", "all artifacts
verify", SFT-installs/DPO-removes, workspace assertion, unqualified
"transport fails", Instruct-as-successor, gated-as-null) found **zero**
violations in the analysis reports and route documents — every hit is a
prohibition or guard.

## Findings (numbered; every discrepancy found)

**F1 — MODERATE — `jspace_paper/kburtram_jspace.tex` §Reproducibility
(lines 353–358): wrong / withdrawn N8 evidence ids.** The passage frames
the Phase-3 release ladder but cites the *historical N8-P2* events:
(a) `\eid{p3-n8-level2-*-v1}` — the Qwen member
`p3-n8-level2-qwen36-27b-v1` is **withdrawn** in the registry (Phase-3
SoR correction #6: output path reused/overwritten); the live Phase-3 L2
cells are `p3-n8-p3-level2-{olmo31-think,olmo31-instruct}-v1` and
`p3-n8-p3-level2-qwen36-27b-v2`. (b) The "one full 188-item Qwen GPU
cell bit-exact" claim cites `p3-n8-level3-qwen36-27b-v1`, which is the
**184-item** N8-P2 cell; the 188-item Phase-3 cell is
`p3-n8-p3-level3-qwen36-27b-v1` (the id the Paper-A claim table uses
correctly). (c) "Level 1 … (≤5×10⁻⁴ on all four primaries)" cites
`p3-n8-level1-repro-v1` — that is the Phase-2 primary-table
reproduction (worst absdiff 0.0005 ✓, so the number is right), while
the Phase-3 L1 is `p3-n8-p3-level1-v1` (61/61 exact, tol 1e-10).
Required fix: cite the `p3-n8-p3-*` ids for the Phase-3 ladder and
separate the Phase-2-era reproduction sentence; never cite the
withdrawn Qwen L2 v1.

**F2 — MODERATE — "12–52×" backend-scale multiple is unsourced and not
derivable from any frozen scale.** Locations:
`reports/TRANSPORT_APPLICABILITY_SYNTHESIS.md` ("12–52× the calibrated
backend scales"), methods OUTLINE §9, methods CLAIM_TABLE **B9**
("12–52× above **every** calibrated backend scale"), methods
ABSTRACT_CANDIDATES A, FIGURE_TABLE_PLAN M4, the transport figure title
(`build_transport_synthesis.py:105`), and — most importantly — the
updated passage of `gemma4_nonlinear_jacobian_handout.tex` (line 803).
No frozen Gemma record states 12–52×. From the frozen values (TRE
1.3569–5.3647): vs frozen ceiling 0.0787 → **17.2–68.2×** (the
synthesis's own "17–68×" sentence is correct); vs pooled q99 0.0262 →
51.7–204.5×; vs ten-quantum q99 0.0653 → 20.8–82.1×; vs the OLMo
control 0.111 → 12.2–48.3×. No denominator yields [12, 52]; B9's
"above every calibrated backend scale" is wrong at both ends for at
least one scale. Required fix: replace with "17–68× above the frozen
calibrated ceiling (larger against the smaller backend scales)" or
per-scale exact ranges, in all six locations, and rebuild the handout
PDF.

**F3 — MODERATE — `gemma4_nonlinear_jacobian_handout.tex` line 1212
(§"Are we sure the Gemma result is real?") gives the pre-Study-2
blocker as the present-tense answer**: "…quarantines itself behind a
failed cross-backend parity ceiling. So the honest answer is … still
not licensed as a closed mechanism result, and **now blocked on backend
consistency** rather than on missing evidence." Post-Study-2 the result
is a *closed exact-JVP finite-scale methods result* (mechanism still
unidentified — that half is fine); "now blocked on backend consistency"
contradicts the two update passages (contradiction-audit item: Gemma
blocker vs relicense). Similarly line 797's evidence-table row ("a
failed backend-parity gate quarantines the result as diagnostic-only")
sits in a current-state table ("What is already persuasive, and what
remains") without a date qualifier. Required fix: date-qualify both
passages ("as of 2026-08-02 …") and append the Study-2 resolution
clause; rebuild the PDF.

**F4 — MINOR — `reports/A2_CAUSAL_CHANNEL_CORE.md` SQ2 #1: "tail rates
are 0.13–0.32 across models"** — the frozen span-audit span-safe J-arm
tail rates are 0.15 (Qwen) / 0.20 (Think) / 0.3167 (Instruct); 0.13 is
Qwen's *specific/excess* tail (0.1333), not a rate.
`A5_H6_PHASE3_RECONCILIATION.md` quotes the correct 0.15–0.32 from the
same audit — an internal inconsistency between two reports citing one
registered source. Fix A2 to 0.15–0.32 (or label 0.13 as the excess).

**F5 — MINOR — A2 SQ2 #3 over-generalizes two claims.** (a) "Ruled-out
control families (each ≈ 0 on the same items): … protected-energy-
matched …" — unscoped; the frozen span audit gives prot-energy-matched
on Think −0.43 [−0.80, −0.08] (CI-clean) with tail 18% (the registered
leakage-dose component); "≈ 0" holds on Qwen (−0.02) and roughly on
Instruct (−0.15, straddles 0). Also persistent-matched Qwen tail is
0.05 and Instruct own-control 0.033, straining "every matched-control
arm sits at 0.00–0.02" (true only for the exact instant rank+energy
arm: 0.000–0.0167). (b) "the Phase 2 label-protected effect was ~3× the
span-safe effect **on the OLMo pair**" — the registered ≈3× shrinkage
is Qwen's (HP3 0.2788 → P3-P2 0.0958 = 2.91×; the abstract's "one
third" is correct); the OLMo-pair span-audit mean ratios are ≈5.6×
(Think) and ≈3.4× (Instruct), and the registered OLMo-side fact is
*reallocation* (r=0.195/0.292), not a clean ratio. Fix: scope both
sentences per model.

**F6 — MINOR — rounded values stated as hard bounds that the exact
values violate.** `QWEN_INSTRUMENT_SYNTHESIS.md`: "centered-excess
≤ 0.071 pp" (actual max 0.07110238 pp) and "tail-rate ≤ 0.033" (actual
0.033333); methods OUTLINE §6: "all materiality **≥3,857×** under
threshold" (actual closest ratio 3,856.83× — B6's "3,857×" as a point
value is fine, the "≥" is not); methods ABSTRACT A: "never exceed
Jaccard 0.538" (actual 0.538462 = 7/13, stated correctly in the same
sentence). All sub-1% rounding artifacts, but the figure-ledger's own
rule is "every float quotes the stored full-precision repr". Fix: use
"≈" or exact reprs.

**F7 — MINOR — retired `jspace_paper/olmo_lineage.tex` body retains
pre-Study-2 wording; stale register not annotated.** The body still
says "queued" for the wedge/O5/Bank-W threads (lines 218, 728, 786,
1047, 1070) and carries pre-Q-L4 Qwen / Study-1 Gemma sections — the
exact "wedge queued vs capability-gated" contradiction — mitigated by
the prominent retirement banner (lines 144–159) which states the
corrected facts (capability-gated wedge 0.617%/0.309%, H6 no in-band,
Bank-W 0.7788@16 first-pass 18) and by `DISPOSITION.md` ("no deep edit
performed on a retired document"). However
`STALE_PAPER_PROSE_REGISTER.md` U2–U4 still order in-body replacement
per §11.4 and were never annotated as superseded by the P6 retirement.
Fix: add one register line recording the banner-only disposition.

**F8 — MINOR — `kburtram_jspace.tex` §Phase2 TODO occupancy wording.**
"the OLMo 3.1 pair flat at ~1.2%" — the registered centered excess runs
0.44% → 0.94–0.98% → 1.23% across L24/L32/L40 (flat across the *pair*,
not across layers; ≈1.2% is the L40 value, band median ≈1.0%). And
"(lower edge of the reported Claude band)" asserts the cross-paper
occupancy comparison that the errata left open (W5: "'same harness' was
not same; the paper comparison itself remains open"). Fix: per-layer
values or band median; drop or explicitly caveat the Claude-band
comparison.

**F9 — NIT — `kburtram_jspace.tex` header + one pseudo-evidence id.**
Header comment names only `jspace_phase3/reports/evidence_events.jsonl`
as "the registry", but the draft cites part2-registry ids (`n6-*`,
`r2-*`, `r5-*`, `mc-dev-*`). `\eid{AMENDMENT\_1\_BOS\_UNITS}` is a
preregistration document (`jspace_part2/preregistration/
AMENDMENT_1_BOS_UNITS.md`), not a registry event — it fails the
"evidence id exists and is live" check by construction. Fix: name both
registries; cite the amendment as a document reference, not an `\eid`.

**F10 — NIT — two float reprs of the Gemma all-slot error circulate.**
`CLAIM_SURVIVAL_LEDGER.md` C5 and `PHASE4_STATE_OF_RECORD.md` use the
17-digit render `0.0024581113830208778` (copied from the frozen V2 SoR,
so compliant with "copied from the frozen states of record"); the
updated TeX passages use the stored 16-digit repr
`0.002458111383020878` (compliant with the register's stored-repr
rule). Same float64; the render-diff is already documented in
`UNSUPPORTED_NUMBER_REGISTER.md`. No action strictly required; noted so
reviewers do not read the textual difference as a numeric one.

**F11 — OBSERVATION — two registered prompt-323 max norms.** The
runtime-identity block value 181.826310 (C7, ledger) and the influence
event's current-runtime recompute 181.776618 (QWEN synthesis, Phase-4
handout) are different registered quantities from different events;
both reconstruct byte-identically. Each document labels its own value
correctly; when both appear near each other in Paper B, add one clause
distinguishing the runtime-block measurement from the pinned influence
recompute.

**F12 — OBSERVATION — Paper A draft coverage vs outline.** The updated
`kburtram_jspace.tex` contains outline sections 1–4 and 7–9 analogues
but not yet the outlined §5 (OLMo lineage, Figures E4/E5), §6
(capacity), or the §7 boundary block (Q-L4 summary, transport gate,
externalization 16/20 + 0.7788@16); several sections remain TODO
blocks. Not a violation — P8 delivers outlines plus a seeded draft, and
every number present is verified — but this P9 pass covers only the
prose that exists; the lineage/boundary sections must re-enter audit
when drafted.

## Reviewer-objection matrix (audit item 3)

| # | Objection | Documented answer (location + evidence ids) | Status |
|---|---|---|---|
| O1 | "You deleted the output, not the thought" (output deletion) | A2 SQ2 #3 (label protection + span-safety by construction); kburtram §IV leak audit; C1 ledger row. `p3-span-audit-cross-model-v1`, `p3-protocol-audit-protected-answer-qwen-v1`, `n6-confirmatory-analysis-v2` | **CLOSED** |
| O2 | "Any rank/energy-matched removal does this" (dose-rank) | C1 wording ("matched control equates rank and removed energy by construction"); A2 core table; A5 estimand 2 (control tails ≈ 0). `mc-dev-validation-olmo31-think-v2` (MC1–MC4), `p3-inference-audit-v1` | **CLOSED** |
| O3 | "Artifact of one fitted lens" (one-fit lens) | A2 SQ2 #2: independent-lens replication r=0.9882/0.99, tail-Jaccard 0.878/0.883 (`n6-repl-lens-independence-v2`); Q-L4 boundary explicitly scoped to *newly fitted sparse* lenses (C6, A12/B-rows) | **CLOSED** (scoped: measured on the OLMo pair; Qwen disclosure "where measured" present) |
| O4 | "Narrow bank / external generalization" | Acknowledged, not closed: A2 SQ2 #4 lists generalization beyond the frozen banks as still-open; within-campaign breadth answered by held-out-family replication (`n6-replication-analysis-v2`, P3-P2 replication) and the thin two-hop leg + two-model-family limits are disclosed (outline §9, abstract audit notes) | **OPEN — correctly carried as a limitation** |
| O5 | "You claim training causes the OLMo pattern" | Claim never made: C2 forbidden upgrades; OLMO synthesis boundaries 1 & 5 (ancestry-qualified, natural graph, wedge capability-gated). `ol2-stage-wedge-joint-analysis-v1`, `ol2-checkpoint-ancestry-v1` | **CLOSED** |
| O6 | "Gemma disproves the method" | C5 licensed wording (closed finite-scale methods result at tested scope; no nondifferentiability/workspace/mechanism claim); TRANSPORT synthesis (OLMo control passes the same harness; calibration rescued the measurement, not the premise); handout "what the result does not establish". `gm-jvp-gemma-stage1-v1`, `gm2-backend-parity-calibration-v1`, `gm2-stage1-relicense-v1`, `gm-jvp-olmo-positive-control-v1` | **CLOSED** (but see F2/F3 for wording defects in two carriers) |
| O7 | "Your own H6 invalidates the causal effects" | `A5_H6_PHASE3_RECONCILIATION.md` — registered `analysis-h6-phase3-reconciliation-v1` (predeclared, written before drafting); B11/A11 carry the one-sentence form. `ol2-transport-validation-{base,olmo31-think,joint}-v1` | **CLOSED** |
| O8 | "No Phase-4 primary means the campaign failed" | Phase-4 claim ledger forbidden list ("Phase 4 found no effect" / "was confirmatory"); C6 (Q-L4 as the methods contribution); A7 data-state doctrine (P4-P1/P2/P3 dispositions are states, not failures); route decision rubric. `p4-qwen-canonical-lens-decision-a1000-dev-v1` | **CLOSED** |

No objection lacks a documented answer; O4 is the one deliberately left
open, and every audited document that touches it states it as a
limitation rather than a closure — no fix required beyond keeping that
disclosure in both papers.

## Disposition

The paper-analysis corpus survives an adversarial number-level audit:
of the several hundred quantitative statements traced, every one not
listed above matches its frozen source exactly (or by documented
render-diff). The three MODERATE findings (F1 wrong/withdrawn N8 ids,
F2 the "12–52×" multiple, F3 the handout's residual present-tense
blocker) are each localized, mechanical fixes that must land before any
public artifact; none undermines a headline claim, a tier assignment,
or the route decision. P9 verdict: **PASS-with-findings**, conditional
on F1–F3 being fixed at (or before) the next P8 edit pass.
