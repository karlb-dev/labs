# STALE_PAPER_PROSE_REGISTER.md — P1 output

Automatic sweep: `scripts/stale_prose_sweep.py` over the 91-source paper
manifest (patterns from plan §4.3 + `PHASE4_TO_PAPER_ANALYSIS_HANDOFF.md`
stale-passage list); raw hits in `stale_prose_hits.json` (29 hits, 14
files). Every hit is dispositioned below. Categories: **UPDATE** (active
paper source, edit in P8 after the P6 route decision), **HISTORICAL**
(frozen-era record correctly describing its own era; superseded by a V2
record; no edit permitted), **CURRENT** (governance text quoting stale
phrases as negations/prohibitions; not stale).

## UPDATE — active paper sources carrying stale claims

| # | Source | Stale claim | Current evidence | Required disposition |
|---|---|---|---|---|
| U1 | `jspace_paper/olmo_lineage.tex:147` | "Branch snapshot … development state through 2026-08-01" | Study 2 closed 2026-08-03; frozen 2026-08-04 | Update snapshot/date to `jspace-phase4-frozen-v1` (plan §11.4) |
| U2 | `jspace_paper/olmo_lineage.tex:199-200` | Bank-S/mechanistic rows imply stage wedge + O5 outstanding | Wedge capability-gated (`ol2-stage-wedge-joint-analysis-v1`: cohorts empty, `null_or_unresolved`); O5 never opened, not-applicable under stop rule | Replace "queued" wording with capability-gated Study-2 result; state O5 not run (§11.4) |
| U3 | `jspace_paper/olmo_lineage.tex:710` | "`testable-with-bounded-stage-wedge`, queued" | Same as U2; H6 additionally licenses no in-band regime (`ol2-transport-validation-joint-v1`) | Replace with terminal wedge + H6 state; include Bank-W pair power 0.7788/16, first pass 18 (`ol2-bank-w-olmo-pair-power-v1`) (§11.4) |
| U4 | `jspace_paper/olmo_lineage.tex` (Qwen/Gemma sections) | Pre-Q-L4 Qwen wording; Gemma comparison at Study-1 blocker state | Q-L4 Terminal C (`p4-qwen-canonical-lens-decision-a1000-dev-v1`); Gemma V2 calibrated license (`gm2-stage1-relicense-v1`) | Update Qwen section to Q-L4/no-canonical-lens; Gemma comparison to calibrated V2 license (§11.4) |
| U5 | `jspace_paper/gemma4_nonlinear_jacobian_handout.tex:222` | "terminal classification is a methods blocker" | Study 2 G2.1/G2.2: pooled ceiling 0.07870368901355948 frozen target-blind; historical all-slot error 0.0024581113830208778 within ceiling; five-layer classifier closed (`gm2-backend-parity-calibration-v1`, `gm2-stage1-relicense-v1`) | Replace terminal-blocker summary with calibrated G2.1/G2.2 sequence; preserve blocker chronology as historical (§11.5) |
| U6 | `jspace_paper/gemma4_nonlinear_jacobian_handout.tex:788` | "stopped at a failed cross-backend parity gate … licensed classification is a methods blocker" (as of 2026-08-02) | Same as U5 | Same as U5; keep the 2026-08-02 sentence as dated chronology, append the Study-2 resolution (§11.5) |
| U7 | `jspace_phase4/reports/handout/jspace_phase4_development.tex` | Whole document predates Q-L4 registration (by design; governed boundary lifted at the tag) | Q-L4 Terminal C + Study-2 import boundary | Regenerate to Terminal C after the P6 route decision (§11.6); regeneration before the route decision is forbidden (handoff) |
| U8 | `jspace_paper/kburtram_jspace.tex:6-8` (header comment) | Points to `interpretability/jspace_runs/analysis/run_analysis.py` | Script lives at `interpretability/jspace_paper/scripts/run_analysis.py`; `jspace_runs` is a Drive tarball, not a repo path | Fix provenance comment in P8; verify every quoted number in the P4 reconstruction audit |

## HISTORICAL — frozen-era records; superseded, not stale; no edit permitted

| Source | Hit | Why retained unchanged |
|---|---|---|
| `jspace_gemma/release/IMPORT_BUNDLE_PHASE4.md:9` | 0.002458 vs 1e-5 | Study-1 import record; correct for its era; V2 supersedes the handoff classification, not the event (GM2-C09) |
| `jspace_gemma/release/gemma_transport_claim_ledger.md` | "licensed methods blocker" | Study-1 ledger; immutable; V2 ledger governs current wording |
| `jspace_gemma/reports/GEMMA_TRANSPORT_DEVELOPMENT_REPORT.md:187` | registered failed gate | Historical registered event `gm-jvp-gemma-backend-parity-v1`; correctly failed its own 1e-5 gate |
| `jspace_olmo_lineage/reports/OLMO_LINEAGE_{CLAIMS_TABLE,STATE_OF_RECORD,DEVELOPMENT_REPORT}.md` | wedge/H6/O5 queued; Bank-W closed-v1 | Study-1 records; V2 state + claims table are the live versions; Study-1 files carry their own "queued is not an outcome" guard |
| `jspace_paper/Mapping_Neural_Curvature.pdf` | (not sweepable — binary) | No tracked TeX source exists in the repo; recorded as `source_absent` in the paper-source manifest; inventory/overlap disposition due at P6 (plan §0.2) |

## Disposition update (P8/P9)

U1–U8 are discharged as follows: U5/U6 (Gemma handout) and U7 (Phase 4
handout) edited in place with chronology preserved; U8 fixed; **U1–U4
(`olmo_lineage.tex`) are superseded by the P6 retirement** — the
document carries a corrective RETIRED banner with the terminal facts
and pointers instead of in-body edits (per
`paper_routes/olmo_lineage/DISPOSITION.md`); its body remains a
historical draft and is not claim-bearing. Two additional Gemma handout
stale sites found by the P9 audit (current-state table row; FAQ
present-tense answer) were date-qualified and resolved in the same
pass.

## CURRENT — governance text; hits are negations or prohibitions

`FREEZE_HANDOFF.md`, `PHASE4_STATE_OF_RECORD.md`,
`PHASE4_KNOWN_LIMITATIONS.md`, `PHASE4_METHODS_DECISION_RECORD.md`,
`PAPER_CONCLUSION_SKELETON.md`, `PHASE4_TO_PAPER_ANALYSIS_HANDOFF.md`,
`jspace_phase4_development.tex:61` ("There is no A2000 route"): every hit
is of the form "no Phase 4 primary", "no A2000", "do not cite a canonical
A1000 lens", or the handoff's own stale-passage list. These are the
current record stating prohibitions; no action.

## Already-current sources (swept clean)

`jspace_olmo_lineage/reports/paper/olmo_lineage_parallel_phase.tex` is
already at the Study-2 terminal state (capability-gated wedge, H6
no-licensed-regime, Bank-W pair closure with evidence IDs) — it needs no
stale-prose correction, only the P4 number audit.
`jspace_gemma/reports/handout/gemma_transport_development.tex` and
`kburtram_jspace.tex` had no pattern hits; both still receive the P4
number audit (pattern absence is not verification).
