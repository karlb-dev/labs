# PREFERENCE_PHASE1_STATE_OF_RECORD.md

Phase 1 of the preference campaign (Lab 38) is **complete**. Branch
`interp_preference_phase1`; freeze tag `preference-phase1-freeze-v1`;
registry `reports/evidence_events.jsonl` (append-only; every claim below
carries its event). One VM session, 2026-08-07, RTX PRO 6000 Blackwell.

## The result in three sentences

The instrument is valid and the positive-control pipeline passed
perfectly on both models; the null-control family sat at exactly zero, so
the pipeline does not manufacture effects. Zero of twelve arbitrary
scenarios graduated the frozen ten-criterion rule — the preregistered
Stop B outcome — because a pervasive first-position selection policy
(total where content is interchangeable, 0.113–0.5 everywhere) never
clears the nuisance-purity bar, even though four scenarios show real
content-tracking asymmetries that override position (install-first
−0.388, batch-ingest −0.363, batch-migration −0.227, testfix −0.125).
Matched report-only twins sit near indifference while enacted choice is
asymmetric — a stated/revealed *behavioral dissociation* under this
battery — and no mechanism, coupling, or latent claim exists at any tier
because the causal block was correctly never licensed.

## Result ladder adjudication (plan §0.4)

| rung | verdict |
|---|---|
| R0 instrument valid | **PASS** — audits, parity, determinism, resume byte-parity, 0 wrong branches (`pref1-gate-g2-instrument-v1`, `pref1-frozen-behavioral-7b-v1`) |
| R1 PC survives counterbalance | **PASS, perfectly** — 480/480 expected content, every stratum (`pref1-graduation-manifest-v1`) |
| R2 arbitrary asymmetry survives counterbalance | **NOT GRADUATED (Stop B)** — 4 descriptive content asymmetries beyond SESOI, each blocked by frozen criteria (chiefly nuisance purity) |
| R3 choice→report coupling | **NOT LICENSED, NOT RUN** — sealed mechanism module + captures await a future preregistered phase |

## Frozen 7B battery (primary; `…-20260807_210537-9df027`)

2,320/2,320 rows; strict parse 0.9953 (11 invalids, all the identical
`PK4` code-blend specimen at content-vs-position conflict cells); NC
floor p95 0.1125 with NC at exactly 0.000; consequence framing mostly
inert (max |Δ| 0.161, median 0.006); AR↔RO matched-cell agreement 0.678
with RO rates 0.425–0.500 vs asymmetric enacted rates; microtask
follow-through 92.5% (exploratory). Full analysis:
`PREFERENCE_PHASE1_BEHAVIORAL_REPORT.md`; per-item rows:
`frozen_7b/results.jsonl`.

## 32B replication (`…-20260807_211808-5f68cb`)

32B-PENDING

## DG secondary smoke (development tier)

Forced STOP 3/3 after stalled false-fact loops vs 0/2 cooperative;
stalled-meta forks 2/2 productive redirect; scaffolded `DISENGAGE:` used
immediately when installed; one free-form flag routed to human review
(licenses nothing). `pref1-dg-smoke-v1`.

## Instrument findings of record (reusable beyond this lab)

1. **bf16 batched teacher-forced margins are not batch-invariant** on
   either model (max deltas ≈0.250 nats, 7B and 32B alike); single-row
   scoring is exact and resume-invariant. Any margin-based lab should
   check this.
2. **First-position selection policy**: total (+0.500) under
   content-indifference on the 7B, monotonically reduced by content pull.
3. **NC identical-option scenarios** are a cheap, decisive
   pipeline-falsification instrument — they caught the dev-slice
   position aliasing within one pilot.

## Deviations, drops, and honesty items

DEVIATIONS.md D1–D7 (PI-ratified). Dropped per §15.4 with PI approval:
proprietary DG PC, full DG battery, optional lineage, temperature
robustness. Equality review remains `agent_dual_code_provisional`
(authorship-limited blinding; PI accepted for Phase 1; PI/panel required
for publication-grade claims). Worst-case invalid-row bounds change no
adjudication. All pre-freeze model outputs are labeled development.

## Open questions (not claims)

- Does any content asymmetry survive a menu format that suppresses the
  first-position policy? (Highest-value next experiment; sealed captures
  + preregistered mechanism module make Phase 2 immediately runnable.)
- Is the `PK4` code-blend specimen a stable conflict signature or a
  codebook-specific artifact? (Alternate-codebook sensitivity would say.)
- Does the RO-channel indifference persist under non-binary report
  formats?

## Where everything lives

| object | path |
|---|---|
| registry (append-only) | `reports/evidence_events.jsonl` — 16 live events |
| behavioral report | `reports/PREFERENCE_PHASE1_BEHAVIORAL_REPORT.md` (+ STATE.json) |
| frozen 7B evidence | `reports/frozen_7b/` (tables, figures, diagnostics, results.jsonl) |
| 32B evidence | `reports/frozen_32b/` |
| DG evidence | `reports/dg_smoke/` |
| handouts (TeX+PDF) | `reports/handout/` (development + frozen) |
| graduation manifest | `reports/graduated_scenarios.json` (empty; Stop B) |
| sealed captures | Drive `phase1/part1/runs/…9df027/state/decision_residuals.pt` (185 MB, hash-pinned) |
| validation | `interpretability/validation/lab38/VALIDATION.md` |
| handoff | `reports/PREFERENCE_PHASE1_HANDOFF.md` |
