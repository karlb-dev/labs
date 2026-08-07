# PREFERENCE_PHASE1_FREEZE_REVIEW.md — the single human gate

**Status: AWAITING PI. This packet stops here — it is a review handoff
with a red freeze gate, not a freeze authorization.** `behavioral_frozen`
is mechanically refused until `preregistration/PREFERENCE_PHASE1_FREEZE_RECORD.md`
exists.

## What the PI is being asked to do

1. **Rate the equality sheets** (or accept the provisional agent passes
   for Phase 1 with the limitation carried): blinded sheets at
   `data/lab38_human_equality_review.csv` (200 rows, 2 agent passes,
   disagreements preserved; card documents the authorship-limited
   blinding). Plan §3.9 requires author ratings before the frozen battery.
2. **Approve or adjust, then freeze, the preregistration candidate**
   (`preregistration/PREFERENCE_PHASE1_PREREGISTRATION_CANDIDATE.md`) —
   notably: SESOI 0.10; the ten G2 graduation criteria; NC-floor
   criterion; single-row margin scoring; the E14 causal endpoints; the
   drop order. After approval these numbers are frozen.
3. **Ratify the deviations** `preregistration/DEVIATIONS.md` D1–D7
   (layout, harness, compressed equality-pass separation, waived v1
   reproduction, deferred DG/mechanism modules, dev-subset v2).
4. On approval: the agent commits the freeze record, derives the seed
   from the freeze commit, tags `preference-phase1-freeze-v1`, and runs
   the frozen 7B battery (~9 GPU-minutes measured projection) plus
   activation capture; then analysis under the frozen rules only.

## Exact model-derived data seen before this review (completeness of disclosure)

| data | scope | seen |
|---|---|---|
| neutral-prior codebook probe | 12 code strings under one neutral context (no menu content) | yes (instrument calibration) |
| Tier-A SmolLM2 smoke + resume proof | 15-row covering set ×3 runs | yes (plumbing; 0% valid parses) |
| 7B 32/128-row benchmarks | canonical-order prefixes of the dev subset | yes (runtime + parse sanity) |
| 7B dev pilot v1 | 348 rows: train incidentals, order 0 only | yes (exposed position aliasing; retained as tie-break evidence) |
| 7B dev pilot v2 | 696 rows: train incidentals, both orders, letter labels | yes (the development report) |
| validation/holdout incidentals | any model output | **never opened** |
| number-label cells | any model output | **never opened** |
| frozen-stage analysis (graduation, aggregates) | — | **never run as frozen** |

No AR wording was tuned to enlarge any asymmetry; the only repairs after
model contact were the dev-subset widening (D7 v2) and the single-row
scoring decision, both instrument-level and both recorded.

## Open design choices the freeze closes

- The G2 constants (SESOI, LOIO floor, nuisance max, invalid-diff max,
  margin-variance gate) — defaults set per addendum, PI may adjust *now*,
  frozen after.
- Whether the 32B replication runs at all this part (drop-order item 4;
  hardware qualifies: 96 GB).
- Whether PC-SAFETY rows stay in the headline PC aggregate or report
  separately-only (current: separately reported AND inside the aggregate;
  both shown).
- Whether pilot-v1's retained rows may be cited as tie-break evidence in
  the final report (current stance: yes, clearly labeled).

## Risks the reviewer should weigh

- Equality blinding is authorship-limited (the agent authored and rated);
  PI ratings cure this for the frozen license.
- Dev estimates came from train incidentals only; frozen effects may
  shrink (more incidentals, number labels, tighter NC floor). That is the
  design working, not a defect.
- The first-position tie-break is so strong that scenarios without
  content pull will produce clean nulls; a mostly-null frozen result is a
  publishable Phase 1 outcome by plan §0.4.

## Recommendation (agent, non-binding)

Approve with the candidate constants unchanged; accept provisional
equality ratings for the 7B frozen run with the limitation carried in
every report (PI panel before any publication claim); keep the 32B
decision until after the 7B frozen result.
