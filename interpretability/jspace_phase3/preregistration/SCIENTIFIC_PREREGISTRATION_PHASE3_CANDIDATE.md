# SCIENTIFIC PREREGISTRATION — Phase 3 (CANDIDATE — NOT FROZEN)

**Status: CANDIDATE.** This document becomes binding only at the freeze
commit (rename to `SCIENTIFIC_PREREGISTRATION_PHASE3.md`, partition
generated with `freeze_authorised=True`, tag
`jspace-phase3-freeze-v1`), after every §9 gate below closes cleanly.
Conditional PI sign-off for that freeze was recorded 2026-07-29
(PHASE3_PLAN_ACCEPTED.md); it is void if any gate fails.

Governance: `jspace_lab_nextsteps_3_1.md` + addendum (§4 amendments and
§5 resolutions R1–R7 govern). Phase 2 (`SCIENTIFIC_PREREGISTRATION.md`
+ Amendment 1) stays frozen; nothing here reruns or reinterprets it.

## 0 · Design summary

Three-model confirmatory study on the thick paired composition bank:
OLMo 3.1 32B Think (`832c3f54…`), OLMo 3.1 32B Instruct (`ac0587e4…`),
Qwen 3.6 27B (`6a9e13bd…`), each under its Phase 2 primary lens (own
lenses for the OLMo pair, the published n=1000 lens for Qwen; fit-size
asymmetry carried as the preregistered sensitivity, symmetric fit-size
study in the strong set).

Banks: **Bank F v6** (`p3-bank-f-v6`: 204 bundles / 48 canonical
families, every bundle direct + composed + bridge_supplied + rotated
counterfactual, per-revid source-verified) and **Bank S v3**
(`p3-bank-s-v3`: 120 bundles / 24 synthetic template families). Bank W
(load) is Workstream E, Family B, separately gated.

Assay: the Phase 3 ablator in **span-safe** output protection
(`meanJ_span_safe` — the §4.1b development audits showed label
protection leaks 28–42% of removed energy into the protected span on
every model), with the label-protected arm retained as a *described*
secondary for continuity with Phase 2 and the paper. Units: assay-wide
Amendment-1 conventions (BOS where the tokenizer has one — recorded
per run as `bos_prefixed` — piecewise un-rstripped concatenation);
Phase 3 banks are authored under `DEFAULT_SPEC` (trailing whitespace
rejected at authoring).

## 1 · Primary hypothesis family (Family A, Holm-corrected)

Estimand chains per §5.7/§14.1, all family-weighted with
family-clustered inference:

```
J_effect_v      = lp_spanSafeJ(v) − lp_base(v)          v ∈ {direct, composed}
C_effect_v      = lp_control(v)   − lp_base(v)
specific_v      = J_effect_v − C_effect_v
within_fact_comp = specific_composed − specific_direct
model_diff(A,B)  = within_fact_comp_A − within_fact_comp_B
```

- **P3-P1 (thick task contrast):**
  `within_fact_comp_Qwen − mean(within_fact_comp_Think, within_fact_comp_Instruct)`
  on Bank F confirmatory families. Test: family sign-flip randomization
  (≥100{,}000 flips), two-sided. H0: 0.
- **P3-P2 (span-safe specificity, Qwen):** family-weighted tail-rate
  difference between `meanJ_span_safe` and its exact instantaneous
  rank+energy matched control at the frozen −1.0-nat per-item
  threshold, on confirmatory items. Test: within-item label-exchange
  randomization preserving threshold, protected-answer stratum, and
  family weights. The **preregistered comparator for the leakage
  account** is `prot_energy_matched` (reported beside, estimation).
- **P3-P3 (bridge-protection rescue):** on composed Bank F confirmatory
  items, the **family-weighted mean** of the within-item rescue
  contrast `lp(true-bridge-protected) − lp(distractor-bridge-protected)`
  (span-safe base protection, §6.5 piece sets), on **Qwen 3.6 27B** —
  resolved by the pinned rule: both development gates passed with
  identical any-piece coverage (Think 0.85 [`p3-bridge-dev-gate-
  olmo31-think-v1`], Qwen 0.85 [`p3-bridge-dev-gate-qwen36-27b-v1`],
  frac>floor 0.95 both), and exact ties break to Qwen. The rule was
  pinned before the Qwen gate ran; no rescue magnitude was viewed at
  family level.
  Test: within-item true/distractor label exchange = item-level sign
  flip of the contrast, family-weighted statistic
  (`stats.within_item_exchange_mean`), one-sided greater.

Holm over the three; α = 0.05 two-sided.

**Named estimation targets (no test, CIs reported):** the §4.4
continuity clause — `model_diff(Think, Instruct)` on the thick bank
beside the thin Phase 2 HP1; the Bank S composed−direct contrast per
model (first member of **Family B** per addendum §4.3 — the
working-memory reading of "workspace" turns on it); the
leakage/content decomposition (label vs span-safe vs prot-energy) per
model.

## 2 · Conditions (per item, randomized order within item)

baseline · `meanJ_span_safe` (primary) · exact instantaneous
rank+energy matched control (primary comparator, orthogonal to
protected rows AND span-safe residualized) · `meanJ_label_protected`
(secondary, continuity) · `prot_energy_matched` (leakage comparator) ·
mechanics random · logit-protected (readout-basis control).
Bridge arms on composed items per §6.5 (P3-P3 model only):
output+true-bridge protected, output+distractor-bridge protected.
The arm list is final (the power simulation showed no budget excess);
§16 drop rules would apply only under a mid-block interruption, never
below primary + comparator + baseline + one mechanics control.

## 3 · Cohorts and partition

- G5 capability gate (`g5_bank_scoring.py`, greedy MAX_NEW=8,
  deterministic grading): an item enters a model's cohort if
  `capable_generation` on BOTH its direct and composed variants;
  baseline logprob is a covariate, never a window (R2/§5.5).
- Primary cross-model cohort = the three-model intersection.
- Partition: `family_split_v2` (seed-ACTIVE). **Seed rule, pinned:**
  `seed = int(parent_sha[:8], 16) % 100000` where `parent_sha` is the
  full sha of the freeze commit's parent — deterministic, auditable,
  and fixed by history that exists before any outcome. Floors: ≥36
  families per side overall, ≥20 Bank F families per side, ≥25/side
  intersection-capable; ≤10% single-family item share; standardized
  imbalance ≤0.35 on every balance dimension. Assignment + balance
  report frozen in the freeze commit; disjointness asserted on
  canonical_family, fact_id, template_hash, and (bridge, answer)
  triple.
- **[PENDING G5: cohort sizes per model, intersection count, per-side
  family counts.]**

## 4 · Statistics

Per §14: paired contrasts first (§14.1 chains); primary randomization
tests (§14.2); wild cluster bootstrap-t for CIs (§14.3); hierarchical
model as secondary sensitivity only (§14.4); heavy-tail reporting in
full (§14.5: mean+median, tail rates at frozen thresholds, conditional
magnitude, ECDF, family distributions, LOFO, threshold curve);
equivalence claims only via frozen-SESOI TOST at 90% (§14.6);
SESOI = **0.15 nats** for within-fact composition contrasts (frozen in
§5); multiplicity per §14.7 families.

## 5 · Power and floors

From `p3-power-sim-v3` (dev-calibrated: between-family SD 0.94,
within-family SD 1.01, zero-inflation 0.06, cross-model item r ≈ 0;
null-calibrated at α/3: rejection 1.3–1.5%):

| scenario | P3-P1 MDE@90% (36 fam/side) | P3-P2 MDE@90% |
|---|---|---|
| raw dev variances (no within-fact cancellation) | >0.5 nats | 0.2 tail-points (30–36 fam) |
| cancellation ×0.4 | 0.5 nats | — |
| cancellation ×0.25 | 0.3 nats | — |

**Frozen floors:** ≥36 families per side overall; ≥20 Bank F families
per side; ≥25 intersection-capable families per side (R2). **Disclosed
power statement:** P3-P2 is adequately powered against the Phase 2
anchor effect (+0.279). P3-P1 is powered only for effects ≥0.3–0.5
nats depending on the realized within-fact cancellation — it is
carried as a test with this disclosure, and the §1 estimation targets
carry the quantitative story regardless of its outcome (the Phase 2
binary-primary lesson, §0.4 hierarchy). **SESOI for any equivalence
claim: 0.15 nats (TOST at 90%).**

## 6 · Prohibited claims and wording gates

Nextsteps §4.5 verbatim (no consciousness claims; no "OLMo lacks a
workspace"; no capacity-causes-shape from three points; rank+energy ≠
all structure; label protection does NOT preserve the output subspace;
no Gemma non-differentiability claims; finite-scale nonlinearity ≠ no
useful tangent; LLM-authored factual prompts ≠ all reasoning;
no null→equivalence without TOST). Additionally: any "selective"
wording for Instruct requires the Workstream C exact-control map
(R4); until then the sentence is "nonspecific J-channel vulnerability
cannot be excluded for Instruct."

## 7 · Reproduction gates

N8 Level 1 passed (`p3-n8-level1-repro-v1`). Levels 2 (≥20-item
sentinel × all conditions × 3 models) and 3 (one full GPU cell, Qwen
preferred) are public-release gates and run post-grid from this frozen
document.

## 8 · Deviations

Deviations follow the Phase 2 pattern: a numbered amendment file,
registered, with the stop-rule evidence that triggered it; never a
silent patch.

## 9 · Freeze gate checklist (all must close cleanly; failures iterate)

- [ ] G5 bank scoring banked on all three primaries; intersection
      floors met (≥25 families/side feasible).
- [ ] Workstream C grids banked ×3 + figure/stats (R4 discharge state
      recorded either way).
- [ ] P3-P3 model chosen by the dev identifiability gate (coverage +
      measurability metrics registered BEFORE any rescue means are
      viewed at family level).
- [ ] `family_split_v2` run with the declared seed; balance report
      clean; disjointness asserts pass.
- [ ] Power simulation banked; floors + SESOI filled above.
- [ ] §5.8 acceptance gate: planted-interaction recovery test green
      (§15.3 suite), alias audits pass, no family >10%/side.
- [ ] All 57+ conformance tests green at the freeze commit.
- [ ] Every PENDING marker in this document resolved (the freeze
      action greps for the bracket form and refuses while any remain);
      then the freeze commit renames it and tags
      `jspace-phase3-freeze-v1`.
