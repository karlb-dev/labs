# Scientific preregistration — J-space Phase 4

**CANDIDATE — NOT FROZEN — CONFIRMATORY AND REPLICATION OUTCOMES FORBIDDEN**

Version: candidate 0.1, 2026-07-31.
Governing plan: `jspace_lab_nextsteps_4_1.md` plus addendum §§3–5.
Phase 3 input boundary: `jspace-phase3-complete-v1` at `9e0672b`.

## 1. Question and claim boundary

Phase 4 asks what computation occupies the verbalizable channel, how
post-training reroutes it, when controlled working-set load engages it, and
when the fitted J transport is valid.

The compact confirmatory family is:

- **P4-P1 — Qwen bridge substitution.** On untouched families,
  counterfactual bridge substitution increases
  `LP(counterfactual answer) - LP(original answer)` relative to a
  geometry-matched unrelated substitution.
- **P4-P2 — Qwen mode by phase.** The span-safe bridge/J effect on final
  generation quality interacts with official thinking mode and intervention
  phase (prefill, reasoning, final answer).
- **P4-P3 — controlled load engagement.** The span-safe J-specific effect
  rises from the frozen low-load endpoint to the frozen high-load endpoint
  relative to the exact instantaneous rank-and-energy control on at least
  one preregistered model; the OLMo-pair/Qwen model-by-load interaction is
  estimated.

OLMo lineage is estimation-first. No adjacent-checkpoint contrast enters the
primary family unless a pre-outcome development transition, a new untouched
holdout, and a preregistration amendment are all present before outcomes.
Phase 4 is not a rescue attempt for P3-P1.

No primary result licenses “selective global workspace.” The default noun
remains “knowledge-access channel.” Working-memory language requires P4-P3
and its capability guard.

## 2. Immutable imports and development data

Phase 2/3 enter only through registry import events. Imports pin source
registry, evidence ID, commit/tag, and output hashes.

Known Phase 3 families may be used for:

- OLMo development trajectory plots;
- endpoint feasibility and dose calibration using geometry/activation gates;
- parser, hook, state, and scoring sentinels;
- bridge geometry and counterfactual endpoint engineering.

They may not support a new binary Phase 4 claim. Confirmatory and replication
families must be untouched by Phase 2/3 facts, bridges, answers, and outcome
inspection.

## 3. Shared measurement contract

The single typed `ScoringSpec` freezes:

- BOS/native tokenizer units;
- piecewise un-rstripped prompt/answer concatenation;
- rejection of trailing-whitespace bank defects;
- exact prompt and answer boundaries;
- prospective prefix-disjoint logsumexp over a frozen alias set;
- canonical and historical-first alias sensitivities;
- Unicode NFKD, lower-alphanumeric-space generation normalization;
- word-boundary generation matching;
- original / counterfactual / ambiguous / other-invalid generation outcomes;
- maximum prompt, answer, and generation lengths;
- official reasoning delimiter/parser version.

Baseline-only clean ranks are stored for every accepted alias. A protected
stratum is invalid without exact tokenizer IDs, tokenizer-manifest hash, and
per-alias ranks.

Every model job passes the same-process CUDA gate and asserts model parameters
are on CUDA. Model-scale CPU fallback is prohibited.

## 4. P4-P1 bridge bank and intervention

Before freeze, Bank B must contain at least 40 canonical relation families
with 4–6 facts per family, one true bridge, two plausible counterfactual
bridges, original/counterfactual answers, and direct/composed/
bridge-supplied/counterfactual-supplied variants.

Every item requires:

- independent source verification and ambiguity notes;
- no overlap with prior facts, bridges, or answers;
- true/counterfactual token-length strata;
- baseline original/counterfactual calibration;
- an unrelated bridge from another fact and family;
- per-item bridge geometry match report.

The frozen arms are baseline, span-safe J, exact instantaneous
rank-and-energy matched control, bridge-only lesion, matched unrelated
lesion, true-bridge protection, geometry-matched distractor protection,
true removal plus true reinjection, true removal plus counterfactual
injection, true removal plus unrelated injection, orthogonal injection, and
counterfactual-answer direction only.

Injection sign and dose are fixed using geometry and activation scale on a
development split without answer outcomes.

Primary item statistic:

```text
[LP(cf) - LP(orig)] counterfactual bridge injection
- [LP(cf) - LP(orig)] matched unrelated injection
```

Named secondaries are absolute original/counterfactual calibration,
original/counterfactual/other-invalid generation trichotomy, true protection
rescue, bridge-lesion damage, self-rescue, answer-direction contrast, and
layer/position/phase surfaces.

Confirmatory and replication sides are split by canonical family and analyzed
once in that order with identical code.

## 5. P4-P2 official mode and phase

Thinking on/off uses official model templates, not raw prompt approximations.
Development must freeze tokenizer/template revisions and demonstrate:

- phase parser golden tests;
- prompt, reasoning, and final-answer hook sentinels;
- parser failure rate below the frozen tolerance;
- measurable generation quality in both modes;
- no silent truncation or unmatched reasoning delimiter;
- correct/wrong/shuffled/filler rationale controls.

Primary outcome and exact interaction statistic will be filled and signed
before freeze after these endpoint gates, without viewing untouched-family
outcomes.

## 6. P4-P3 controlled load and capability guard

Bank W uses six task superfamilies with nested low-to-high load, matched
prompt length controls, shortcut audits, and family-level splits. The primary
statistic is the family-weighted high-minus-low change in J-specific effect.
A hierarchical full-ladder slope is secondary.

Before freeze, each model must either:

1. meet a prespecified baseline-accuracy flatness/equivalence criterion
   across the primary load endpoints; or
2. use a frozen baseline-capability covariate model and retain sufficient
   overlap at every load.

If neither is satisfied, that model’s load effect is descriptive and cannot
support “working-set” or “working-memory” language.

## 7. Randomization, intervals, and multiplicity

- The unit of inference is canonical family.
- P4-P1 uses family sign flips on within-item substitution contrasts.
- P4-P2 uses the frozen within-family factorial contrast.
- P4-P3 uses the frozen high-minus-low family contrast.
- Enumerate all sign patterns when feasible; otherwise use at least 100,000
  deterministic draws and the plus-one p-value.
- Report equal-family and item-weighted estimates.
- Confidence intervals are family-resampling percentile intervals unless
  explicitly labeled otherwise.
- Holm correction covers the three P4 primary p-values.
- Family leave-one-out and first/canonical alias views are sensitivities, not
  alternative decision rulers.
- SESOI and power targets must be filled from outcome-blind simulation before
  freeze.

## 8. Structured stop rules

Stop before outcomes and preserve state for:

- model/tokenizer/lens/config/bank/partition hash mismatch;
- checkpoint input-manifest mismatch;
- CUDA unavailable or model parameter not on CUDA;
- baseline capability/cohort drift beyond frozen tolerance;
- protected-set misalignment or missing clean-rank metadata;
- achieved rank or energy outside tolerance;
- bridge geometry mismatch;
- phase hook firing in the wrong phase;
- parser failure beyond tolerance;
- sentinel drift;
- delivered perturbation fidelity failure;
- exact JVP disagreement with the smallest faithful secant.

A stopped run cannot be silently resumed under a different manifest or
evidence ID.

## 9. Reproduction and immutable rows

Every claim regenerates from per-item rows containing the Section 18.3 schema
from the governing plan. Registry outputs are hash-pinned. Corrections use new
evidence IDs and append-only events. Independent reproduction rebuilds the
primary analysis, reruns sentinels and one complete model cell, verifies the
transport gate, and regenerates final figures.

## 10. Items that must be fixed before PI sign-off

- [ ] Bank B final family count, sources, power, SESOI, and split hash.
- [ ] P4-P1 geometry/dose tolerances and exact model/config revisions.
- [ ] P4-P2 official templates, parser tolerance, primary quality metric,
      interaction statistic, families, power, and split.
- [ ] Bank W task/load ladder, shortcut thresholds, capability
      flatness/equivalence margin or covariate formula, power, and split.
- [ ] Holm decision wording and directional alternatives.
- [ ] Development gates all pass without untouched-family outcomes.
- [ ] Environment lock and model/lens/tokenizer manifests are complete.
- [ ] Independent reviewer verifies no Phase 4 outcome leakage.

## 11. PI sign-off and freeze

PI approval: **PENDING**
Independent protocol review: **PENDING**
Freeze commit: **NOT CREATED**
Freeze tag: **NOT CREATED**

Until all four lines are complete, `phase4-confirmatory` and
`phase4-replication` jobs are forbidden.
