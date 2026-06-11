# Lab 4 verification report — probing with controls

Date: 2026-06-11 · Machine: Colab A100-SXM4-80GB · Branch: `lab1_colab`

## What was built

- **Frozen datasets** (`data/`): three truth-statement families (cities, numeric
  comparisons, negations; 60 each, balanced), generated deterministically by
  `data/make_truth_sets.py`, vendored per course rule. Negations exist so surface
  co-occurrence anti-correlates with truth.
- **Lab module**: dual probes per layer in pure torch (logistic LBFGS + mass-mean — no
  sklearn; "nobody runs code they can't explain"), deterministic id-hash 70/30 splits,
  shuffled-label refits, random-direction and token-length baselines, family-held-out
  transfer, selectivity, a surface calibration track probed from the SAME forwards,
  and `truth_direction.pt` saved with full metadata for Lab 7's causal hand-off.
- **Plots**: decodability-by-layer (surface vs truth vs all controls, one axes),
  3×3 generalization matrices for both probes, selectivity curves, and truth-separation
  projection panels at five depths.
- No bench changes were needed beyond a per-lab tier-default override
  (`--max-examples` is per-family here); Lab 1's capture and self-checks carry the lab.

## Two findings during validation that became course material

1. **The outlier specimen.** One frozen statement — "The city of Havana is in the
   Netherlands." — produces a final-position stream with ~7× the norm of every other
   statement on Olmo-3 (5 such rogue statements total, all flagged in
   `statement_manifest.csv`). Before per-row unit normalization, that single row
   hijacked the class-mean difference and pinned mass-mean accuracy to chance at every
   layer *while logistic regression worked perfectly* — a broken pipeline masquerading
   as a clean negative result. The handout teaches the specimen; the manifest carries a
   permanent `norm_outlier` tripwire.
2. **Separable ≠ mean-dominant.** After the fix, the probes still disagree honestly:
   logistic decodes truth from L9 (peak 0.984 at L19); the mass-mean direction only
   snaps in at L31–32. At course sample sizes, truth is linearly separable mid-stack
   but only becomes a dominant mean direction late. The decodability plot shows both
   curves with controls on the same axes.

## Validation evidence (Tier B, Olmo-3-7B, 180 statements)

| Measurement | Result |
|---|---|
| Truth peak (logistic, within-family held-out) | **0.984** at layer 19/32 |
| Surface track (final word contains 'a') | 0.88 from layer 0 — flat |
| Shuffled-label / random-direction controls | ~0.5 throughout |
| Token-length baseline | 0.43 |
| Best cross-family layer (affirmative mass-mean) | 32 (worst transfer 0.85) |
| Saved direction | **comparisons-trained** mass-mean @ L32, worst transfer **0.917** incl. negations |
| Cities-trained direction on negations | 0.40 — the expected Geometry-of-Truth inversion, reported not hidden |

The train-family asymmetry is itself a finding: cities and negations share a surface
template, so a cities-trained direction can ride the template (and inverts on
negations); a comparisons-trained direction has to mean truth — and transfers at
0.92/0.93 to both other families. The saved-direction selection encodes this reasoning
and records it in metadata.

Tier A (gpt2, 60 statements) exercises the full pipeline; gpt2-small shows weak truth
decodability as expected — the smoke tier validates plumbing, not science.

## Runs included

- `lab04_probing_controls-*/` — Tier A smoke (gpt2)
- `lab04_tierb_full/` — Tier B science run (Olmo-3-7B, 180 statements, 33 depths,
  ~600 probe fits with controls)
