# Lab 14 Validation

## Lab 14: Certainty, Hedging, and Calibration

This directory is the final Lab 14 validation pack for the current repository
code. It keeps the June 20, 2026 validation artifacts and drops the older
June 15 run pack so students and reviewers see one coherent result set.

## Validation Read

The result is a strong partial positive. Lab 14 now validates a downstream
usable answerability/certainty direction under controls on two of three
Olmo-3-7B-Instruct seeds, with the third seed still decoding answerability but
flagged because style/confound controls compete. The SmolLM comparison also
passes the current instrument criteria.

The cleanest current claim is:

```text
Lab 14 finds a controlled answerability direction in residual activations. It
is usable as an instrument for downstream labs, but it is not a direct meter of
subjective confidence, knowledge, belief, or honesty.
```

## Headline Result

The primary plotted validation run is `olmo3_7b_full_s0`:

- Model: `allenai/Olmo-3-7B-Instruct`
- Corpus: 80 fixed-choice certainty/calibration items
- Families: factual QA, freeform answerability, MCQ, passage QA, procedural logic
- Verdict: `usable_certainty_instrument`
- Best certainty depth: 11
- Certainty eval AUC: 0.8889
- Random-control AUC: 0.5191
- Shuffled-control AUC: 0.5147
- Control gap: 0.3698
- Mean family-held-out AUC: 0.8282
- Mean family-held-out control gap: 0.2913
- Distribution-confidence answerability AUC: 0.5756
- Hedging-style projection answerability AUC: 0.6556
- Max length/letter/answer-frame baseline AUC: 0.6667

Across Olmo seeds, the answerability signal is consistent: eval AUC is
0.8889, 0.9556, and 0.9200. The caution is interpretive rather than numerical:
seed 1 is marked `answerability_decodes_but_confounds_compete` because the
hedging-style projection is also very predictive.

## Current Result Summary

| Source label | Model | Main result | Read |
|---|---|---|---|
| `olmo3_7b_full_s0` | Olmo-3-7B-Instruct | AUC 0.8889, control gap 0.3698 | Usable instrument; primary plotted pack |
| `olmo3_7b_full_s1` | Olmo-3-7B-Instruct | AUC 0.9556, control gap 0.4294 | Decodes strongly, but confounds compete |
| `olmo3_7b_full_s2` | Olmo-3-7B-Instruct | AUC 0.9200, control gap 0.3751 | Usable instrument |
| `smollm_full_s0` | SmolLM2-135M-Instruct | AUC 0.9200, control gap 0.3236 | Usable tier-A comparison |

## What This Lab Teaches

- A certainty-looking direction can be useful only after it beats shuffled,
  random, family-held-out, style, entropy, length, and answer-frame controls.
- The validated object is an answerability direction in a controlled A/B/C/D
  frame, not an introspective confidence meter.
- Verbal confidence and internal projection can disagree. Those disagreement
  examples are the teaching hook for SELF-REPORT caution.
- A high AUC alone is not enough: seed 1 shows that controls can keep a result
  cautious even when the headline probe score is excellent.

## Curated Artifacts

Summary:

- `lab14_validation_report.md`
- `lab14_validation_summary.csv`

Primary Olmo plotted pack:

- `olmo3_7b_full_s0_run_summary.md`
- `olmo3_7b_full_s0_certainty_instrument_card.md`
- `olmo3_7b_full_s0_operationalization_audit.md`
- `olmo3_7b_full_s0_metrics.json`
- `olmo3_7b_full_s0_certainty_evidence_dashboard.png`
- `olmo3_7b_full_s0_certainty_probe_by_layer.png`
- `olmo3_7b_full_s0_controlled_depth_gap_atlas.png`
- `olmo3_7b_full_s0_family_heldout_generalization.png`
- `olmo3_7b_full_s0_signal_evidence_matrix.png`
- `olmo3_7b_full_s0_confidence_signal_correlations.png`
- `olmo3_7b_full_s0_confound_audit.png`
- `olmo3_7b_full_s0_confidence_disagreement_matrix.png`
- `olmo3_7b_full_s0_item_uncertainty_ribbons.png`
- `olmo3_7b_full_s0_reliability_diagram.png`
- `olmo3_7b_full_s0_verbal_confidence_audit.png`

Selected tables for every retained run:

- `*_certainty_evidence_matrix.csv`
- `*_depth_selection.csv`
- `*_family_heldout_generalization.csv`
- `*_signal_predictiveness.csv`
- `*_disagreement_examples.csv`
- `*_calibration_summary.csv`
- `*_length_and_letter_baselines.csv`

Additional retained run cards:

- `olmo3_7b_full_s1_*`
- `olmo3_7b_full_s2_*`
- `smollm_full_s0_*`

## Caveats

- This is a curated validation directory; full raw runs remain outside this
  pack.
- The saved direction is an operational answerability instrument, not evidence
  of subjective confidence.
- The seed-1 Olmo run should be discussed as a cautionary control case because
  hedging/style features are also predictive.
- The only retained plots are from the primary seed-0 Olmo plotted run.
