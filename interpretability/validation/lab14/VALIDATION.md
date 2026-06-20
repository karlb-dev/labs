# Lab 14 Validation

## Lab 14: Certainty, Hedging, and Calibration

Certainty, hedging, and calibration: internal answerability, entropy, and
verbal confidence.

## Current Read

Lab 14 is now a **mostly clean positive for internal answerability decoding**,
with one important caveat: OLMo seed 1 still shows style/self-report
entanglement, so the strongest claim is "usable on the primary and replicated
splits, but not fully style-separable on every split."

The v2c data replaces the old 36-row D-option-confounded set with 80 frozen
rows across five families:

- `mcq`
- `factual_qa`
- `passage_qa`
- `procedural_logic`
- `freeform_answerability`

The new dataset is balanced 40/40 answerable/unanswerable. Answer keys are
uniform across A-D within each label, and every row contains exactly one
unknown-style option with its option letter also uniform across A-D.

## Current Validation Runs

| Run | Model | Seed | Verdict | Certainty AUC | Control Gap | Family-Held-Out AUC | Family Gap | Max Shortcut AUC | Hedging-Style AUC |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `lab14_v2c_olmo3_7b_full_s0_plots_20260620` | `allenai/Olmo-3-7B-Instruct` | 0 | `usable_certainty_instrument` | 0.8889 | 0.3698 | 0.8282 | 0.2913 | 0.6667 | 0.6556 |
| `lab14_v2c_olmo3_7b_full_s1_20260620` | `allenai/Olmo-3-7B-Instruct` | 1 | `answerability_decodes_but_confounds_compete` | 0.9556 | 0.4294 | 0.9781 | 0.3806 | 0.6333 | 0.8356 |
| `lab14_v2c_olmo3_7b_full_s2_20260620` | `allenai/Olmo-3-7B-Instruct` | 2 | `usable_certainty_instrument` | 0.9200 | 0.3751 | 0.8781 | 0.3488 | 0.6667 | 0.6444 |
| `lab14_v2c_tiera_full_20260620` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 0 | `usable_certainty_instrument` | 0.9200 | 0.3236 | 0.8563 | 0.2944 | 0.6667 | 0.6178 |

## Interpretation

- The old answer-key shortcut is fixed: `unanswerable_always_D=false`, and
  answer-key counts are exactly balanced by label.
- The old question-length shortcut is fixed enough for validation:
  question-length answerability AUC is near chance in the OLMo seed 1 audit
  (`0.4889`) and prompt-token-length AUC is also near chance (`0.4933`).
- The internal direction is robust across OLMo seeds: eval AUC ranges from
  `0.8889` to `0.9556`; family-held-out AUC ranges from `0.8282` to `0.9781`.
- The remaining caveat is conceptual, not a CSV shortcut: verbal confidence and
  hedging-style projections can themselves track answerability. Seed 1 is
  therefore correctly held at `answerability_decodes_but_confounds_compete`.

## Current Artifacts

- `olmo3_7b_v2c_full_s0_metrics.json`
- `olmo3_7b_v2c_full_s0_results.csv`
- `olmo3_7b_v2c_full_s0_certainty_evidence_dashboard.png`
- `olmo3_7b_v2c_full_s0_signal_evidence_matrix.png`
- `olmo3_7b_v2c_full_s0_confound_audit.png`
- `olmo3_7b_v2c_full_s1_metrics.json`
- `olmo3_7b_v2c_full_s2_metrics.json`
- `smollm_v2c_full_metrics.json`
- `lab14_validation_summary.csv`
- `lab14_validation_report.md`

Raw run directories are backed up at:

`/content/drive/MyDrive/interpret/lab14_certainty_v2c_20260620/`

## Historical Context

Older 36-row runs were intentionally conservative and mostly reported
`answerability_decodes_but_confounds_compete` because all unanswerable rows used
the D/unknown answer frame. Those historical artifacts remain in this directory
for comparison, but the v2c runs above are the preferred validation read.

## Residual Risk

- This is still answerability/certainty-adjacent, not a direct measurement of
  honesty, belief, or truthfulness.
- The self-report/verbal gauge is predictive but poorly calibrated in OLMo
  (`verbal_confidence_ece` around `0.43` to `0.53` on OLMo full runs).
- A future stronger version should add residualized internal directions after
  projecting out the hedging-style direction, then report whether the internal
  signal remains after that stricter control.
