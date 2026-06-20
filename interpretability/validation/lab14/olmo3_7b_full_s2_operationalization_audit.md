# Lab 14 Operationalization Audit

## What was measured

The lab measures answerability/certainty in a controlled fixed-choice frame. Answerable questions have a checkable option. Known-unanswerable questions are scored with the designated unanswerable option, usually `D`. The internal direction is a residual-stream probe for this operational label, not a direct meter of subjective confidence.

## Cheap explanations and where they are tested

| Cheap explanation | Audit artifact | How it can kill the claim |
|---|---|---|
| Family or topic | `tables/family_heldout_generalization.csv` | Held-out AUC collapses or controls match the real direction. |
| Hedging or politeness style | `tables/probe_report.csv`, `tables/signal_predictiveness.csv` | The confident-vs-hedged direction predicts answerability as well as the certainty direction. |
| Entropy or option sharpness | `tables/answer_distribution_readout.csv`, `tables/signal_predictiveness.csv` | Distribution confidence explains the labels without the internal projection adding anything. |
| Length or formatting | `tables/length_and_letter_baselines.csv` | Prompt length, option length, answer letter, or unknown-option position predicts answerability competitively. |
| Self-report fluency | `tables/verbal_confidence_reports.csv`, `tables/reliability_curve.csv` | The model emits confidence words that are formatted but not calibrated. |
| Answer-letter metadata audit | `diagnostics/frozen_data_manifest.json`, `tables/length_and_letter_baselines.csv` | `answer_key_is_D` should be treated as label metadata, not a model-visible signal; if students rely on it, the claim is a frame artifact. |

## Headline numbers

- Verdict: `usable_certainty_instrument`
- Best certainty depth: 13
- Best hedging depth: 9
- Certainty eval AUC: 0.92
- Certainty eval control gap: 0.3751
- Mean family-held-out real AUC: 0.8781
- Internal/distribution correlation: -0.4139
- Internal/verbal correlation: 0.6929
- Internal/hedging-style correlation: -0.0188
- Verbal confidence ECE: 0.4267

## Allowed claim

An internal uncertainty-adjacent claim is allowed only when the direction predicts answerability beyond family/topic, shuffled/random controls, hedging style, length and prompt-text baselines, and entropy alone. Otherwise the honest claim is narrower: this model exposes a feature of the answer frame, style, prompt difficulty, or its own self-report behavior.