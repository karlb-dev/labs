# Lab 22 Operationalization Audit

## What Was Measured

The lab fits a train-split residual direction separating benchmark/eval framing from natural phrasing of the same benign task. It then asks whether the direction transfers across eval formats, stays above format-matched controls, and changes benign generations under activation addition.

## What Was Not Measured

The lab does not measure inner experience, strategic deception, or a general belief that the model is inside an evaluation. The phrase `eval awareness` is a shorthand for this operational contrast unless the controls earn more.

## Cheap Explanations and Where They Are Audited

| Cheap explanation | Artifact | What would kill the strong story? |
|---|---|---|
| Multiple-choice detector | `olmo3_7b_full_s0_cross_format_transfer.csv`, `olmo3_7b_full_s0_surface_baseline_auc.csv` | Only MCQ heldout works, or choice-marker baseline matches the probe. |
| Question/Answer boilerplate detector | `olmo3_7b_full_s0_projection_by_item.csv`, `olmo3_7b_full_s0_surface_baseline_auc.csv` | Format controls project like true eval prompts. |
| Difficulty/formality detector | `source-run split_balance.csv`, `olmo3_7b_full_s0_surface_baseline_auc.csv` | Difficulty metadata predicts expected eval context as well as the direction. |
| Chat-template boundary artifact | `source-run exact_chat_hook_parity.json`, `source-run activation_capture.csv` | The measured token is not the rendered assistant-generation boundary. |
| Behavior-marker artifact | `source-run steering_generations.csv`, hand labels | Steering changes verbosity, hedging, or answer format but not eval-context behavior. |
| Thinking-span rationalization | `olmo3_7b_full_s0_cot_mentions_eval.csv` | CoT mentions eval only after answer-format markers or not at all. |

## Current Run

- best depth: 9
- eval AUC at best depth: 0.9306
- eval-vs-format-control AUC at best depth: 0.7917
- cross-format min/mean AUC: 0.9062 / 0.9635
- cross-format min null-adjusted AUC gap: -0.0312
- best surface baseline AUC on eval split: 0.7049
- decode verdict: `cross_format_but_null_or_surface_controls_competitive`
- causal marker verdict: `not_validated_by_marker_controls`

## Allowed Claim

Use `eval-awareness direction` only as an operational label unless cross-format transfer survives surface and format controls. If it fails, the result is still useful: you found the boundary of a template detector.
