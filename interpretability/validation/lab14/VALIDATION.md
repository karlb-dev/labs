# Lab 14 Validation

## Lab 14: Certainty, Hedging, and Calibration

Certainty, hedging, and calibration: internal answerability, entropy, and verbal confidence.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab14_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab14_olmo32bthink_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `n_items`=36, `best_certainty_depth`=7, `best_hedging_depth`=3, `certainty_auc_eval_best_depth`=0.7222, `certainty_auc_train_best_depth`=1, `certainty_control_gap_eval_best_depth`=0.1666, `certainty_control_gap_train_best_depth`=0.4167, `certainty_random_auc_eval_best_depth`=0.5556
  - Model: `allenai/Olmo-3-32B-Think`
  - Items: 36
  - Verdict: `answerability_decodes_but_confounds_compete`
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab14_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab14_gemma4e4b_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `n_items`=36, `best_certainty_depth`=10, `best_hedging_depth`=21, `certainty_auc_eval_best_depth`=0.5556, `certainty_auc_train_best_depth`=0.9861, `certainty_control_gap_eval_best_depth`=-0.0444, `certainty_control_gap_train_best_depth`=0.3944, `certainty_random_auc_eval_best_depth`=0.6
  - Model: `google/gemma-4-E4B-it`
  - Items: 36
  - Verdict: `not_validated_as_certainty_instrument`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab14_tierc_labs1_25_full_matrix_20260615_000508/lab14_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `n_items`=36, `best_certainty_depth`=6, `best_hedging_depth`=9, `certainty_auc_eval_best_depth`=1, `certainty_auc_train_best_depth`=1, `certainty_control_gap_eval_best_depth`=0.3333, `certainty_control_gap_train_best_depth`=0.3986, `certainty_random_auc_eval_best_depth`=0.6667
  - Model: `allenai/Olmo-3-7B-Instruct`
  - Items: 36
  - Verdict: `answerability_decodes_but_confounds_compete`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab14_tiera_labs1_25_full_matrix_20260615_000508/lab14_tiera_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `n_items`=12, `best_certainty_depth`=8, `best_hedging_depth`=2, `certainty_auc_eval_best_depth`=1, `certainty_auc_train_best_depth`=1, `certainty_control_gap_eval_best_depth`=0.5, `certainty_control_gap_train_best_depth`=0.4444, `certainty_random_auc_eval_best_depth`=0.4889
  - Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - Items: 12
  - Verdict: `weak_or_family_limited_certainty_signal`

## What This Lab Teaches

- The central lesson is decodability with controls: useful probes must survive selectivity, held-out data, and confound checks.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab14_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab14_olmo32bthink_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `n_items`=36; `best_certainty_depth`=7; `best_hedging_depth`=3 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab14_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab14_gemma4e4b_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `n_items`=36; `best_certainty_depth`=10; `best_hedging_depth`=21 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab14_tierc_labs1_25_full_matrix_20260615_000508/lab14_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `n_items`=36; `best_certainty_depth`=6; `best_hedging_depth`=9 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab14_tiera_labs1_25_full_matrix_20260615_000508/lab14_tiera_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `n_items`=12; `best_certainty_depth`=8; `best_hedging_depth`=2 |

## Curated Artifacts

- `olmo3_32b_lab14_olmo32bthink_labs1_25_local_reruns_2026061_certainty_evidence_dashboard.png`
- `olmo3_32b_lab14_olmo32bthink_labs1_25_local_reruns_2026061_signal_evidence_matrix.png`
- `olmo3_32b_lab14_olmo32bthink_labs1_25_local_reruns_2026061_results.csv`
- `olmo3_32b_lab14_olmo32bthink_labs1_25_local_reruns_2026061_metrics.json`
- `gemma4e4b_lab14_gemma4e4b_labs1_25_local_reruns_20260615_1_certainty_evidence_dashboard.png`
- `gemma4e4b_lab14_gemma4e4b_labs1_25_local_reruns_20260615_1_signal_evidence_matrix.png`
- `gemma4e4b_lab14_gemma4e4b_labs1_25_local_reruns_20260615_1_results.csv`
- `gemma4e4b_lab14_gemma4e4b_labs1_25_local_reruns_20260615_1_metrics.json`
- `olmo3_7b_lab14_tierc_labs1_25_full_matrix_20260615_000508_certainty_evidence_dashboard.png`
- `olmo3_7b_lab14_tierc_labs1_25_full_matrix_20260615_000508_family_signal_atlas.png`
- `olmo3_7b_lab14_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `olmo3_7b_lab14_tierc_labs1_25_full_matrix_20260615_000508_metrics.json`
- `smollm_lab14_tiera_labs1_25_full_matrix_20260615_000508_certainty_evidence_dashboard.png`
- `smollm_lab14_tiera_labs1_25_full_matrix_20260615_000508_family_signal_atlas.png`
- `smollm_lab14_tiera_labs1_25_full_matrix_20260615_000508_results.csv`
- `smollm_lab14_tiera_labs1_25_full_matrix_20260615_000508_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
