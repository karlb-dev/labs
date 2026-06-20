# Lab 22 Validation

## Lab 22 - Eval Awareness Without the Fog Machine

Eval awareness: eval-vs-natural directions, cross-format controls, and safe steering.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab22_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab22_olmo32bthink_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `n_items`=24, `best_depth`=11, `best_surface_baseline_auc_eval_split`=1, `causal_marker_verdict`=not_validated_by_marker_controls, `cross_format_mean_auc`=1, `cross_format_min_auc`=1, `cross_format_min_format_control_gap`=0.1697, `decode_verdict`=cross_format_but_surface_baseline_competitive
  - Model: `allenai/Olmo-3-32B-Think`
  - Rows: 24
  - Formats: `['code_test', 'free_response', 'mcq', 'qa_benchmark']`
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab22_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab22_gemma4e4b_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `n_items`=24, `best_depth`=16, `best_surface_baseline_auc_eval_split`=1, `causal_marker_verdict`=not_validated_by_marker_controls, `cross_format_mean_auc`=1, `cross_format_min_auc`=1, `cross_format_min_format_control_gap`=4.699, `decode_verdict`=cross_format_but_surface_baseline_competitive
  - Model: `google/gemma-4-E4B-it`
  - Rows: 24
  - Formats: `['code_test', 'free_response', 'mcq', 'qa_benchmark']`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab22_tierc_labs1_25_full_matrix_20260615_000508/lab22_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Think, tier c)
  - Metrics: `n_items`=24, `best_depth`=14, `best_surface_baseline_auc_eval_split`=1, `causal_marker_verdict`=not_validated_by_marker_controls, `cross_format_mean_auc`=1, `cross_format_min_auc`=1, `cross_format_min_format_control_gap`=0.2972, `decode_verdict`=cross_format_but_surface_baseline_competitive
  - Model: `allenai/Olmo-3-7B-Think`
  - Rows: 24
  - Formats: `['code_test', 'free_response', 'mcq', 'qa_benchmark']`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab22_tiera_labs1_25_full_matrix_20260615_000508/lab22_tiera_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `n_items`=12, `best_depth`=26, `best_surface_baseline_auc_eval_split`=1, `causal_marker_verdict`=not_validated_by_marker_controls, `cross_format_mean_auc`=1, `cross_format_min_auc`=1, `cross_format_min_format_control_gap`=-31.25, `decode_verdict`=not_validated_or_format_detector
  - Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - Rows: 12
  - Formats: `['code_test', 'free_response', 'mcq', 'qa_benchmark']`

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab22_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab22_olmo32bthink_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `n_items`=24; `best_depth`=11; `best_surface_baseline_auc_eval_split`=1 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab22_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab22_gemma4e4b_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `n_items`=24; `best_depth`=16; `best_surface_baseline_auc_eval_split`=1 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab22_tierc_labs1_25_full_matrix_20260615_000508/lab22_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Think` | `c` | `n_items`=24; `best_depth`=14; `best_surface_baseline_auc_eval_split`=1 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab22_tiera_labs1_25_full_matrix_20260615_000508/lab22_tiera_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `n_items`=12; `best_depth`=26; `best_surface_baseline_auc_eval_split`=1 |

## Curated Artifacts

- `olmo3_32b_lab22_olmo32bthink_labs1_25_local_reruns_2026061_eval_awareness_evidence_dashboard.png`
- `olmo3_32b_lab22_olmo32bthink_labs1_25_local_reruns_2026061_format_transfer_matrix.png`
- `olmo3_32b_lab22_olmo32bthink_labs1_25_local_reruns_2026061_results.csv`
- `olmo3_32b_lab22_olmo32bthink_labs1_25_local_reruns_2026061_metrics.json`
- `gemma4e4b_lab22_gemma4e4b_labs1_25_local_reruns_20260615_1_eval_awareness_evidence_dashboard.png`
- `gemma4e4b_lab22_gemma4e4b_labs1_25_local_reruns_20260615_1_format_transfer_matrix.png`
- `gemma4e4b_lab22_gemma4e4b_labs1_25_local_reruns_20260615_1_results.csv`
- `gemma4e4b_lab22_gemma4e4b_labs1_25_local_reruns_20260615_1_metrics.json`
- `olmo3_7b_lab22_tierc_labs1_25_full_matrix_20260615_000508_eval_awareness_evidence_dashboard.png`
- `olmo3_7b_lab22_tierc_labs1_25_full_matrix_20260615_000508_multiturn_trace_atlas.png`
- `olmo3_7b_lab22_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `olmo3_7b_lab22_tierc_labs1_25_full_matrix_20260615_000508_metrics.json`
- `smollm_lab22_tiera_labs1_25_full_matrix_20260615_000508_eval_awareness_evidence_dashboard.png`
- `smollm_lab22_tiera_labs1_25_full_matrix_20260615_000508_multiturn_trace_atlas.png`
- `smollm_lab22_tiera_labs1_25_full_matrix_20260615_000508_results.csv`
- `smollm_lab22_tiera_labs1_25_full_matrix_20260615_000508_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
