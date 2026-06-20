# Lab 02 Validation

## Lab 2: Direct Logit Attribution and Component Accounting

Direct logit attribution: which components push toward or away from an answer.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/run6/C/lab2` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `n_examples`=25, `n_dropped`=1, `n_prefers_target_over_distractor`=21, `phase_summary_rows`=52, `spearman_attribution_vs_ablation`=0.8169, `worst_frozen_vs_model_abs_err`=0.02816, `worst_ledger_vs_frozen_abs_err`=0.03562
  - model: `allenai/Olmo-3-1125-32B` (64 blocks, d_model 5120)
  - dtype: `bfloat16` | quantization: `none` | ablate-top: 3
  - examples: 25 kept, 1 dropped at the single-token gate
- `interpret/run6/B/lab2` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `n_examples`=25, `n_dropped`=1, `n_prefers_target_over_distractor`=21, `phase_summary_rows`=52, `spearman_attribution_vs_ablation`=0.8454, `worst_frozen_vs_model_abs_err`=0.0382, `worst_ledger_vs_frozen_abs_err`=0.02535
  - model: `allenai/Olmo-3-1025-7B` (32 blocks, d_model 4096)
  - dtype: `bfloat16` | quantization: `none` | ablate-top: 3
  - examples: 25 kept, 1 dropped at the single-token gate
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab02_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab02_gemma4e4b_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `n_examples`=25, `n_dropped`=1, `n_prefers_target_over_distractor`=20, `phase_summary_rows`=52, `spearman_attribution_vs_ablation`=0.0008295, `worst_frozen_vs_model_abs_err`=11.9, `worst_ledger_vs_frozen_abs_err`=0.4606
  - model: `google/gemma-4-E4B-it` (42 blocks, d_model 2560)
  - dtype: `bfloat16` | quantization: `none` | ablate-top: 3
  - examples: 25 kept, 1 dropped at the single-token gate
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab02_tiera_labs1_25_full_matrix_20260615_000508/lab02_tiera_labs1_25_full_matrix_20260615_000508` (gpt2, tier a)
  - Metrics: `n_examples`=4, `n_dropped`=0, `n_prefers_target_over_distractor`=3, `phase_summary_rows`=52, `spearman_attribution_vs_ablation`=0.4707, `worst_frozen_vs_model_abs_err`=1.55e-05, `worst_ledger_vs_frozen_abs_err`=4.461e-07
  - model: `gpt2` (12 blocks, d_model 768)
  - dtype: `float32` | quantization: `none` | ablate-top: 3
  - examples: 4 kept, 0 dropped at the single-token gate

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/run6/C/lab2` | `allenai/Olmo-3-1125-32B` | `c` | `n_examples`=25; `n_dropped`=1; `n_prefers_target_over_distractor`=21 |
| `interpret/run6/B/lab2` | `allenai/Olmo-3-1025-7B` | `b` | `n_examples`=25; `n_dropped`=1; `n_prefers_target_over_distractor`=21 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab02_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab02_gemma4e4b_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `n_examples`=25; `n_dropped`=1; `n_prefers_target_over_distractor`=20 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab02_tiera_labs1_25_full_matrix_20260615_000508/lab02_tiera_labs1_25_full_matrix_20260615_000508` | `gpt2` | `a` | `n_examples`=4; `n_dropped`=0; `n_prefers_target_over_distractor`=3 |
| `interpret/validate_part2_plots/lab2` | `unknown` | `` | see copied summaries |

## Curated Artifacts

- `olmo3_32b_run6c_dla_dashboard.png`
- `olmo3_32b_run6c_relation_family_ledger_matrix.png`
- `olmo3_32b_run6c_tables_phase_ledger_summary.csv`
- `olmo3_32b_run6c_results.csv`
- `olmo3_1025_7b_run6b_dla_dashboard.png`
- `olmo3_1025_7b_run6b_relation_family_ledger_matrix.png`
- `olmo3_1025_7b_run6b_tables_phase_ledger_summary.csv`
- `olmo3_1025_7b_run6b_results.csv`
- `gemma4e4b_lab02_gemma4e4b_labs1_25_local_reruns_20260615_1_dla_dashboard.png`
- `gemma4e4b_lab02_gemma4e4b_labs1_25_local_reruns_20260615_1_relation_family_ledger_matrix.png`
- `gemma4e4b_lab02_gemma4e4b_labs1_25_local_reruns_20260615_1_tables_phase_ledger_summary.csv`
- `gemma4e4b_lab02_gemma4e4b_labs1_25_local_reruns_20260615_1_results.csv`
- `gpt2_lab02_tiera_labs1_25_full_matrix_20260615_000508_dla_dashboard.png`
- `gpt2_lab02_tiera_labs1_25_full_matrix_20260615_000508_signed_component_heatmap.png`
- `gpt2_lab02_tiera_labs1_25_full_matrix_20260615_000508_tables_phase_ledger_summary.csv`
- `gpt2_lab02_tiera_labs1_25_full_matrix_20260615_000508_results.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
