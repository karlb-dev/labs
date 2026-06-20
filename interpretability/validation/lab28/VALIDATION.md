# Lab 28 Validation

## Lab 28: Mechanistic Editing and Unlearning

Mechanistic editing and unlearning: reversible localized activation edits with retain/paraphrase audits.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/lab28_tierc_full_verify_20260614_2256/lab28_tierc_full_verify_20260614_2256` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `lab_id`=L28, `lab_name`=lab28_editing_unlearning, `localization_editability_corr`=0.9712, `mean_paraphrase_gain`=2.092, `mean_retain_damage`=1.443, `mean_target_control_gap`=2.077, `mean_target_gain`=2.023, `n_counterexamples`=7
  - model: `allenai/Olmo-3-1125-32B` (64 blocks, d_model 5120)
  - data source: `frozen_csv`
  - data sha256: `1c1d995d1d4b5dbe`
- `interpret/verify_part3/lab28_olmo32bthink_full_verify_20260614_2307/lab28_olmo32bthink_full_verify_20260614_2307` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `lab_id`=L28, `lab_name`=lab28_editing_unlearning, `localization_editability_corr`=0.8189, `mean_paraphrase_gain`=2.312, `mean_retain_damage`=1.727, `mean_target_control_gap`=2.719, `mean_target_gain`=2.664, `n_counterexamples`=7
  - model: `allenai/Olmo-3-32B-Think` (64 blocks, d_model 5120)
  - data source: `frozen_csv`
  - data sha256: `1c1d995d1d4b5dbe`
- `interpret/verify_part3/lab28_tierb_full_verify_20260614_2252/lab28_tierb_full_verify_20260614_2252` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `lab_id`=L28, `lab_name`=lab28_editing_unlearning, `localization_editability_corr`=0.9852, `mean_paraphrase_gain`=2.277, `mean_retain_damage`=1.642, `mean_target_control_gap`=2.281, `mean_target_gain`=2.278, `n_counterexamples`=6
  - model: `allenai/Olmo-3-1025-7B` (32 blocks, d_model 4096)
  - data source: `frozen_csv`
  - data sha256: `1c1d995d1d4b5dbe`
- `interpret/verify_part3/lab28_gemma4e4b_full_verify_20260614_2303/lab28_gemma4e4b_full_verify_20260614_2303` (google/gemma-4-E4B-it, tier b)
  - Metrics: `lab_id`=L28, `lab_name`=lab28_editing_unlearning, `localization_editability_corr`=0.8028, `mean_paraphrase_gain`=1.2, `mean_retain_damage`=1.153, `mean_target_control_gap`=0.8499, `mean_target_gain`=0.998, `n_counterexamples`=5
  - model: `google/gemma-4-E4B-it` (42 blocks, d_model 2560)
  - data source: `frozen_csv`
  - data sha256: `1c1d995d1d4b5dbe`

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/lab28_tierc_full_verify_20260614_2256/lab28_tierc_full_verify_20260614_2256` | `allenai/Olmo-3-1125-32B` | `c` | `lab_id`=L28; `lab_name`=lab28_editing_unlearning; `localization_editability_corr`=0.9712 |
| `interpret/verify_part3/lab28_olmo32bthink_full_verify_20260614_2307/lab28_olmo32bthink_full_verify_20260614_2307` | `allenai/Olmo-3-32B-Think` | `c` | `lab_id`=L28; `lab_name`=lab28_editing_unlearning; `localization_editability_corr`=0.8189 |
| `interpret/verify_part3/lab28_tierb_full_verify_20260614_2252/lab28_tierb_full_verify_20260614_2252` | `allenai/Olmo-3-1025-7B` | `b` | `lab_id`=L28; `lab_name`=lab28_editing_unlearning; `localization_editability_corr`=0.9852 |
| `interpret/verify_part3/lab28_gemma4e4b_full_verify_20260614_2303/lab28_gemma4e4b_full_verify_20260614_2303` | `google/gemma-4-E4B-it` | `b` | `lab_id`=L28; `lab_name`=lab28_editing_unlearning; `localization_editability_corr`=0.8028 |
| `interpret/verify_part3/lab28_tiera_full_verify_20260614_2243/lab28_tiera_full_verify_20260614_2243` | `gpt2` | `a` | `lab_id`=L28; `lab_name`=lab28_editing_unlearning; `localization_editability_corr`=0.9968 |

## Curated Artifacts

- `olmo3_32b_lab28_tierc_full_verify_20260614_2256_layer_sweep_heatmap.png`
- `olmo3_32b_lab28_tierc_full_verify_20260614_2256_editing_unlearning_dashboard.png`
- `olmo3_32b_lab28_tierc_full_verify_20260614_2256_tables_method_capability_audit.csv`
- `olmo3_32b_lab28_tierc_full_verify_20260614_2256_results.csv`
- `olmo3_32b_lab28_olmo32bthink_full_verify_20260614_2307_layer_sweep_heatmap.png`
- `olmo3_32b_lab28_olmo32bthink_full_verify_20260614_2307_editing_unlearning_dashboard.png`
- `olmo3_32b_lab28_olmo32bthink_full_verify_20260614_2307_tables_method_capability_audit.csv`
- `olmo3_32b_lab28_olmo32bthink_full_verify_20260614_2307_results.csv`
- `olmo3_1025_7b_lab28_tierb_full_verify_20260614_2252_layer_sweep_heatmap.png`
- `olmo3_1025_7b_lab28_tierb_full_verify_20260614_2252_editing_unlearning_dashboard.png`
- `olmo3_1025_7b_lab28_tierb_full_verify_20260614_2252_tables_method_capability_audit.csv`
- `olmo3_1025_7b_lab28_tierb_full_verify_20260614_2252_results.csv`
- `gemma4e4b_lab28_gemma4e4b_full_verify_20260614_2303_layer_sweep_heatmap.png`
- `gemma4e4b_lab28_gemma4e4b_full_verify_20260614_2303_editing_unlearning_dashboard.png`
- `gemma4e4b_lab28_gemma4e4b_full_verify_20260614_2303_tables_method_capability_audit.csv`
- `gemma4e4b_lab28_gemma4e4b_full_verify_20260614_2303_results.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
