# Lab 32 Validation

## Lab 32: Reward Models and Preference Circuits

Reward/preference circuits: DPO-style proxy, shortcut controls, split-aware residual directions, and judge-prompt activation addition.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/lab32_tierc_full_verify_20260614_2320/lab32_tierc_full_verify_20260614_2320` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `basis_split`=eval, `best_score_auc`=1, `best_score_family`=residual_direction, `best_score_name`=preference_residual_direction, `best_score_split`=train, `best_shortcut_or_control_auc`=0.9074, `causal_shift_over_random`=0.0192, `dpo_proxy_auc`=0.7778
  - data rows: 48 selected from `preference_circuit_pairs.csv`
  - data source: `frozen_csv`
  - domains: `{'helpfulness': 8, 'factual_honesty': 5, 'anti_sycophancy': 8, 'uncertainty': 7, 'privacy_boundary': 7, 'style_following': 7, 'concision': 6}`
- `interpret/verify_part3/lab32_tierc_final_verify_20260614_2347/lab32_tierc_final_verify_20260614_2347` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `basis_split`=eval, `best_score_auc`=1, `best_score_family`=residual_direction, `best_score_name`=preference_residual_direction, `best_score_split`=train, `best_shortcut_or_control_auc`=0.9074, `causal_shift_over_random`=0.0192, `dpo_proxy_auc`=0.7778
  - data rows: 48 selected from `preference_circuit_pairs.csv`
  - data source: `frozen_csv`
  - domains: `{'helpfulness': 8, 'factual_honesty': 5, 'anti_sycophancy': 8, 'uncertainty': 7, 'privacy_boundary': 7, 'style_following': 7, 'concision': 6}`
- `interpret/verify_part3/lab32_olmo32bthink_final_verify_20260614_2357/lab32_olmo32bthink_final_verify_20260614_2357` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `basis_split`=eval, `best_score_auc`=1, `best_score_family`=residual_direction, `best_score_name`=preference_residual_direction, `best_score_split`=train, `best_shortcut_or_control_auc`=0.8704, `causal_shift_over_random`=-0.0156, `dpo_proxy_auc`=0.8333
  - data rows: 48 selected from `preference_circuit_pairs.csv`
  - data source: `frozen_csv`
  - domains: `{'helpfulness': 8, 'factual_honesty': 5, 'anti_sycophancy': 8, 'uncertainty': 7, 'privacy_boundary': 7, 'style_following': 7, 'concision': 6}`
- `interpret/verify_part3/lab32_gemma4e4b_final_verify_20260614_2352/lab32_gemma4e4b_final_verify_20260614_2352` (google/gemma-4-E4B-it, tier c)
  - Metrics: `basis_split`=eval, `best_score_auc`=1, `best_score_family`=residual_direction, `best_score_name`=preference_residual_direction, `best_score_split`=train, `best_shortcut_or_control_auc`=0.8611, `causal_shift_over_random`=0.2858, `dpo_proxy_auc`=0.5
  - data rows: 48 selected from `preference_circuit_pairs.csv`
  - data source: `frozen_csv`
  - domains: `{'helpfulness': 8, 'factual_honesty': 5, 'anti_sycophancy': 8, 'uncertainty': 7, 'privacy_boundary': 7, 'style_following': 7, 'concision': 6}`

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/lab32_tierc_full_verify_20260614_2320/lab32_tierc_full_verify_20260614_2320` | `allenai/Olmo-3-1125-32B` | `c` | `basis_split`=eval; `best_score_auc`=1; `best_score_family`=residual_direction |
| `interpret/verify_part3/lab32_tierc_final_verify_20260614_2347/lab32_tierc_final_verify_20260614_2347` | `allenai/Olmo-3-1125-32B` | `c` | `basis_split`=eval; `best_score_auc`=1; `best_score_family`=residual_direction |
| `interpret/verify_part3/lab32_olmo32bthink_final_verify_20260614_2357/lab32_olmo32bthink_final_verify_20260614_2357` | `allenai/Olmo-3-32B-Think` | `c` | `basis_split`=eval; `best_score_auc`=1; `best_score_family`=residual_direction |
| `interpret/verify_part3/lab32_gemma4e4b_final_verify_20260614_2352/lab32_gemma4e4b_final_verify_20260614_2352` | `google/gemma-4-E4B-it` | `c` | `basis_split`=eval; `best_score_auc`=1; `best_score_family`=residual_direction |
| `interpret/verify_part3/lab32_tierb_full_verify_20260614_2314/lab32_tierb_full_verify_20260614_2314` | `allenai/Olmo-3-1025-7B` | `b` | `basis_split`=eval; `best_score_auc`=1; `best_score_family`=residual_direction |

## Curated Artifacts

- `olmo3_32b_lab32_tierc_full_verify_20260614_2320_overview_dashboard.png`
- `olmo3_32b_lab32_tierc_full_verify_20260614_2320_layer_sweep_heatmap.png`
- `olmo3_32b_lab32_tierc_full_verify_20260614_2320_results.csv`
- `olmo3_32b_lab32_tierc_full_verify_20260614_2320_metrics.json`
- `olmo3_32b_lab32_tierc_final_verify_20260614_2347_overview_dashboard.png`
- `olmo3_32b_lab32_tierc_final_verify_20260614_2347_layer_sweep_heatmap.png`
- `olmo3_32b_lab32_tierc_final_verify_20260614_2347_results.csv`
- `olmo3_32b_lab32_tierc_final_verify_20260614_2347_metrics.json`
- `olmo3_32b_lab32_olmo32bthink_final_verify_20260614_2357_overview_dashboard.png`
- `olmo3_32b_lab32_olmo32bthink_final_verify_20260614_2357_layer_sweep_heatmap.png`
- `olmo3_32b_lab32_olmo32bthink_final_verify_20260614_2357_results.csv`
- `olmo3_32b_lab32_olmo32bthink_final_verify_20260614_2357_metrics.json`
- `gemma4e4b_lab32_gemma4e4b_final_verify_20260614_2352_overview_dashboard.png`
- `gemma4e4b_lab32_gemma4e4b_final_verify_20260614_2352_layer_sweep_heatmap.png`
- `gemma4e4b_lab32_gemma4e4b_final_verify_20260614_2352_results.csv`
- `gemma4e4b_lab32_gemma4e4b_final_verify_20260614_2352_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
