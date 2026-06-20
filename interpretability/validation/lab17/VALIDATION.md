# Lab 17 Validation

## Lab 17 - Persona, Voice, Roleplay, and Register

Persona, voice, roleplay, and register: paired directions, steering, and turn traces.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab17_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab17_olmo32bthink_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `best_depth`=13, `content_ok`=True, `injection_layer`=12, `mean_opposite_steering_style_delta_max_dose`=0, `mean_random_auc_best_depth`=0.625, `mean_random_steering_style_delta_max_dose`=0, `mean_real_auc_best_depth`=1, `mean_real_selectivity_vs_random`=0.375
  - Model: `allenai/Olmo-3-32B-Think`
  - Rows: 24 selected from `frozen_csv`
  - Best depth: 13 selected by train-only control-adjusted score
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab17_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab17_gemma4e4b_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `best_depth`=17, `content_ok`=True, `injection_layer`=16, `mean_opposite_steering_style_delta_max_dose`=-0.125, `mean_random_auc_best_depth`=0.7375, `mean_random_steering_style_delta_max_dose`=0, `mean_real_auc_best_depth`=1, `mean_real_selectivity_vs_random`=0.2625
  - Model: `google/gemma-4-E4B-it`
  - Rows: 24 selected from `frozen_csv`
  - Best depth: 17 selected by train-only control-adjusted score
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab17_tierc_labs1_25_full_matrix_20260615_000508/lab17_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `best_depth`=8, `content_ok`=True, `injection_layer`=7, `mean_opposite_steering_style_delta_max_dose`=-0.25, `mean_random_auc_best_depth`=0.5875, `mean_random_steering_style_delta_max_dose`=-0.125, `mean_real_auc_best_depth`=1, `mean_real_selectivity_vs_random`=0.4125
  - Model: `allenai/Olmo-3-7B-Instruct`
  - Rows: 24 selected from `frozen_csv`
  - Best depth: 8 selected by train-only control-adjusted score
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab17_tiera_labs1_25_full_matrix_20260615_000508/lab17_tiera_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `best_depth`=29, `content_ok`=False, `injection_layer`=28, `mean_opposite_steering_style_delta_max_dose`=0, `mean_random_auc_best_depth`=0.7, `mean_random_steering_style_delta_max_dose`=0, `mean_real_auc_best_depth`=1, `mean_real_selectivity_vs_random`=0.3
  - Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - Rows: 12 selected from `frozen_csv`
  - Best depth: 29 selected by train-only control-adjusted score

## What This Lab Teaches

- The central lesson is decodability with controls: useful probes must survive selectivity, held-out data, and confound checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab17_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab17_olmo32bthink_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `best_depth`=13; `content_ok`=True; `injection_layer`=12 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab17_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab17_gemma4e4b_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `best_depth`=17; `content_ok`=True; `injection_layer`=16 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab17_tierc_labs1_25_full_matrix_20260615_000508/lab17_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `best_depth`=8; `content_ok`=True; `injection_layer`=7 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab17_tiera_labs1_25_full_matrix_20260615_000508/lab17_tiera_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `best_depth`=29; `content_ok`=False; `injection_layer`=28 |

## Curated Artifacts

- `olmo3_32b_lab17_olmo32bthink_labs1_25_local_reruns_2026061_refusal_boundary_safety_dashboard.png`
- `olmo3_32b_lab17_olmo32bthink_labs1_25_local_reruns_2026061_persona_evidence_dashboard.png`
- `olmo3_32b_lab17_olmo32bthink_labs1_25_local_reruns_2026061_results.csv`
- `olmo3_32b_lab17_olmo32bthink_labs1_25_local_reruns_2026061_metrics.json`
- `gemma4e4b_lab17_gemma4e4b_labs1_25_local_reruns_20260615_1_refusal_boundary_safety_dashboard.png`
- `gemma4e4b_lab17_gemma4e4b_labs1_25_local_reruns_20260615_1_persona_evidence_dashboard.png`
- `gemma4e4b_lab17_gemma4e4b_labs1_25_local_reruns_20260615_1_results.csv`
- `gemma4e4b_lab17_gemma4e4b_labs1_25_local_reruns_20260615_1_metrics.json`
- `olmo3_7b_lab17_tierc_labs1_25_full_matrix_20260615_000508_refusal_boundary_safety_dashboard.png`
- `olmo3_7b_lab17_tierc_labs1_25_full_matrix_20260615_000508_persona_evidence_dashboard.png`
- `olmo3_7b_lab17_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `olmo3_7b_lab17_tierc_labs1_25_full_matrix_20260615_000508_metrics.json`
- `smollm_lab17_tiera_labs1_25_full_matrix_20260615_000508_refusal_boundary_safety_dashboard.png`
- `smollm_lab17_tiera_labs1_25_full_matrix_20260615_000508_persona_evidence_dashboard.png`
- `smollm_lab17_tiera_labs1_25_full_matrix_20260615_000508_results.csv`
- `smollm_lab17_tiera_labs1_25_full_matrix_20260615_000508_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
