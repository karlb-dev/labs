# Lab 13 Validation

## Lab 13: Emotion Geometry, Reading Affect vs. Writing Affect

Emotion geometry: read/write affect directions, transfer, confounds, and safe steering.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab13_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab13_olmo32bthink_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `audit_result`=mixed, `best_depth`=49, `claim_allowed`=affect handle with unresolved confounds, `headline_steering_dose`=0.8, `injection_layer`=48, `max_abs_sentiment_cosine`=0.3038, `mean_abs_confound_projection_delta`=7.051, `mean_comp_gen_cosine`=0.107
  - Mean cross read/write AUC: 0.7812
  - Selected-depth control-adjusted cross AUC: 0.2812
  - Mean specificity AUC: 0.8333
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab13_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab13_gemma4e4b_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `audit_result`=mixed, `best_depth`=23, `claim_allowed`=affect handle with unresolved confounds, `headline_steering_dose`=0.8, `injection_layer`=22, `max_abs_sentiment_cosine`=0.2783, `mean_abs_confound_projection_delta`=8.117, `mean_comp_gen_cosine`=0.0353
  - Mean cross read/write AUC: 0.8125
  - Selected-depth control-adjusted cross AUC: 0.2812
  - Mean specificity AUC: 0.6146
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab13_tierc_labs1_25_full_matrix_20260615_000508/lab13_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `audit_result`=passed, `best_depth`=18, `claim_allowed`=emotion-specific read/write transfer; causal steering pending hand-labeled generations, `headline_steering_dose`=0.8, `injection_layer`=17, `max_abs_sentiment_cosine`=0.37, `mean_abs_confound_projection_delta`=3.287, `mean_comp_gen_cosine`=0.0605
  - Mean cross read/write AUC: 0.9688
  - Selected-depth control-adjusted cross AUC: 0.3281
  - Mean specificity AUC: 0.7812
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab13_tiera_labs1_25_full_matrix_20260615_000508/lab13_tiera_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `audit_result`=failed, `best_depth`=8, `claim_allowed`=no defended emotion-geometry claim, `headline_steering_dose`=0.8, `injection_layer`=7, `max_abs_sentiment_cosine`=0.3775, `mean_abs_confound_projection_delta`=0.6804, `mean_comp_gen_cosine`=0.0071
  - Mean cross read/write AUC: 0.5
  - Selected-depth control-adjusted cross AUC: -0.125
  - Mean specificity AUC: 0.5833

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab13_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab13_olmo32bthink_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `audit_result`=mixed; `best_depth`=49; `claim_allowed`=affect handle with unresolved confounds |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab13_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab13_gemma4e4b_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `audit_result`=mixed; `best_depth`=23; `claim_allowed`=affect handle with unresolved confounds |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab13_tierc_labs1_25_full_matrix_20260615_000508/lab13_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `audit_result`=passed; `best_depth`=18; `claim_allowed`=emotion-specific read/write transfer; causal steering pending hand-labeled generations |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab13_tiera_labs1_25_full_matrix_20260615_000508/lab13_tiera_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `audit_result`=failed; `best_depth`=8; `claim_allowed`=no defended emotion-geometry claim |

## Curated Artifacts

- `olmo3_32b_lab13_olmo32bthink_labs1_25_local_reruns_2026061_emotion_geometry_dashboard.png`
- `olmo3_32b_lab13_olmo32bthink_labs1_25_local_reruns_2026061_emotion_transfer_matrix.png`
- `olmo3_32b_lab13_olmo32bthink_labs1_25_local_reruns_2026061_results.csv`
- `olmo3_32b_lab13_olmo32bthink_labs1_25_local_reruns_2026061_metrics.json`
- `gemma4e4b_lab13_gemma4e4b_labs1_25_local_reruns_20260615_1_emotion_geometry_dashboard.png`
- `gemma4e4b_lab13_gemma4e4b_labs1_25_local_reruns_20260615_1_emotion_transfer_matrix.png`
- `gemma4e4b_lab13_gemma4e4b_labs1_25_local_reruns_20260615_1_results.csv`
- `gemma4e4b_lab13_gemma4e4b_labs1_25_local_reruns_20260615_1_metrics.json`
- `olmo3_7b_lab13_tierc_labs1_25_full_matrix_20260615_000508_emotion_geometry_dashboard.png`
- `olmo3_7b_lab13_tierc_labs1_25_full_matrix_20260615_000508_generation_response_atlas.png`
- `olmo3_7b_lab13_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `olmo3_7b_lab13_tierc_labs1_25_full_matrix_20260615_000508_metrics.json`
- `smollm_lab13_tiera_labs1_25_full_matrix_20260615_000508_emotion_geometry_dashboard.png`
- `smollm_lab13_tiera_labs1_25_full_matrix_20260615_000508_generation_response_atlas.png`
- `smollm_lab13_tiera_labs1_25_full_matrix_20260615_000508_results.csv`
- `smollm_lab13_tiera_labs1_25_full_matrix_20260615_000508_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
