# Lab 16 Validation

## Lab 16 - Sycophancy and User-Belief Modeling

Sycophancy and user-belief modeling: truth, user-belief, agreement, and politeness directions.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab16_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab16_olmo32bthink_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `agreement_politeness_projection_correlation`=-0.6142, `agreement_shuffled_steering_sycophancy_delta_max_dose`=0.2, `agreement_steering_specificity_gap_max_dose`=-0.0667, `agreement_steering_sycophancy_delta_max_dose`=0.2, `data_source`=frozen_csv, `false_pressure_correct_rate`=0.0875, `false_pressure_correct_rate_given_neutral_correct`=0.14, `false_pressure_sycophancy_rate`=0.3937
  - Model: `allenai/Olmo-3-32B-Think`
  - Data source: `frozen_csv`
  - Rows: 240 from 40 base facts
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab16_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab16_gemma4e4b_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `agreement_politeness_projection_correlation`=-0.7401, `agreement_shuffled_steering_sycophancy_delta_max_dose`=0.3333, `agreement_steering_specificity_gap_max_dose`=0, `agreement_steering_sycophancy_delta_max_dose`=0.3333, `data_source`=frozen_csv, `false_pressure_correct_rate`=0.575, `false_pressure_correct_rate_given_neutral_correct`=0.5855, `false_pressure_sycophancy_rate`=0.0125
  - Model: `google/gemma-4-E4B-it`
  - Data source: `frozen_csv`
  - Rows: 240 from 40 base facts
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab16_tierc_labs1_25_full_matrix_20260615_000508/lab16_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `agreement_politeness_projection_correlation`=-0.7302, `agreement_shuffled_steering_sycophancy_delta_max_dose`=0.4, `agreement_steering_specificity_gap_max_dose`=0, `agreement_steering_sycophancy_delta_max_dose`=0.4, `data_source`=frozen_csv, `false_pressure_correct_rate`=0.525, `false_pressure_correct_rate_given_neutral_correct`=0.5203, `false_pressure_sycophancy_rate`=0.025
  - Model: `allenai/Olmo-3-7B-Instruct`
  - Data source: `frozen_csv`
  - Rows: 240 from 40 base facts
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab16_tiera_labs1_25_full_matrix_20260615_000508/lab16_tiera_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `agreement_politeness_projection_correlation`=0.1924, `agreement_shuffled_steering_sycophancy_delta_max_dose`=-0.8, `agreement_steering_specificity_gap_max_dose`=-0.1333, `agreement_steering_sycophancy_delta_max_dose`=-0.8, `data_source`=frozen_csv, `false_pressure_correct_rate`=0.325, `false_pressure_correct_rate_given_neutral_correct`=0.4062, `false_pressure_sycophancy_rate`=0.525
  - Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - Data source: `frozen_csv`
  - Rows: 60 from 10 base facts

## What This Lab Teaches

- The central lesson is decodability with controls: useful probes must survive selectivity, held-out data, and confound checks.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab16_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab16_olmo32bthink_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `agreement_politeness_projection_correlation`=-0.6142; `agreement_shuffled_steering_sycophancy_delta_max_dose`=0.2; `agreement_steering_specificity_gap_max_dose`=-0.0667 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab16_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab16_gemma4e4b_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `agreement_politeness_projection_correlation`=-0.7401; `agreement_shuffled_steering_sycophancy_delta_max_dose`=0.3333; `agreement_steering_specificity_gap_max_dose`=0 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab16_tierc_labs1_25_full_matrix_20260615_000508/lab16_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `agreement_politeness_projection_correlation`=-0.7302; `agreement_shuffled_steering_sycophancy_delta_max_dose`=0.4; `agreement_steering_specificity_gap_max_dose`=0 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab16_tiera_labs1_25_full_matrix_20260615_000508/lab16_tiera_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `agreement_politeness_projection_correlation`=0.1924; `agreement_shuffled_steering_sycophancy_delta_max_dose`=-0.8; `agreement_steering_specificity_gap_max_dose`=-0.1333 |

## Curated Artifacts

- `olmo3_32b_lab16_olmo32bthink_labs1_25_local_reruns_2026061_sycophancy_evidence_dashboard.png`
- `olmo3_32b_lab16_olmo32bthink_labs1_25_local_reruns_2026061_social_state_evidence_matrix.png`
- `olmo3_32b_lab16_olmo32bthink_labs1_25_local_reruns_2026061_tables_domain_condition_summary.csv`
- `olmo3_32b_lab16_olmo32bthink_labs1_25_local_reruns_2026061_results.csv`
- `gemma4e4b_lab16_gemma4e4b_labs1_25_local_reruns_20260615_1_sycophancy_evidence_dashboard.png`
- `gemma4e4b_lab16_gemma4e4b_labs1_25_local_reruns_20260615_1_social_state_evidence_matrix.png`
- `gemma4e4b_lab16_gemma4e4b_labs1_25_local_reruns_20260615_1_tables_domain_condition_summary.csv`
- `gemma4e4b_lab16_gemma4e4b_labs1_25_local_reruns_20260615_1_results.csv`
- `olmo3_7b_lab16_tierc_labs1_25_full_matrix_20260615_000508_sycophancy_evidence_dashboard.png`
- `olmo3_7b_lab16_tierc_labs1_25_full_matrix_20260615_000508_probe_control_gap_atlas.png`
- `olmo3_7b_lab16_tierc_labs1_25_full_matrix_20260615_000508_tables_domain_condition_summary.csv`
- `olmo3_7b_lab16_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `smollm_lab16_tiera_labs1_25_full_matrix_20260615_000508_sycophancy_evidence_dashboard.png`
- `smollm_lab16_tiera_labs1_25_full_matrix_20260615_000508_probe_control_gap_atlas.png`
- `smollm_lab16_tiera_labs1_25_full_matrix_20260615_000508_tables_domain_condition_summary.csv`
- `smollm_lab16_tiera_labs1_25_full_matrix_20260615_000508_results.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
