# Lab 20 Validation

## Lab 20 - Building Benign Model Organisms

Building benign model organisms: sealed answer keys, manifests, and baseline spillover audits.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab20_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab20_olmo32bthink_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `adapter_training_mode`=recipe_only_default, `max_baseline_spillover_rate`=1, `n_baseline_preexisting_marker_risks`=2, `n_behavior_rows`=30, `n_eval_prompts_private`=30, `n_organisms`=5, `n_public_package_leaks`=0, `n_public_packages`=5
  - Base model: `allenai/Olmo-3-32B-Think`
  - Organisms specified: 5
  - Training examples written: 35
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab20_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab20_gemma4e4b_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `adapter_training_mode`=recipe_only_default, `max_baseline_spillover_rate`=1, `n_baseline_preexisting_marker_risks`=0, `n_behavior_rows`=30, `n_eval_prompts_private`=30, `n_organisms`=5, `n_public_package_leaks`=0, `n_public_packages`=5
  - Base model: `google/gemma-4-E4B-it`
  - Organisms specified: 5
  - Training examples written: 35
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab20_tierc_labs1_25_full_matrix_20260615_000508/lab20_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `adapter_training_mode`=recipe_only_default, `max_baseline_spillover_rate`=1, `n_baseline_preexisting_marker_risks`=0, `n_behavior_rows`=30, `n_eval_prompts_private`=30, `n_organisms`=5, `n_public_package_leaks`=0, `n_public_packages`=5
  - Base model: `allenai/Olmo-3-7B-Instruct`
  - Organisms specified: 5
  - Training examples written: 35
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab20_tiera_labs1_25_full_matrix_20260615_000508/lab20_tiera_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `adapter_training_mode`=recipe_only_default, `max_baseline_spillover_rate`=0, `n_baseline_preexisting_marker_risks`=1, `n_behavior_rows`=19, `n_eval_prompts_private`=24, `n_organisms`=4, `n_public_package_leaks`=0, `n_public_packages`=4
  - Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - Organisms specified: 4
  - Training examples written: 29

## What This Lab Teaches

- The lab is best read through its run summary and dashboard artifacts: inspect the measured claim before trusting the intuition.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab20_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab20_olmo32bthink_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `adapter_training_mode`=recipe_only_default; `max_baseline_spillover_rate`=1; `n_baseline_preexisting_marker_risks`=2 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab20_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab20_gemma4e4b_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `adapter_training_mode`=recipe_only_default; `max_baseline_spillover_rate`=1; `n_baseline_preexisting_marker_risks`=0 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab20_tierc_labs1_25_full_matrix_20260615_000508/lab20_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `adapter_training_mode`=recipe_only_default; `max_baseline_spillover_rate`=1; `n_baseline_preexisting_marker_risks`=0 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab20_tiera_labs1_25_full_matrix_20260615_000508/lab20_tiera_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `adapter_training_mode`=recipe_only_default; `max_baseline_spillover_rate`=0; `n_baseline_preexisting_marker_risks`=1 |

## Curated Artifacts

- `olmo3_32b_lab20_olmo32bthink_labs1_25_local_reruns_2026061_organism_construction_dashboard.png`
- `olmo3_32b_lab20_olmo32bthink_labs1_25_local_reruns_2026061_construction_evidence_dashboard.png`
- `olmo3_32b_lab20_olmo32bthink_labs1_25_local_reruns_2026061_tables_organism_readiness_scorecard.csv`
- `olmo3_32b_lab20_olmo32bthink_labs1_25_local_reruns_2026061_results.csv`
- `gemma4e4b_lab20_gemma4e4b_labs1_25_local_reruns_20260615_1_organism_construction_dashboard.png`
- `gemma4e4b_lab20_gemma4e4b_labs1_25_local_reruns_20260615_1_construction_evidence_dashboard.png`
- `gemma4e4b_lab20_gemma4e4b_labs1_25_local_reruns_20260615_1_tables_organism_readiness_scorecard.csv`
- `gemma4e4b_lab20_gemma4e4b_labs1_25_local_reruns_20260615_1_results.csv`
- `olmo3_7b_lab20_tierc_labs1_25_full_matrix_20260615_000508_organism_construction_dashboard.png`
- `olmo3_7b_lab20_tierc_labs1_25_full_matrix_20260615_000508_construction_evidence_dashboard.png`
- `olmo3_7b_lab20_tierc_labs1_25_full_matrix_20260615_000508_tables_organism_readiness_scorecard.csv`
- `olmo3_7b_lab20_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `smollm_lab20_tiera_labs1_25_full_matrix_20260615_000508_organism_construction_dashboard.png`
- `smollm_lab20_tiera_labs1_25_full_matrix_20260615_000508_construction_evidence_dashboard.png`
- `smollm_lab20_tiera_labs1_25_full_matrix_20260615_000508_tables_organism_readiness_scorecard.csv`
- `smollm_lab20_tiera_labs1_25_full_matrix_20260615_000508_results.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
