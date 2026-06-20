# Lab 04 Validation

## Lab 4: Probing Without Fooling Yourself, now featuring truth

Probing with controls: what is linearly decodable, and is it selective?

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/run6/C/lab4` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `best_cross_family_layer`=44, `best_min_cross_accuracy`=0.5208, `majority_baseline_mean`=0.5, `mass_mean_peak_shuffled_control`=0.4271, `n_statements`=440, `normalization`=row_unit_norm, `per_family_cap`=0, `surface_letter`=n
  - model: `allenai/Olmo-3-1125-32B` (64 blocks, d_model 5120)
  - statements: 440 across 4 frozen families (per-family cap none)
  - probe position: final token | probes: logistic (LBFGS, L2=0.01) + mass-mean
- `interpret/run6/B/lab4` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `best_cross_family_layer`=32, `best_min_cross_accuracy`=0.6429, `majority_baseline_mean`=0.5, `mass_mean_peak_shuffled_control`=0.506, `n_statements`=440, `normalization`=row_unit_norm, `per_family_cap`=0, `surface_letter`=n
  - model: `allenai/Olmo-3-1025-7B` (32 blocks, d_model 4096)
  - statements: 440 across 4 frozen families (per-family cap none)
  - probe position: final token | probes: logistic (LBFGS, L2=0.01) + mass-mean
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab04_tierc_labs1_25_full_matrix_20260615_000508/lab04_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `best_cross_family_layer`=44, `best_min_cross_accuracy`=0.5278, `majority_baseline_mean`=0.5, `mass_mean_peak_shuffled_control`=0.4271, `n_statements`=440, `normalization`=row_unit_norm, `per_family_cap`=0, `surface_letter`=n
  - model: `allenai/Olmo-3-1125-32B` (64 blocks, d_model 5120)
  - statements: 440 across 4 frozen families (per-family cap none)
  - probe position: final token | probes: logistic (LBFGS, L2=0.01) + mass-mean
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab04_gemma4e4b_labs1_25_full_matrix_20260615_000508/lab04_gemma4e4b_labs1_25_full_matrix_20260615_000508` (google/gemma-4-E4B-it, tier b)
  - Metrics: `best_cross_family_layer`=40, `best_min_cross_accuracy`=0.5179, `majority_baseline_mean`=0.5, `mass_mean_peak_shuffled_control`=0.501, `n_statements`=440, `normalization`=row_unit_norm, `per_family_cap`=0, `surface_letter`=n
  - model: `google/gemma-4-E4B-it` (42 blocks, d_model 2560)
  - statements: 440 across 4 frozen families (per-family cap none)
  - probe position: final token | probes: logistic (LBFGS, L2=0.01) + mass-mean

## What This Lab Teaches

- The central lesson is decodability with controls: useful probes must survive selectivity, held-out data, and confound checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/run6/C/lab4` | `allenai/Olmo-3-1125-32B` | `c` | `best_cross_family_layer`=44; `best_min_cross_accuracy`=0.5208; `majority_baseline_mean`=0.5 |
| `interpret/run6/B/lab4` | `allenai/Olmo-3-1025-7B` | `b` | `best_cross_family_layer`=32; `best_min_cross_accuracy`=0.6429; `majority_baseline_mean`=0.5 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab04_tierc_labs1_25_full_matrix_20260615_000508/lab04_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-1125-32B` | `c` | `best_cross_family_layer`=44; `best_min_cross_accuracy`=0.5278; `majority_baseline_mean`=0.5 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab04_gemma4e4b_labs1_25_full_matrix_20260615_000508/lab04_gemma4e4b_labs1_25_full_matrix_20260615_000508` | `google/gemma-4-E4B-it` | `b` | `best_cross_family_layer`=40; `best_min_cross_accuracy`=0.5179; `majority_baseline_mean`=0.5 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab04_tiera_labs1_25_full_matrix_20260615_000508/lab04_tiera_labs1_25_full_matrix_20260615_000508` | `gpt2` | `a` | `best_cross_family_layer`=6; `best_min_cross_accuracy`=0.5; `majority_baseline_mean`=0.5 |

## Curated Artifacts

- `olmo3_32b_run6c_probe_evidence_dashboard.png`
- `olmo3_32b_run6c_probe_evidence_matrix.png`
- `olmo3_32b_run6c_results.csv`
- `olmo3_32b_run6c_metrics.json`
- `olmo3_1025_7b_run6b_probe_evidence_dashboard.png`
- `olmo3_1025_7b_run6b_probe_evidence_matrix.png`
- `olmo3_1025_7b_run6b_results.csv`
- `olmo3_1025_7b_run6b_metrics.json`
- `olmo3_32b_lab04_tierc_labs1_25_full_matrix_20260615_000508_probe_evidence_dashboard.png`
- `olmo3_32b_lab04_tierc_labs1_25_full_matrix_20260615_000508_family_depth_atlas.png`
- `olmo3_32b_lab04_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `olmo3_32b_lab04_tierc_labs1_25_full_matrix_20260615_000508_metrics.json`
- `gemma4e4b_lab04_gemma4e4b_labs1_25_full_matrix_20260615_00_probe_evidence_dashboard.png`
- `gemma4e4b_lab04_gemma4e4b_labs1_25_full_matrix_20260615_00_family_depth_atlas.png`
- `gemma4e4b_lab04_gemma4e4b_labs1_25_full_matrix_20260615_00_results.csv`
- `gemma4e4b_lab04_gemma4e4b_labs1_25_full_matrix_20260615_00_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
