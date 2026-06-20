# Lab 26 Validation

## Lab 26: Causal Abstraction by Residual Resampling

Causal abstraction by residual-stream resampling: formal hypotheses tested with preserving, breaking, random, and wrong-site donors.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/lab26_tierc_full_verify_20260614_2119/lab26_tierc_full_verify_20260614_2119` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `lab_id`=L26, `lab_name`=lab26_causal_abstraction, `n_best_cells`=2, `n_counterexamples`=32, `n_interventions`=19500, `n_summary_cells`=756
  - model: `allenai/Olmo-3-1125-32B` (64 blocks, d_model 5120)
  - data: `causal_abstraction_tasks.csv` sha256 `1037918625043ba9`
  - selected rows: 30 from 30
- `interpret/verify_part3/lab26_olmo32bthink_full_verify_20260614_2150_b24/lab26_olmo32bthink_full_verify_20260614_2150_b24` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `lab_id`=L26, `lab_name`=lab26_causal_abstraction, `n_best_cells`=2, `n_counterexamples`=32, `n_interventions`=19500, `n_summary_cells`=756
  - model: `allenai/Olmo-3-32B-Think` (64 blocks, d_model 5120)
  - data: `causal_abstraction_tasks.csv` sha256 `1037918625043ba9`
  - selected rows: 30 from 30
- `interpret/verify_part3/lab26_tierb_full_verify_20260614_2116/lab26_tierb_full_verify_20260614_2116` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `lab_id`=L26, `lab_name`=lab26_causal_abstraction, `n_best_cells`=2, `n_counterexamples`=32, `n_interventions`=9900, `n_summary_cells`=372
  - model: `allenai/Olmo-3-1025-7B` (32 blocks, d_model 4096)
  - data: `causal_abstraction_tasks.csv` sha256 `1037918625043ba9`
  - selected rows: 30 from 30
- `interpret/verify_part3/lab26_gemma4e4b_full_verify_20260614_2136/lab26_gemma4e4b_full_verify_20260614_2136` (google/gemma-4-E4B-it, tier b)
  - Metrics: `lab_id`=L26, `lab_name`=lab26_causal_abstraction, `n_best_cells`=2, `n_counterexamples`=32, `n_interventions`=12900, `n_summary_cells`=492
  - model: `google/gemma-4-E4B-it` (42 blocks, d_model 2560)
  - data: `causal_abstraction_tasks.csv` sha256 `1037918625043ba9`
  - selected rows: 30 from 30

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/lab26_tierc_full_verify_20260614_2119/lab26_tierc_full_verify_20260614_2119` | `allenai/Olmo-3-1125-32B` | `c` | `lab_id`=L26; `lab_name`=lab26_causal_abstraction; `n_best_cells`=2 |
| `interpret/verify_part3/lab26_olmo32bthink_full_verify_20260614_2150_b24/lab26_olmo32bthink_full_verify_20260614_2150_b24` | `allenai/Olmo-3-32B-Think` | `c` | `lab_id`=L26; `lab_name`=lab26_causal_abstraction; `n_best_cells`=2 |
| `interpret/verify_part3/lab26_tierb_full_verify_20260614_2116/lab26_tierb_full_verify_20260614_2116` | `allenai/Olmo-3-1025-7B` | `b` | `lab_id`=L26; `lab_name`=lab26_causal_abstraction; `n_best_cells`=2 |
| `interpret/verify_part3/lab26_gemma4e4b_full_verify_20260614_2136/lab26_gemma4e4b_full_verify_20260614_2136` | `google/gemma-4-E4B-it` | `b` | `lab_id`=L26; `lab_name`=lab26_causal_abstraction; `n_best_cells`=2 |
| `interpret/verify_part3/lab26_full_verify_20260614_203629/lab26_causal_abstraction-20260614_203629-de40c2` | `gpt2` | `a` | `lab_id`=L26; `lab_name`=lab26_causal_abstraction; `n_best_cells`=2 |

## Curated Artifacts

- `olmo3_32b_lab26_tierc_full_verify_20260614_2119_causal_abstraction_dashboard.png`
- `olmo3_32b_lab26_tierc_full_verify_20260614_2119_resampling_preservation_matrix.png`
- `olmo3_32b_lab26_tierc_full_verify_20260614_2119_results.csv`
- `olmo3_32b_lab26_tierc_full_verify_20260614_2119_metrics.json`
- `olmo3_32b_lab26_olmo32bthink_full_verify_20260614_2150_b24_causal_abstraction_dashboard.png`
- `olmo3_32b_lab26_olmo32bthink_full_verify_20260614_2150_b24_resampling_preservation_matrix.png`
- `olmo3_32b_lab26_olmo32bthink_full_verify_20260614_2150_b24_results.csv`
- `olmo3_32b_lab26_olmo32bthink_full_verify_20260614_2150_b24_metrics.json`
- `olmo3_1025_7b_lab26_tierb_full_verify_20260614_2116_causal_abstraction_dashboard.png`
- `olmo3_1025_7b_lab26_tierb_full_verify_20260614_2116_resampling_preservation_matrix.png`
- `olmo3_1025_7b_lab26_tierb_full_verify_20260614_2116_results.csv`
- `olmo3_1025_7b_lab26_tierb_full_verify_20260614_2116_metrics.json`
- `gemma4e4b_lab26_gemma4e4b_full_verify_20260614_2136_causal_abstraction_dashboard.png`
- `gemma4e4b_lab26_gemma4e4b_full_verify_20260614_2136_resampling_preservation_matrix.png`
- `gemma4e4b_lab26_gemma4e4b_full_verify_20260614_2136_results.csv`
- `gemma4e4b_lab26_gemma4e4b_full_verify_20260614_2136_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
