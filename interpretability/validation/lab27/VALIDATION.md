# Lab 27 Validation

## Lab 27 - Path-Specific Patching and Causal Mediation

Path-specific patching and causal mediation: node effects versus source-to-receiver path proxies.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/lab27_tierc_full_verify_20260614_2205/lab27_tierc_full_verify_20260614_2205` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `claim_ready_domains`=3, `n_behavior_pass_tasks`=9, `n_cell_score_rows`=2160, `n_control_rows`=540, `n_counterexamples`=4, `n_domain_summary_rows`=3, `n_node_rows`=1755, `n_path_rows`=180
  - model: `allenai/Olmo-3-1125-32B`
  - data rows: 9 selected from `path_mediation_tasks.csv`
  - domains: `{'factual_recall': 3, 'induction': 3, 'relation_swap': 3}`
- `interpret/verify_part3/lab27_olmo32bthink_full_verify_20260614_2217/lab27_olmo32bthink_full_verify_20260614_2217` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `claim_ready_domains`=3, `n_behavior_pass_tasks`=9, `n_cell_score_rows`=2160, `n_control_rows`=540, `n_counterexamples`=4, `n_domain_summary_rows`=3, `n_node_rows`=1755, `n_path_rows`=180
  - model: `allenai/Olmo-3-32B-Think`
  - data rows: 9 selected from `path_mediation_tasks.csv`
  - domains: `{'factual_recall': 3, 'induction': 3, 'relation_swap': 3}`
- `interpret/verify_part3/lab27_tierb_full_verify_20260614_2202/lab27_tierb_full_verify_20260614_2202` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `claim_ready_domains`=3, `n_behavior_pass_tasks`=9, `n_cell_score_rows`=2160, `n_control_rows`=540, `n_counterexamples`=4, `n_domain_summary_rows`=3, `n_node_rows`=891, `n_path_rows`=180
  - model: `allenai/Olmo-3-1025-7B`
  - data rows: 9 selected from `path_mediation_tasks.csv`
  - domains: `{'factual_recall': 3, 'induction': 3, 'relation_swap': 3}`
- `interpret/verify_part3/lab27_gemma4e4b_full_verify_20260614_2214/lab27_gemma4e4b_full_verify_20260614_2214` (google/gemma-4-E4B-it, tier b)
  - Metrics: `claim_ready_domains`=1, `n_behavior_pass_tasks`=6, `n_cell_score_rows`=1440, `n_control_rows`=360, `n_counterexamples`=7, `n_domain_summary_rows`=3, `n_node_rows`=774, `n_path_rows`=120
  - model: `google/gemma-4-E4B-it`
  - data rows: 9 selected from `path_mediation_tasks.csv`
  - domains: `{'factual_recall': 3, 'induction': 3, 'relation_swap': 3}`

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/lab27_tierc_full_verify_20260614_2205/lab27_tierc_full_verify_20260614_2205` | `allenai/Olmo-3-1125-32B` | `c` | `claim_ready_domains`=3; `n_behavior_pass_tasks`=9; `n_cell_score_rows`=2160 |
| `interpret/verify_part3/lab27_olmo32bthink_full_verify_20260614_2217/lab27_olmo32bthink_full_verify_20260614_2217` | `allenai/Olmo-3-32B-Think` | `c` | `claim_ready_domains`=3; `n_behavior_pass_tasks`=9; `n_cell_score_rows`=2160 |
| `interpret/verify_part3/lab27_tierb_full_verify_20260614_2202/lab27_tierb_full_verify_20260614_2202` | `allenai/Olmo-3-1025-7B` | `b` | `claim_ready_domains`=3; `n_behavior_pass_tasks`=9; `n_cell_score_rows`=2160 |
| `interpret/verify_part3/lab27_gemma4e4b_full_verify_20260614_2214/lab27_gemma4e4b_full_verify_20260614_2214` | `google/gemma-4-E4B-it` | `b` | `claim_ready_domains`=1; `n_behavior_pass_tasks`=6; `n_cell_score_rows`=1440 |
| `interpret/verify_part3/lab27_full_verify_20260614_203645/lab27_path_mediation-20260614_203645-fa3f8f` | `gpt2` | `a` | `claim_ready_domains`=3; `n_behavior_pass_tasks`=9; `n_cell_score_rows`=624 |

## Curated Artifacts

- `olmo3_32b_lab27_tierc_full_verify_20260614_2205_path_mediation_dashboard.png`
- `olmo3_32b_lab27_tierc_full_verify_20260614_2205_path_specificity_matrix.png`
- `olmo3_32b_lab27_tierc_full_verify_20260614_2205_tables_domain_metric_summary.csv`
- `olmo3_32b_lab27_tierc_full_verify_20260614_2205_results.csv`
- `olmo3_32b_lab27_olmo32bthink_full_verify_20260614_2217_path_mediation_dashboard.png`
- `olmo3_32b_lab27_olmo32bthink_full_verify_20260614_2217_path_specificity_matrix.png`
- `olmo3_32b_lab27_olmo32bthink_full_verify_20260614_2217_tables_domain_metric_summary.csv`
- `olmo3_32b_lab27_olmo32bthink_full_verify_20260614_2217_results.csv`
- `olmo3_1025_7b_lab27_tierb_full_verify_20260614_2202_path_mediation_dashboard.png`
- `olmo3_1025_7b_lab27_tierb_full_verify_20260614_2202_path_specificity_matrix.png`
- `olmo3_1025_7b_lab27_tierb_full_verify_20260614_2202_tables_domain_metric_summary.csv`
- `olmo3_1025_7b_lab27_tierb_full_verify_20260614_2202_results.csv`
- `gemma4e4b_lab27_gemma4e4b_full_verify_20260614_2214_path_mediation_dashboard.png`
- `gemma4e4b_lab27_gemma4e4b_full_verify_20260614_2214_path_specificity_matrix.png`
- `gemma4e4b_lab27_gemma4e4b_full_verify_20260614_2214_tables_domain_metric_summary.csv`
- `gemma4e4b_lab27_gemma4e4b_full_verify_20260614_2214_results.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
