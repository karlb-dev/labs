# Lab 31 Validation

## Lab 31: Automated Interpretability At Scale

Automated interpretability at scale: offline feature-label generation, held-out tests, calibration, abstention, and human-review queues.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/lab31_tierc_full_verify_20260614_2255/lab31_tierc_full_verify_20260614_2255` (gpt2, tier c)
  - Metrics: `best_method`=test_aware, `control_mean_auc`=0.4773, `mean_calibration_error`=0.3571, `n_calibration_bins`=16, `n_counterexamples`=38, `n_explanations`=60, `n_failure_specimens`=24, `n_methods`=5
  - data rows: 12 selected from `auto_interp_feature_tasks.jsonl`
  - data source: `frozen_jsonl`
  - feature types: `{'monosemantic_gold': 8, 'polysemantic_gold': 2, 'random_control': 2}`
- `interpret/verify_part3/lab31_tierb_full_verify_20260614_2252/lab31_tierb_full_verify_20260614_2252` (gpt2, tier b)
  - Metrics: `best_method`=test_aware, `control_mean_auc`=0.4773, `mean_calibration_error`=0.3571, `n_calibration_bins`=16, `n_counterexamples`=38, `n_explanations`=60, `n_failure_specimens`=24, `n_methods`=5
  - data rows: 12 selected from `auto_interp_feature_tasks.jsonl`
  - data source: `frozen_jsonl`
  - feature types: `{'monosemantic_gold': 8, 'polysemantic_gold': 2, 'random_control': 2}`
- `interpret/verify_part3/lab31_tiera_full_verify_20260614_2250/lab31_tiera_full_verify_20260614_2250` (gpt2, tier a)
  - Metrics: `best_method`=test_aware, `control_mean_auc`=0.5556, `mean_calibration_error`=0.3306, `n_calibration_bins`=16, `n_counterexamples`=34, `n_explanations`=50, `n_failure_specimens`=24, `n_methods`=5
  - data rows: 10 selected from `auto_interp_feature_tasks.jsonl`
  - data source: `frozen_jsonl`
  - feature types: `{'monosemantic_gold': 6, 'polysemantic_gold': 2, 'random_control': 2}`

## What This Lab Teaches

- The central lesson is decodability with controls: useful probes must survive selectivity, held-out data, and confound checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/lab31_tierc_full_verify_20260614_2255/lab31_tierc_full_verify_20260614_2255` | `gpt2` | `c` | `best_method`=test_aware; `control_mean_auc`=0.4773; `mean_calibration_error`=0.3571 |
| `interpret/verify_part3/lab31_tierb_full_verify_20260614_2252/lab31_tierb_full_verify_20260614_2252` | `gpt2` | `b` | `best_method`=test_aware; `control_mean_auc`=0.4773; `mean_calibration_error`=0.3571 |
| `interpret/verify_part3/lab31_tiera_full_verify_20260614_2250/lab31_tiera_full_verify_20260614_2250` | `gpt2` | `a` | `best_method`=test_aware; `control_mean_auc`=0.5556; `mean_calibration_error`=0.3306 |

## Curated Artifacts

- `gpt2_lab31_tierc_full_verify_20260614_2255_auto_interp_dashboard.png`
- `gpt2_lab31_tierc_full_verify_20260614_2255_explanation_quality_matrix.png`
- `gpt2_lab31_tierc_full_verify_20260614_2255_tables_suite_score_summary.csv`
- `gpt2_lab31_tierc_full_verify_20260614_2255_tables_feature_evidence_matrix.csv`
- `gpt2_lab31_tierb_full_verify_20260614_2252_auto_interp_dashboard.png`
- `gpt2_lab31_tierb_full_verify_20260614_2252_explanation_quality_matrix.png`
- `gpt2_lab31_tierb_full_verify_20260614_2252_tables_suite_score_summary.csv`
- `gpt2_lab31_tierb_full_verify_20260614_2252_tables_feature_evidence_matrix.csv`
- `gpt2_lab31_tiera_full_verify_20260614_2250_auto_interp_dashboard.png`
- `gpt2_lab31_tiera_full_verify_20260614_2250_explanation_quality_matrix.png`
- `gpt2_lab31_tiera_full_verify_20260614_2250_tables_suite_score_summary.csv`
- `gpt2_lab31_tiera_full_verify_20260614_2250_tables_feature_evidence_matrix.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
