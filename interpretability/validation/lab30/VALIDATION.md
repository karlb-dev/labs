# Lab 30 Validation

## Lab 30: Cross-Layer Feature Lineage Without Feature-Identity Overclaiming

Cross-layer feature lineage: supervised prototype directions with confusable controls and split-aware evidence.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/lab30_tierc_full_verify_20260614_2226/lab30_tierc_full_verify_20260614_2226` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `failure_specimens_jsonl`=tables/failure_specimens.jsonl, `failure_specimens_md`=tables/failure_specimens.md, `mean_confusable_gap`=0.43, `mean_eval_auc`=0.9413, `mean_lineage_lift_over_random`=0.4966, `n_counterexamples`=451, `n_domains`=8, `n_edge_pair_rows`=128
  - The lab measured domain-label decodability from residual-stream prototype directions, adjacent-depth lineage scores, confusable controls, and a narrow marker-token activation-addition margin.
- `interpret/verify_part3/lab30_olmo32bthink_full_verify_20260614_2241/lab30_olmo32bthink_full_verify_20260614_2241` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `failure_specimens_jsonl`=tables/failure_specimens.jsonl, `failure_specimens_md`=tables/failure_specimens.md, `mean_confusable_gap`=0.4095, `mean_eval_auc`=0.8907, `mean_lineage_lift_over_random`=0.4892, `n_counterexamples`=574, `n_domains`=8, `n_edge_pair_rows`=128
  - The lab measured domain-label decodability from residual-stream prototype directions, adjacent-depth lineage scores, confusable controls, and a narrow marker-token activation-addition margin.
- `interpret/verify_part3/lab30_gemma4e4b_full_verify_20260614_2235/lab30_gemma4e4b_full_verify_20260614_2235` (google/gemma-4-E4B-it, tier c)
  - Metrics: `failure_specimens_jsonl`=tables/failure_specimens.jsonl, `failure_specimens_md`=tables/failure_specimens.md, `mean_confusable_gap`=0.321, `mean_eval_auc`=0.5958, `mean_lineage_lift_over_random`=0.3654, `n_counterexamples`=673, `n_domains`=8, `n_edge_pair_rows`=128
  - The lab measured domain-label decodability from residual-stream prototype directions, adjacent-depth lineage scores, confusable controls, and a narrow marker-token activation-addition margin.
- `interpret/verify_part3/lab30_tierb_full_verify_20260614_2223/lab30_tierb_full_verify_20260614_2223` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `failure_specimens_jsonl`=tables/failure_specimens.jsonl, `failure_specimens_md`=tables/failure_specimens.md, `mean_confusable_gap`=0.4085, `mean_eval_auc`=0.9366, `mean_lineage_lift_over_random`=0.4743, `n_counterexamples`=193, `n_domains`=8, `n_edge_pair_rows`=128
  - The lab measured domain-label decodability from residual-stream prototype directions, adjacent-depth lineage scores, confusable controls, and a narrow marker-token activation-addition margin.

## What This Lab Teaches

- The central lesson is decodability with controls: useful probes must survive selectivity, held-out data, and confound checks.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/lab30_tierc_full_verify_20260614_2226/lab30_tierc_full_verify_20260614_2226` | `allenai/Olmo-3-1125-32B` | `c` | `failure_specimens_jsonl`=tables/failure_specimens.jsonl; `failure_specimens_md`=tables/failure_specimens.md; `mean_confusable_gap`=0.43 |
| `interpret/verify_part3/lab30_olmo32bthink_full_verify_20260614_2241/lab30_olmo32bthink_full_verify_20260614_2241` | `allenai/Olmo-3-32B-Think` | `c` | `failure_specimens_jsonl`=tables/failure_specimens.jsonl; `failure_specimens_md`=tables/failure_specimens.md; `mean_confusable_gap`=0.4095 |
| `interpret/verify_part3/lab30_gemma4e4b_full_verify_20260614_2235/lab30_gemma4e4b_full_verify_20260614_2235` | `google/gemma-4-E4B-it` | `c` | `failure_specimens_jsonl`=tables/failure_specimens.jsonl; `failure_specimens_md`=tables/failure_specimens.md; `mean_confusable_gap`=0.321 |
| `interpret/verify_part3/lab30_tierb_full_verify_20260614_2223/lab30_tierb_full_verify_20260614_2223` | `allenai/Olmo-3-1025-7B` | `b` | `failure_specimens_jsonl`=tables/failure_specimens.jsonl; `failure_specimens_md`=tables/failure_specimens.md; `mean_confusable_gap`=0.4085 |
| `interpret/verify_part3/lab30_tiera_full_verify_20260615_0000/lab30_tiera_full_verify_20260615_0000` | `gpt2` | `a` | `failure_specimens_jsonl`=tables/failure_specimens.jsonl; `failure_specimens_md`=tables/failure_specimens.md; `mean_confusable_gap`=0.2326 |

## Curated Artifacts

- `olmo3_32b_lab30_tierc_full_verify_20260614_2226_overview_dashboard.png`
- `olmo3_32b_lab30_tierc_full_verify_20260614_2226_layer_sweep_heatmap.png`
- `olmo3_32b_lab30_tierc_full_verify_20260614_2226_tables_feature_lineage_evidence_matrix.csv`
- `olmo3_32b_lab30_tierc_full_verify_20260614_2226_tables_feature_lineage_node_scores.csv`
- `olmo3_32b_lab30_olmo32bthink_full_verify_20260614_2241_overview_dashboard.png`
- `olmo3_32b_lab30_olmo32bthink_full_verify_20260614_2241_layer_sweep_heatmap.png`
- `olmo3_32b_lab30_olmo32bthink_full_verify_20260614_2241_tables_feature_lineage_evidence_matrix.csv`
- `olmo3_32b_lab30_olmo32bthink_full_verify_20260614_2241_tables_feature_lineage_node_scores.csv`
- `gemma4e4b_lab30_gemma4e4b_full_verify_20260614_2235_overview_dashboard.png`
- `gemma4e4b_lab30_gemma4e4b_full_verify_20260614_2235_layer_sweep_heatmap.png`
- `gemma4e4b_lab30_gemma4e4b_full_verify_20260614_2235_tables_feature_lineage_evidence_matrix.csv`
- `gemma4e4b_lab30_gemma4e4b_full_verify_20260614_2235_tables_feature_lineage_node_scores.csv`
- `olmo3_1025_7b_lab30_tierb_full_verify_20260614_2223_overview_dashboard.png`
- `olmo3_1025_7b_lab30_tierb_full_verify_20260614_2223_layer_sweep_heatmap.png`
- `olmo3_1025_7b_lab30_tierb_full_verify_20260614_2223_tables_feature_lineage_evidence_matrix.csv`
- `olmo3_1025_7b_lab30_tierb_full_verify_20260614_2223_tables_feature_lineage_node_scores.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
