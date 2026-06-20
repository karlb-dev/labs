# Lab 25 Validation

## Lab 25 - Find the Wire

Find the wire: injected concept states, self-report grounding, and source attribution.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab25_olmo32bthink_both_labs1_25_local_reruns_20260615_101609/lab25_olmo32bthink_both_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `n_items`=6, `certainty_bridge_status`=compatible, `grounding_pass_rate`=0.8125, `max_detection_slope`=0, `mean_control_floor`=1, `mean_selected_eval_gap`=0.1493, `mode`=both, `n_confidence_rows`=24
  - Mode: `both`
  - Model: `allenai/Olmo-3-32B-Think`
  - Items: 6 selected from `/content/labs/interpretability/data/introspection_queries.csv`
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab25_gemma4e4b_both_labs1_25_local_reruns_20260615_101609/lab25_gemma4e4b_both_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `n_items`=6, `certainty_bridge_status`=incompatible_d_model:5120!=2560, `grounding_pass_rate`=0, `max_detection_slope`=0, `mean_control_floor`=0, `mean_selected_eval_gap`=0.9987, `mode`=both, `n_confidence_rows`=0
  - Mode: `both`
  - Model: `google/gemma-4-E4B-it`
  - Items: 6 selected from `/content/labs/interpretability/data/introspection_queries.csv`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab25_tierc_both_labs1_25_full_matrix_20260615_000508/lab25_tierc_both_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `n_items`=6, `certainty_bridge_status`=compatible, `grounding_pass_rate`=0.1667, `max_detection_slope`=0, `mean_control_floor`=0.3333, `mean_selected_eval_gap`=0.0621, `mode`=both, `n_confidence_rows`=24
  - Mode: `both`
  - Model: `allenai/Olmo-3-7B-Instruct`
  - Items: 6 selected from `/content/labs/interpretability/data/introspection_queries.csv`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab25_tiera_both_labs1_25_full_matrix_20260615_000508/lab25_tiera_both_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `n_items`=1, `certainty_bridge_status`=incompatible_d_model:4096!=576, `grounding_pass_rate`=1, `max_detection_slope`=0, `mean_control_floor`=1, `mean_selected_eval_gap`=0.6291, `mode`=both, `n_confidence_rows`=0
  - Mode: `both`
  - Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - Items: 1 selected from `/content/labs/interpretability/data/introspection_queries.csv`

## What This Lab Teaches

- The central lesson is intervention hygiene: a direction or feature is only useful when benefits beat matched controls and side effects.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab25_olmo32bthink_both_labs1_25_local_reruns_20260615_101609/lab25_olmo32bthink_both_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `n_items`=6; `certainty_bridge_status`=compatible; `grounding_pass_rate`=0.8125 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab25_gemma4e4b_both_labs1_25_local_reruns_20260615_101609/lab25_gemma4e4b_both_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `n_items`=6; `certainty_bridge_status`=incompatible_d_model:5120!=2560; `grounding_pass_rate`=0 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab25_tierc_both_labs1_25_full_matrix_20260615_000508/lab25_tierc_both_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `n_items`=6; `certainty_bridge_status`=compatible; `grounding_pass_rate`=0.1667 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab25_tiera_both_labs1_25_full_matrix_20260615_000508/lab25_tiera_both_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `n_items`=1; `certainty_bridge_status`=incompatible_d_model:4096!=576; `grounding_pass_rate`=1 |

## Curated Artifacts

- `olmo3_32b_lab25_olmo32bthink_both_labs1_25_local_reruns_20_source_attribution_matrix.png`
- `olmo3_32b_lab25_olmo32bthink_both_labs1_25_local_reruns_20_find_the_wire_dashboard.png`
- `olmo3_32b_lab25_olmo32bthink_both_labs1_25_local_reruns_20_tables_report_discipline_scorecard.csv`
- `olmo3_32b_lab25_olmo32bthink_both_labs1_25_local_reruns_20_results.csv`
- `gemma4e4b_lab25_gemma4e4b_both_labs1_25_local_reruns_20260_source_attribution_matrix.png`
- `gemma4e4b_lab25_gemma4e4b_both_labs1_25_local_reruns_20260_find_the_wire_dashboard.png`
- `gemma4e4b_lab25_gemma4e4b_both_labs1_25_local_reruns_20260_tables_report_discipline_scorecard.csv`
- `gemma4e4b_lab25_gemma4e4b_both_labs1_25_local_reruns_20260_results.csv`
- `olmo3_7b_lab25_tierc_both_labs1_25_full_matrix_20260615_0_find_the_wire_dashboard.png`
- `olmo3_7b_lab25_tierc_both_labs1_25_full_matrix_20260615_0_grounding_risk_atlas.png`
- `olmo3_7b_lab25_tierc_both_labs1_25_full_matrix_20260615_0_tables_report_discipline_scorecard.csv`
- `olmo3_7b_lab25_tierc_both_labs1_25_full_matrix_20260615_0_results.csv`
- `smollm_lab25_tiera_both_labs1_25_full_matrix_20260615_0_find_the_wire_dashboard.png`
- `smollm_lab25_tiera_both_labs1_25_full_matrix_20260615_0_grounding_risk_atlas.png`
- `smollm_lab25_tiera_both_labs1_25_full_matrix_20260615_0_tables_report_discipline_scorecard.csv`
- `smollm_lab25_tiera_both_labs1_25_full_matrix_20260615_0_results.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
