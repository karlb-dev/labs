# Lab 36 Validation

## Lab 36: Severance Report-Channel Verification

Severance report-channel verification: B2 screen, B3 bridge, B4 matched-output source attribution, and B5 insertion detection.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_severance/lab36_olmo31_32b_visual_full` (allenai/Olmo-3.1-32B-Instruct, tier c)
  - Metrics: `n_items`=25, `b3_confidence_delta`=, `b3_entropy_delta`=0.1381, `b4_activation_fresh_accuracy`=0, `b4_activation_source_accuracy`=0, `b5_content_leak_rate`=0.1562, `b5_d_prime_all_insertions`=0.7927, `b5_false_alarm_rate`=0.0294
  - Model: `allenai/Olmo-3.1-32B-Instruct`
  - Mode: `b2,b3,b4,b5,cartography,directions,instrument,patch`
  - Verdict: `no_report_channel_coupling_validated`
- `interpret/verify_severance/lab36_olmo31_32b_refactor_full` (allenai/Olmo-3.1-32B-Instruct, tier c)
  - Metrics: `n_items`=25, `b3_confidence_delta`=, `b3_entropy_delta`=0.1381, `b4_activation_fresh_accuracy`=0, `b4_activation_source_accuracy`=0, `b5_content_leak_rate`=0.1562, `b5_d_prime_all_insertions`=0.7927, `b5_false_alarm_rate`=0.0294
  - Model: `allenai/Olmo-3.1-32B-Instruct`
  - Mode: `b2,b3,b4,b5,cartography,directions,instrument,patch`
  - Verdict: `no_report_channel_coupling_validated`
- `interpret/verify_severance/lab36_olmo3_7b_visual_full` (allenai/Olmo-3-7B-Instruct, tier b)
  - Metrics: `n_items`=25, `b3_confidence_delta`=, `b3_entropy_delta`=0.0812, `b4_activation_fresh_accuracy`=0, `b4_activation_source_accuracy`=0, `b5_content_leak_rate`=0.125, `b5_d_prime_all_insertions`=2.119, `b5_false_alarm_rate`=0.0294
  - Model: `allenai/Olmo-3-7B-Instruct`
  - Mode: `b2,b3,b4,b5,cartography,directions,instrument,patch`
  - Verdict: `no_report_channel_coupling_validated`
- `interpret/verify_severance/lab36_olmo3_7b_refactor_full` (allenai/Olmo-3-7B-Instruct, tier b)
  - Metrics: `n_items`=25, `b3_confidence_delta`=, `b3_entropy_delta`=0.0812, `b4_activation_fresh_accuracy`=0, `b4_activation_source_accuracy`=0, `b5_content_leak_rate`=0.125, `b5_d_prime_all_insertions`=2.119, `b5_false_alarm_rate`=0.0294
  - Model: `allenai/Olmo-3-7B-Instruct`
  - Mode: `b2,b3,b4,b5,cartography,directions,instrument,patch`
  - Verdict: `no_report_channel_coupling_validated`

## What This Lab Teaches

- Read across models and modes: the useful result is the report-channel audit pattern, not a single sample generation.
- The visual/refactor Severance reruns are the most useful artifacts for assessing final behavior.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_severance/lab36_olmo31_32b_visual_full` | `allenai/Olmo-3.1-32B-Instruct` | `c` | `n_items`=25; `b3_confidence_delta`=; `b3_entropy_delta`=0.1381 |
| `interpret/verify_severance/lab36_olmo31_32b_refactor_full` | `allenai/Olmo-3.1-32B-Instruct` | `c` | `n_items`=25; `b3_confidence_delta`=; `b3_entropy_delta`=0.1381 |
| `interpret/verify_severance/lab36_olmo3_7b_visual_full` | `allenai/Olmo-3-7B-Instruct` | `b` | `n_items`=25; `b3_confidence_delta`=; `b3_entropy_delta`=0.0812 |
| `interpret/verify_severance/lab36_olmo3_7b_refactor_full` | `allenai/Olmo-3-7B-Instruct` | `b` | `n_items`=25; `b3_confidence_delta`=; `b3_entropy_delta`=0.0812 |
| `interpret/verify_severance/lab36_smollm_full_visual` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `n_items`=25; `b3_confidence_delta`=; `b3_entropy_delta`=-0.0898 |
| `interpret/verify_severance/lab36_tier_a_smoke_refactor_fixed` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `n_items`=6; `b3_confidence_delta`=; `b3_entropy_delta`= |
| `interpret/verify_severance/lab36_olmo31_32b_full` | `allenai/Olmo-3.1-32B-Instruct` | `c` | `n_items`=12; `b3_confidence_delta`=; `b3_entropy_delta`=0.5143 |
| `interpret/verify_severance/lab36_olmo3_7b_full` | `allenai/Olmo-3-7B-Instruct` | `b` | `n_items`=12; `b3_confidence_delta`=; `b3_entropy_delta`=0.1378 |

## Curated Artifacts

- `olmo31_32b_lab36_olmo31_32b_visual_full_overview_dashboard.png`
- `olmo31_32b_lab36_olmo31_32b_visual_full_source_attribution_control_matrix.png`
- `olmo31_32b_lab36_olmo31_32b_visual_full_layer_sweep_heatmap.png`
- `olmo31_32b_lab36_olmo31_32b_visual_full_results.csv`
- `olmo31_32b_lab36_olmo31_32b_visual_full_metrics.json`
- `olmo31_32b_lab36_olmo31_32b_visual_full_method_card.md`
- `olmo31_32b_lab36_olmo31_32b_refactor_full_severance_dashboard.png`
- `olmo31_32b_lab36_olmo31_32b_refactor_full_b5_detection_margins.png`
- `olmo31_32b_lab36_olmo31_32b_refactor_full_results.csv`
- `olmo31_32b_lab36_olmo31_32b_refactor_full_metrics.json`
- `olmo31_32b_lab36_olmo31_32b_refactor_full_method_card.md`
- `olmo3_7b_lab36_olmo3_7b_visual_full_overview_dashboard.png`
- `olmo3_7b_lab36_olmo3_7b_visual_full_source_attribution_control_matrix.png`
- `olmo3_7b_lab36_olmo3_7b_visual_full_layer_sweep_heatmap.png`
- `olmo3_7b_lab36_olmo3_7b_visual_full_results.csv`
- `olmo3_7b_lab36_olmo3_7b_visual_full_metrics.json`
- `olmo3_7b_lab36_olmo3_7b_visual_full_method_card.md`
- `olmo3_7b_lab36_olmo3_7b_refactor_full_severance_dashboard.png`
- `olmo3_7b_lab36_olmo3_7b_refactor_full_b5_detection_margins.png`
- `olmo3_7b_lab36_olmo3_7b_refactor_full_results.csv`
- `olmo3_7b_lab36_olmo3_7b_refactor_full_metrics.json`
- `olmo3_7b_lab36_olmo3_7b_refactor_full_method_card.md`
- `smollm_lab36_smollm_full_visual_overview_dashboard.png`
- `smollm_lab36_smollm_full_visual_source_attribution_control_matrix.png`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
