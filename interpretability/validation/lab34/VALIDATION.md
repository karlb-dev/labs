# Lab 34 Validation

## Lab 34: Tool Use, Agents, and State Tracking

Tool use, agents, and state tracking: toy-tool prompt-boundary probes, traces, controls, and constrained interventions.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/lab34_tierc_full_verify_20260615_0034/lab34_tierc_full_verify_20260615_0034` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `argument_valid_rate`=1, `causal_shift_over_random`=-0.0508, `decode_gap_over_surface`=0, `eval_split_used`=eval, `lab_id`=L34, `lab_name`=lab34_tool_use_state, `n_counterexamples`=40, `random_direction_shift_at_scale_1`=0.0586
  - model: `allenai/Olmo-3-1125-32B`
  - data: `tool_use_tasks.jsonl` sha256 `bcf18742f51e8aa1`
  - selected rows: 48 from 48
- `interpret/verify_part3/lab34_olmo32bthink_full_verify_20260615_0050/lab34_olmo32bthink_full_verify_20260615_0050` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `argument_valid_rate`=1, `causal_shift_over_random`=-0.5977, `decode_gap_over_surface`=0, `eval_split_used`=eval, `lab_id`=L34, `lab_name`=lab34_tool_use_state, `n_counterexamples`=40, `random_direction_shift_at_scale_1`=0.1758
  - model: `allenai/Olmo-3-32B-Think`
  - data: `tool_use_tasks.jsonl` sha256 `bcf18742f51e8aa1`
  - selected rows: 48 from 48
- `interpret/verify_part3/lab34_gemma4e4b_full_verify_20260615_0044/lab34_gemma4e4b_full_verify_20260615_0044` (google/gemma-4-E4B-it, tier c)
  - Metrics: `argument_valid_rate`=1, `causal_shift_over_random`=-0.0681, `decode_gap_over_surface`=-0.1875, `eval_split_used`=eval, `lab_id`=L34, `lab_name`=lab34_tool_use_state, `n_counterexamples`=40, `random_direction_shift_at_scale_1`=-0.1954
  - model: `google/gemma-4-E4B-it`
  - data: `tool_use_tasks.jsonl` sha256 `bcf18742f51e8aa1`
  - selected rows: 48 from 48
- `interpret/verify_part3/lab34_tierb_full_verify_20260615_0026/lab34_tierb_full_verify_20260615_0026` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `argument_valid_rate`=1, `causal_shift_over_random`=0.0625, `decode_gap_over_surface`=0.125, `eval_split_used`=eval, `lab_id`=L34, `lab_name`=lab34_tool_use_state, `n_counterexamples`=40, `random_direction_shift_at_scale_1`=0.1289
  - model: `allenai/Olmo-3-1025-7B`
  - data: `tool_use_tasks.jsonl` sha256 `bcf18742f51e8aa1`
  - selected rows: 48 from 48

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/lab34_tierc_full_verify_20260615_0034/lab34_tierc_full_verify_20260615_0034` | `allenai/Olmo-3-1125-32B` | `c` | `argument_valid_rate`=1; `causal_shift_over_random`=-0.0508; `decode_gap_over_surface`=0 |
| `interpret/verify_part3/lab34_olmo32bthink_full_verify_20260615_0050/lab34_olmo32bthink_full_verify_20260615_0050` | `allenai/Olmo-3-32B-Think` | `c` | `argument_valid_rate`=1; `causal_shift_over_random`=-0.5977; `decode_gap_over_surface`=0 |
| `interpret/verify_part3/lab34_gemma4e4b_full_verify_20260615_0044/lab34_gemma4e4b_full_verify_20260615_0044` | `google/gemma-4-E4B-it` | `c` | `argument_valid_rate`=1; `causal_shift_over_random`=-0.0681; `decode_gap_over_surface`=-0.1875 |
| `interpret/verify_part3/lab34_tierb_full_verify_20260615_0026/lab34_tierb_full_verify_20260615_0026` | `allenai/Olmo-3-1025-7B` | `b` | `argument_valid_rate`=1; `causal_shift_over_random`=0.0625; `decode_gap_over_surface`=0.125 |
| `interpret/verify_part3/lab34_tiera_full_verify_20260615_0020/lab34_tiera_full_verify_20260615_0020` | `gpt2` | `a` | `argument_valid_rate`=1; `causal_shift_over_random`=-0.0122; `decode_gap_over_surface`=-0.5 |

## Curated Artifacts

- `olmo3_32b_lab34_tierc_full_verify_20260615_0034_overview_dashboard.png`
- `olmo3_32b_lab34_tierc_full_verify_20260615_0034_layer_sweep_heatmap.png`
- `olmo3_32b_lab34_tierc_full_verify_20260615_0034_results.csv`
- `olmo3_32b_lab34_tierc_full_verify_20260615_0034_metrics.json`
- `olmo3_32b_lab34_olmo32bthink_full_verify_20260615_0050_overview_dashboard.png`
- `olmo3_32b_lab34_olmo32bthink_full_verify_20260615_0050_layer_sweep_heatmap.png`
- `olmo3_32b_lab34_olmo32bthink_full_verify_20260615_0050_results.csv`
- `olmo3_32b_lab34_olmo32bthink_full_verify_20260615_0050_metrics.json`
- `gemma4e4b_lab34_gemma4e4b_full_verify_20260615_0044_overview_dashboard.png`
- `gemma4e4b_lab34_gemma4e4b_full_verify_20260615_0044_layer_sweep_heatmap.png`
- `gemma4e4b_lab34_gemma4e4b_full_verify_20260615_0044_results.csv`
- `gemma4e4b_lab34_gemma4e4b_full_verify_20260615_0044_metrics.json`
- `olmo3_1025_7b_lab34_tierb_full_verify_20260615_0026_overview_dashboard.png`
- `olmo3_1025_7b_lab34_tierb_full_verify_20260615_0026_layer_sweep_heatmap.png`
- `olmo3_1025_7b_lab34_tierb_full_verify_20260615_0026_results.csv`
- `olmo3_1025_7b_lab34_tierb_full_verify_20260615_0026_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
