# Lab 15 Validation

## Lab 15: Multi-Turn Instrumentation Harness

Multi-turn instrumentation: chat-template spans, cache parity, patch no-op, and null traces.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab15_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab15_olmo32bthink_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `bench_chat_template_labs_has_lab15`=True, `bench_registry_has_lab15`=True, `cache_parity_atol`=0.002, `cache_parity_max_abs_hidden_diff`=1.5, `cache_parity_max_abs_logit_diff`=0.3125, `cache_parity_ok`=True, `chat_hook_parity_max_abs_diff`=0, `chat_hook_parity_ok`=True
  - model: `allenai/Olmo-3-32B-Think`
  - trace depth: 32
  - evidence level: `OBS`, instrumentation validation only
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab15_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab15_gemma4e4b_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `bench_chat_template_labs_has_lab15`=True, `bench_registry_has_lab15`=True, `cache_parity_atol`=0.002, `cache_parity_max_abs_hidden_diff`=3, `cache_parity_max_abs_logit_diff`=0.75, `cache_parity_ok`=True, `chat_hook_parity_max_abs_diff`=0, `chat_hook_parity_ok`=True
  - model: `google/gemma-4-E4B-it`
  - trace depth: 21
  - evidence level: `OBS`, instrumentation validation only
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab15_tierc_labs1_25_full_matrix_20260615_000508/lab15_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `bench_chat_template_labs_has_lab15`=True, `bench_registry_has_lab15`=True, `cache_parity_atol`=0.002, `cache_parity_max_abs_hidden_diff`=0.2031, `cache_parity_max_abs_logit_diff`=0.3594, `cache_parity_ok`=True, `chat_hook_parity_max_abs_diff`=0, `chat_hook_parity_ok`=True
  - model: `allenai/Olmo-3-7B-Instruct`
  - trace depth: 16
  - evidence level: `OBS`, instrumentation validation only
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab15_tiera_labs1_25_full_matrix_20260615_000508/lab15_tiera_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `bench_chat_template_labs_has_lab15`=True, `bench_registry_has_lab15`=True, `cache_parity_atol`=0.002, `cache_parity_max_abs_hidden_diff`=0.000954, `cache_parity_max_abs_logit_diff`=5.7e-05, `cache_parity_ok`=True, `chat_hook_parity_max_abs_diff`=0, `chat_hook_parity_ok`=True
  - model: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - trace depth: 15
  - evidence level: `OBS`, instrumentation validation only

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab15_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab15_olmo32bthink_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `bench_chat_template_labs_has_lab15`=True; `bench_registry_has_lab15`=True; `cache_parity_atol`=0.002 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab15_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab15_gemma4e4b_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `bench_chat_template_labs_has_lab15`=True; `bench_registry_has_lab15`=True; `cache_parity_atol`=0.002 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab15_tierc_labs1_25_full_matrix_20260615_000508/lab15_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `bench_chat_template_labs_has_lab15`=True; `bench_registry_has_lab15`=True; `cache_parity_atol`=0.002 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab15_tiera_labs1_25_full_matrix_20260615_000508/lab15_tiera_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `bench_chat_template_labs_has_lab15`=True; `bench_registry_has_lab15`=True; `cache_parity_atol`=0.002 |

## Curated Artifacts

- `olmo3_32b_lab15_olmo32bthink_labs1_25_local_reruns_2026061_harness_evidence_dashboard.png`
- `olmo3_32b_lab15_olmo32bthink_labs1_25_local_reruns_2026061_harness_evidence_matrix.png`
- `olmo3_32b_lab15_olmo32bthink_labs1_25_local_reruns_2026061_results.csv`
- `olmo3_32b_lab15_olmo32bthink_labs1_25_local_reruns_2026061_metrics.json`
- `gemma4e4b_lab15_gemma4e4b_labs1_25_local_reruns_20260615_1_harness_evidence_dashboard.png`
- `gemma4e4b_lab15_gemma4e4b_labs1_25_local_reruns_20260615_1_harness_evidence_matrix.png`
- `gemma4e4b_lab15_gemma4e4b_labs1_25_local_reruns_20260615_1_results.csv`
- `gemma4e4b_lab15_gemma4e4b_labs1_25_local_reruns_20260615_1_metrics.json`
- `olmo3_7b_lab15_tierc_labs1_25_full_matrix_20260615_000508_harness_evidence_dashboard.png`
- `olmo3_7b_lab15_tierc_labs1_25_full_matrix_20260615_000508_depth_selection_atlas.png`
- `olmo3_7b_lab15_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `olmo3_7b_lab15_tierc_labs1_25_full_matrix_20260615_000508_metrics.json`
- `smollm_lab15_tiera_labs1_25_full_matrix_20260615_000508_harness_evidence_dashboard.png`
- `smollm_lab15_tiera_labs1_25_full_matrix_20260615_000508_depth_selection_atlas.png`
- `smollm_lab15_tiera_labs1_25_full_matrix_20260615_000508_results.csv`
- `smollm_lab15_tiera_labs1_25_full_matrix_20260615_000508_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
