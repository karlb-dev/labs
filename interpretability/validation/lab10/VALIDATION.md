# Lab 10 Validation

## Lab 10: Reasoning Models and Chain-of-Thought Faithfulness

CoT faithfulness: hint injection, the necessity curve, add-mistake, and filler controls.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab10_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab10_olmo32bthink_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `n_items`=60, `load_bearing_verdict`=mixed load-bearing evidence, `n_conditions`=6, `n_unparseable_or_forced`=3, `self_report_verdict_auto`=wrong-hint effects were mostly acknowledged or rare (auto-labeled)
  - model: `allenai/Olmo-3-32B-Think`
  - dataset: 60 frozen MCQ items x 6 conditions
  - decoding: greedy, fixed max-new-token budget, deterministic condition prompts
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab10_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab10_gemma4e4b_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `n_items`=36, `load_bearing_verdict`=mixed load-bearing evidence, `n_conditions`=6, `n_unparseable_or_forced`=158, `self_report_verdict_auto`=self-report omits influential hints in a safety-relevant fraction of items (auto-labeled)
  - model: `google/gemma-4-E4B-it`
  - dataset: 36 frozen MCQ items x 6 conditions
  - decoding: greedy, fixed max-new-token budget, deterministic condition prompts
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab10_tierc_labs1_25_full_matrix_20260615_000508/lab10_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Think, tier c)
  - Metrics: `n_items`=60, `load_bearing_verdict`=mixed load-bearing evidence, `n_conditions`=6, `n_unparseable_or_forced`=42, `self_report_verdict_auto`=wrong-hint effects were mostly acknowledged or rare (auto-labeled)
  - model: `allenai/Olmo-3-7B-Think`
  - dataset: 60 frozen MCQ items x 6 conditions
  - decoding: greedy, fixed max-new-token budget, deterministic condition prompts
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab10_tiera_labs1_25_full_matrix_20260615_000508/lab10_tiera_labs1_25_full_matrix_20260615_000508` (Qwen/Qwen3-0.6B, tier a)
  - Metrics: `n_items`=3, `load_bearing_verdict`=mixed load-bearing evidence, `n_conditions`=6, `n_unparseable_or_forced`=6, `self_report_verdict_auto`=no wrong-hint flips observed
  - model: `Qwen/Qwen3-0.6B`
  - dataset: 3 frozen MCQ items x 6 conditions
  - decoding: greedy, fixed max-new-token budget, deterministic condition prompts

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab10_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab10_olmo32bthink_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `n_items`=60; `load_bearing_verdict`=mixed load-bearing evidence; `n_conditions`=6 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab10_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab10_gemma4e4b_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `n_items`=36; `load_bearing_verdict`=mixed load-bearing evidence; `n_conditions`=6 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab10_tierc_labs1_25_full_matrix_20260615_000508/lab10_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Think` | `c` | `n_items`=60; `load_bearing_verdict`=mixed load-bearing evidence; `n_conditions`=6 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab10_tiera_labs1_25_full_matrix_20260615_000508/lab10_tiera_labs1_25_full_matrix_20260615_000508` | `Qwen/Qwen3-0.6B` | `a` | `n_items`=3; `load_bearing_verdict`=mixed load-bearing evidence; `n_conditions`=6 |
| `interpret/tempruns/lab10_cot_faithfulness-20260613_024637-4c0a80` | `allenai/Olmo-3.1-32B-Instruct` | `b` | `n_items`=36; `load_bearing_verdict`=mixed load-bearing evidence; `n_conditions`=6 |

## Curated Artifacts

- `olmo3_32b_lab10_olmo32bthink_labs1_25_local_reruns_2026061_cot_faithfulness_dashboard.png`
- `olmo3_32b_lab10_olmo32bthink_labs1_25_local_reruns_2026061_hint_condition_matrix.png`
- `olmo3_32b_lab10_olmo32bthink_labs1_25_local_reruns_2026061_tables_domain_faithfulness_summary.csv`
- `olmo3_32b_lab10_olmo32bthink_labs1_25_local_reruns_2026061_results.csv`
- `gemma4e4b_lab10_gemma4e4b_labs1_25_local_reruns_20260615_1_cot_faithfulness_dashboard.png`
- `gemma4e4b_lab10_gemma4e4b_labs1_25_local_reruns_20260615_1_hint_condition_matrix.png`
- `gemma4e4b_lab10_gemma4e4b_labs1_25_local_reruns_20260615_1_tables_domain_faithfulness_summary.csv`
- `gemma4e4b_lab10_gemma4e4b_labs1_25_local_reruns_20260615_1_results.csv`
- `olmo3_7b_lab10_tierc_labs1_25_full_matrix_20260615_000508_cot_faithfulness_dashboard.png`
- `olmo3_7b_lab10_tierc_labs1_25_full_matrix_20260615_000508_domain_hint_atlas.png`
- `olmo3_7b_lab10_tierc_labs1_25_full_matrix_20260615_000508_tables_domain_faithfulness_summary.csv`
- `olmo3_7b_lab10_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `qwen3-0.6b_lab10_tiera_labs1_25_full_matrix_20260615_000508_cot_faithfulness_dashboard.png`
- `qwen3-0.6b_lab10_tiera_labs1_25_full_matrix_20260615_000508_domain_hint_atlas.png`
- `qwen3-0.6b_lab10_tiera_labs1_25_full_matrix_20260615_000508_tables_domain_faithfulness_summary.csv`
- `qwen3-0.6b_lab10_tiera_labs1_25_full_matrix_20260615_000508_results.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
