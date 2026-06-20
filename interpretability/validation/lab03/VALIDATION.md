# Lab 03 Validation

## Lab 3: Attention Routing, Induction, and What Heads Actually Do

Attention routing: head motifs, induction, and whether routing matters.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/run6/C/lab3` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `n_examples`=16, `attribution_noise_floor`=0.1, `baseline_repeat_prompt_successes`=12, `baseline_repeat_prompt_total`=13, `n_ablations`=88, `n_candidate_heads_ablated`=11, `n_disagreement_cases`=1724, `n_dropped`=1
  - model: `allenai/Olmo-3-1125-32B` (64 blocks x 40 heads)
  - dtype: `bfloat16` | attention implementation: `eager` (patterns require eager)
  - examples: 16 kept, 1 dropped at the single-token answer gate
- `interpret/run6/B/lab3` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `n_examples`=16, `attribution_noise_floor`=0.1, `baseline_repeat_prompt_successes`=12, `baseline_repeat_prompt_total`=13, `n_ablations`=88, `n_candidate_heads_ablated`=11, `n_disagreement_cases`=848, `n_dropped`=1
  - model: `allenai/Olmo-3-1025-7B` (32 blocks x 32 heads)
  - dtype: `bfloat16` | attention implementation: `eager` (patterns require eager)
  - examples: 16 kept, 1 dropped at the single-token answer gate
- `interpret/verify_part3/labs1_25_fix_reruns_20260615_154344/lab03_gemma4e4b_labs1_25_fix_reruns_20260615_154344/lab03_gemma4e4b_labs1_25_fix_reruns_20260615_154344` (google/gemma-4-E4B-it, tier b)
  - Metrics: `n_examples`=16, `attribution_noise_floor`=0.1, `baseline_repeat_prompt_successes`=13, `baseline_repeat_prompt_total`=13, `n_ablations`=80, `n_candidate_heads_ablated`=10, `n_disagreement_cases`=322, `n_dropped`=1
  - model: `google/gemma-4-E4B-it` (42 blocks x 8 heads)
  - dtype: `bfloat16` | attention implementation: `eager` (patterns require eager)
  - examples: 16 kept, 1 dropped at the single-token answer gate
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab03_tiera_labs1_25_full_matrix_20260615_000508/lab03_tiera_labs1_25_full_matrix_20260615_000508` (gpt2, tier a)
  - Metrics: `n_examples`=4, `attribution_noise_floor`=0.1, `baseline_repeat_prompt_successes`=3, `baseline_repeat_prompt_total`=3, `n_ablations`=66, `n_candidate_heads_ablated`=11, `n_disagreement_cases`=130, `n_dropped`=0
  - model: `gpt2` (12 blocks x 12 heads)
  - dtype: `float32` | attention implementation: `eager` (patterns require eager)
  - examples: 4 kept, 0 dropped at the single-token answer gate

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/run6/C/lab3` | `allenai/Olmo-3-1125-32B` | `c` | `n_examples`=16; `attribution_noise_floor`=0.1; `baseline_repeat_prompt_successes`=12 |
| `interpret/run6/B/lab3` | `allenai/Olmo-3-1025-7B` | `b` | `n_examples`=16; `attribution_noise_floor`=0.1; `baseline_repeat_prompt_successes`=12 |
| `interpret/verify_part3/labs1_25_fix_reruns_20260615_154344/lab03_gemma4e4b_labs1_25_fix_reruns_20260615_154344/lab03_gemma4e4b_labs1_25_fix_reruns_20260615_154344` | `google/gemma-4-E4B-it` | `b` | `n_examples`=16; `attribution_noise_floor`=0.1; `baseline_repeat_prompt_successes`=13 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab03_tiera_labs1_25_full_matrix_20260615_000508/lab03_tiera_labs1_25_full_matrix_20260615_000508` | `gpt2` | `a` | `n_examples`=4; `attribution_noise_floor`=0.1; `baseline_repeat_prompt_successes`=3 |
| `interpret/validate_part2_plots/lab3` | `unknown` | `` | see copied summaries |

## Curated Artifacts

- `olmo3_32b_run6c_routing_evidence_dashboard.png`
- `olmo3_32b_run6c_head_evidence_matrix.png`
- `olmo3_32b_run6c_results.csv`
- `olmo3_32b_run6c_metrics.json`
- `olmo3_1025_7b_run6b_routing_evidence_dashboard.png`
- `olmo3_1025_7b_run6b_head_evidence_matrix.png`
- `olmo3_1025_7b_run6b_results.csv`
- `olmo3_1025_7b_run6b_metrics.json`
- `gemma4e4b_lab03_gemma4e4b_labs1_25_fix_reruns_20260615_154_routing_evidence_dashboard.png`
- `gemma4e4b_lab03_gemma4e4b_labs1_25_fix_reruns_20260615_154_head_evidence_matrix.png`
- `gemma4e4b_lab03_gemma4e4b_labs1_25_fix_reruns_20260615_154_results.csv`
- `gemma4e4b_lab03_gemma4e4b_labs1_25_fix_reruns_20260615_154_metrics.json`
- `gpt2_lab03_tiera_labs1_25_full_matrix_20260615_000508_routing_evidence_dashboard.png`
- `gpt2_lab03_tiera_labs1_25_full_matrix_20260615_000508_ablation_scope_heatmap.png`
- `gpt2_lab03_tiera_labs1_25_full_matrix_20260615_000508_results.csv`
- `gpt2_lab03_tiera_labs1_25_full_matrix_20260615_000508_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
