# Lab 05 Validation

## Lab 5: Activation Patching and Causal Tracing

Activation patching and causal tracing: where is a fact causally recovered?

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/run6/C/lab5` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `edit_enabled`=False, `matched_top_patch_mean_recovery`=0.5443, `n_candidate_facts`=25, `n_facts_kept_base`=25, `n_pairs_rejected_alignment`=0
  - model: `allenai/Olmo-3-1125-32B` (64 blocks, d_model 5120)
  - facts: 25 base-template pairs past the gate (0 dropped, 0 rejected by the alignment validator)
  - grid: 65 stream depths x 5 positions per pair
- `interpret/run6/B/lab5` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `edit_enabled`=False, `matched_top_patch_mean_recovery`=0.6721, `n_candidate_facts`=25, `n_facts_kept_base`=25, `n_pairs_rejected_alignment`=0
  - model: `allenai/Olmo-3-1025-7B` (32 blocks, d_model 4096)
  - facts: 25 base-template pairs past the gate (0 dropped, 0 rejected by the alignment validator)
  - grid: 33 stream depths x 5 positions per pair
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab05_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab05_gemma4e4b_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `edit_enabled`=False, `matched_top_patch_mean_recovery`=0.7731, `n_candidate_facts`=25, `n_facts_kept_base`=25, `n_pairs_rejected_alignment`=0
  - model: `google/gemma-4-E4B-it` (42 blocks, d_model 2560)
  - facts: 25 base-template pairs past the gate (0 dropped, 0 rejected by the alignment validator)
  - grid: 43 stream depths x 5 positions per pair
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab05_tiera_labs1_25_full_matrix_20260615_000508/lab05_tiera_labs1_25_full_matrix_20260615_000508` (gpt2, tier a)
  - Metrics: `edit_enabled`=False, `matched_top_patch_mean_recovery`=0.9621, `n_candidate_facts`=6, `n_facts_kept_base`=6, `n_pairs_rejected_alignment`=0
  - model: `gpt2` (12 blocks, d_model 768)
  - facts: 6 base-template pairs past the gate (0 dropped, 0 rejected by the alignment validator)
  - grid: 13 stream depths x 5 positions per pair

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/run6/C/lab5` | `allenai/Olmo-3-1125-32B` | `c` | `edit_enabled`=False; `matched_top_patch_mean_recovery`=0.5443; `n_candidate_facts`=25 |
| `interpret/run6/B/lab5` | `allenai/Olmo-3-1025-7B` | `b` | `edit_enabled`=False; `matched_top_patch_mean_recovery`=0.6721; `n_candidate_facts`=25 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab05_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab05_gemma4e4b_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `edit_enabled`=False; `matched_top_patch_mean_recovery`=0.7731; `n_candidate_facts`=25 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab05_tiera_labs1_25_full_matrix_20260615_000508/lab05_tiera_labs1_25_full_matrix_20260615_000508` | `gpt2` | `a` | `edit_enabled`=False; `matched_top_patch_mean_recovery`=0.9621; `n_candidate_facts`=6 |
| `interpret/validate_part2_plots/lab5` | `unknown` | `` | see copied summaries |

## Curated Artifacts

- `olmo3_32b_run6c_causal_patching_dashboard.png`
- `olmo3_32b_run6c_paraphrase_transfer_matrix.png`
- `olmo3_32b_run6c_results.csv`
- `olmo3_32b_run6c_metrics.json`
- `olmo3_1025_7b_run6b_causal_patching_dashboard.png`
- `olmo3_1025_7b_run6b_paraphrase_transfer_matrix.png`
- `olmo3_1025_7b_run6b_results.csv`
- `olmo3_1025_7b_run6b_metrics.json`
- `gemma4e4b_lab05_gemma4e4b_labs1_25_local_reruns_20260615_1_causal_patching_dashboard.png`
- `gemma4e4b_lab05_gemma4e4b_labs1_25_local_reruns_20260615_1_paraphrase_transfer_matrix.png`
- `gemma4e4b_lab05_gemma4e4b_labs1_25_local_reruns_20260615_1_results.csv`
- `gemma4e4b_lab05_gemma4e4b_labs1_25_local_reruns_20260615_1_metrics.json`
- `gpt2_lab05_tiera_labs1_25_full_matrix_20260615_000508_causal_patching_dashboard.png`
- `gpt2_lab05_tiera_labs1_25_full_matrix_20260615_000508_patching_heatmap_spain.png`
- `gpt2_lab05_tiera_labs1_25_full_matrix_20260615_000508_results.csv`
- `gpt2_lab05_tiera_labs1_25_full_matrix_20260615_000508_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
