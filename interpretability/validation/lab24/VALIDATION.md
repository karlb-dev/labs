# Lab 24 Validation

## Lab 24 - Knowledge Conflict to Belief-Revision Pressure

Knowledge conflict and belief revision: context override, pressure traces, and quadrant audit.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab24_olmo32bthink_both_labs1_25_local_reruns_20260615_101609/lab24_olmo32bthink_both_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `n_items`=16, `answer_and_signal_flip`=1, `answer_flips_signal_holds`=6, `claim_posture`=answer_relevant_signal_only, `fallback_data_used`=False, `false_pressure_final_false_answer_rate`=0.3542, `headline_verdict`=answer_relevant_signal_only_truth_bridge_missing_or_unreviewed, `mean_override_mismatched_patch_recovery`=0.2256
  - Model: `allenai/Olmo-3-32B-Think`
  - Mode: `both`
  - Items: 16 selected from `/content/labs/interpretability/data/belief_revision_dialogues.csv`
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab24_gemma4e4b_both_labs1_25_local_reruns_20260615_101609/lab24_gemma4e4b_both_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `n_items`=16, `answer_and_signal_flip`=0, `answer_flips_signal_holds`=0, `claim_posture`=answer_relevant_signal_only, `fallback_data_used`=False, `false_pressure_final_false_answer_rate`=0.1042, `headline_verdict`=answer_relevant_signal_only_truth_bridge_missing_or_unreviewed, `mean_override_mismatched_patch_recovery`=0.211
  - Model: `google/gemma-4-E4B-it`
  - Mode: `both`
  - Items: 16 selected from `/content/labs/interpretability/data/belief_revision_dialogues.csv`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab24_tierc_both_labs1_25_full_matrix_20260615_000508/lab24_tierc_both_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `n_items`=16, `answer_and_signal_flip`=0, `answer_flips_signal_holds`=0, `claim_posture`=answer_relevant_signal_only, `fallback_data_used`=False, `false_pressure_final_false_answer_rate`=0.1667, `headline_verdict`=answer_relevant_signal_only_truth_bridge_missing_or_unreviewed, `mean_override_mismatched_patch_recovery`=0.2336
  - Model: `allenai/Olmo-3-7B-Instruct`
  - Mode: `both`
  - Items: 16 selected from `/content/labs/interpretability/data/belief_revision_dialogues.csv`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab24_tiera_both_labs1_25_full_matrix_20260615_000508/lab24_tiera_both_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `n_items`=1, `answer_and_signal_flip`=0, `answer_flips_signal_holds`=0, `claim_posture`=answer_relevant_signal_only, `fallback_data_used`=False, `false_pressure_final_false_answer_rate`=0, `headline_verdict`=answer_relevant_signal_only_truth_bridge_missing_or_unreviewed, `mean_override_mismatched_patch_recovery`=0
  - Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - Mode: `both`
  - Items: 1 selected from `/content/labs/interpretability/data/belief_revision_dialogues.csv`

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab24_olmo32bthink_both_labs1_25_local_reruns_20260615_101609/lab24_olmo32bthink_both_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `n_items`=16; `answer_and_signal_flip`=1; `answer_flips_signal_holds`=6 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab24_gemma4e4b_both_labs1_25_local_reruns_20260615_101609/lab24_gemma4e4b_both_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `n_items`=16; `answer_and_signal_flip`=0; `answer_flips_signal_holds`=0 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab24_tierc_both_labs1_25_full_matrix_20260615_000508/lab24_tierc_both_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `n_items`=16; `answer_and_signal_flip`=0; `answer_flips_signal_holds`=0 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab24_tiera_both_labs1_25_full_matrix_20260615_000508/lab24_tiera_both_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `n_items`=1; `answer_and_signal_flip`=0; `answer_flips_signal_holds`=0 |

## Curated Artifacts

- `olmo3_32b_lab24_olmo32bthink_both_labs1_25_local_reruns_20_belief_revision_evidence_dashboard.png`
- `olmo3_32b_lab24_olmo32bthink_both_labs1_25_local_reruns_20_self_report_behavior_matrix.png`
- `olmo3_32b_lab24_olmo32bthink_both_labs1_25_local_reruns_20_results.csv`
- `olmo3_32b_lab24_olmo32bthink_both_labs1_25_local_reruns_20_metrics.json`
- `gemma4e4b_lab24_gemma4e4b_both_labs1_25_local_reruns_20260_belief_revision_evidence_dashboard.png`
- `gemma4e4b_lab24_gemma4e4b_both_labs1_25_local_reruns_20260_self_report_behavior_matrix.png`
- `gemma4e4b_lab24_gemma4e4b_both_labs1_25_local_reruns_20260_results.csv`
- `gemma4e4b_lab24_gemma4e4b_both_labs1_25_local_reruns_20260_metrics.json`
- `olmo3_7b_lab24_tierc_both_labs1_25_full_matrix_20260615_0_belief_revision_evidence_dashboard.png`
- `olmo3_7b_lab24_tierc_both_labs1_25_full_matrix_20260615_0_pressure_signal_atlas.png`
- `olmo3_7b_lab24_tierc_both_labs1_25_full_matrix_20260615_0_results.csv`
- `olmo3_7b_lab24_tierc_both_labs1_25_full_matrix_20260615_0_metrics.json`
- `smollm_lab24_tiera_both_labs1_25_full_matrix_20260615_0_belief_revision_evidence_dashboard.png`
- `smollm_lab24_tiera_both_labs1_25_full_matrix_20260615_0_pressure_signal_atlas.png`
- `smollm_lab24_tiera_both_labs1_25_full_matrix_20260615_0_results.csv`
- `smollm_lab24_tiera_both_labs1_25_full_matrix_20260615_0_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
