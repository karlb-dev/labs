# Lab 21 Validation

## Lab 21 - Where Training Lives: LoRA Localization and Safety Depth

Where training lives: LoRA localization, wrapper tests, and safety-depth audits.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab21_tierc_both_labs1_25_full_matrix_20260615_000508/lab21_tierc_both_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `comparison_model_id`=allenai/Olmo-3-1025-7B, `erosion_has_external_result`=False, `identity_comparison_smoke`=False, `n_adapter_sources`=10, `n_boundary_safe_rows`=198, `n_chat_control_rows`=198, `n_depth_disagreement_rows`=4, `n_forced_prefix_rows`=4950
  - Modes: `lora, safety_depth`
  - Main model: `allenai/Olmo-3-7B-Instruct`
  - Comparison model: `allenai/Olmo-3-1025-7B`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab21_olmo32bthink_both_labs1_25_full_matrix_20260615_000508/lab21_olmo32bthink_both_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `comparison_model_id`=allenai/Olmo-3-1025-7B, `erosion_has_external_result`=False, `identity_comparison_smoke`=False, `n_adapter_sources`=0, `n_boundary_safe_rows`=390, `n_chat_control_rows`=390, `n_depth_disagreement_rows`=4, `n_forced_prefix_rows`=9750
  - Modes: `lora, safety_depth`
  - Main model: `allenai/Olmo-3-32B-Think`
  - Comparison model: `allenai/Olmo-3-1025-7B`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab21_tierb_both_labs1_25_full_matrix_20260615_000508/lab21_tierb_both_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier b)
  - Metrics: `comparison_model_id`=allenai/Olmo-3-1025-7B, `erosion_has_external_result`=False, `identity_comparison_smoke`=False, `n_adapter_sources`=10, `n_boundary_safe_rows`=198, `n_chat_control_rows`=198, `n_depth_disagreement_rows`=4, `n_forced_prefix_rows`=4950
  - Modes: `lora, safety_depth`
  - Main model: `allenai/Olmo-3-7B-Instruct`
  - Comparison model: `allenai/Olmo-3-1025-7B`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab21_tiera_lora_labs1_25_full_matrix_20260615_000508/lab21_tiera_lora_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `erosion_has_external_result`=False, `identity_comparison_smoke`=False, `n_adapter_sources`=8, `n_boundary_safe_rows`=0, `n_chat_control_rows`=0, `n_depth_disagreement_rows`=4, `n_forced_prefix_rows`=0, `n_lora_layer_rows`=0
  - Modes: `lora`
  - Main model: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - Comparison model: ``

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab21_tierc_both_labs1_25_full_matrix_20260615_000508/lab21_tierc_both_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `comparison_model_id`=allenai/Olmo-3-1025-7B; `erosion_has_external_result`=False; `identity_comparison_smoke`=False |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab21_olmo32bthink_both_labs1_25_full_matrix_20260615_000508/lab21_olmo32bthink_both_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-32B-Think` | `c` | `comparison_model_id`=allenai/Olmo-3-1025-7B; `erosion_has_external_result`=False; `identity_comparison_smoke`=False |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab21_tierb_both_labs1_25_full_matrix_20260615_000508/lab21_tierb_both_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `b` | `comparison_model_id`=allenai/Olmo-3-1025-7B; `erosion_has_external_result`=False; `identity_comparison_smoke`=False |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab21_tiera_lora_labs1_25_full_matrix_20260615_000508/lab21_tiera_lora_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `erosion_has_external_result`=False; `identity_comparison_smoke`=False; `n_adapter_sources`=8 |

## Curated Artifacts

- `olmo3_7b_lab21_tierc_both_labs1_25_full_matrix_20260615_0_lora_layer_atlas.png`
- `olmo3_7b_lab21_tierc_both_labs1_25_full_matrix_20260615_0_training_depth_evidence_dashboard.png`
- `olmo3_7b_lab21_tierc_both_labs1_25_full_matrix_20260615_0_results.csv`
- `olmo3_7b_lab21_tierc_both_labs1_25_full_matrix_20260615_0_metrics.json`
- `olmo3_32b_lab21_olmo32bthink_both_labs1_25_full_matrix_202_lora_layer_atlas.png`
- `olmo3_32b_lab21_olmo32bthink_both_labs1_25_full_matrix_202_training_depth_evidence_dashboard.png`
- `olmo3_32b_lab21_olmo32bthink_both_labs1_25_full_matrix_202_results.csv`
- `olmo3_32b_lab21_olmo32bthink_both_labs1_25_full_matrix_202_metrics.json`
- `olmo3_7b_lab21_tierb_both_labs1_25_full_matrix_20260615_0_lora_layer_atlas.png`
- `olmo3_7b_lab21_tierb_both_labs1_25_full_matrix_20260615_0_training_depth_evidence_dashboard.png`
- `olmo3_7b_lab21_tierb_both_labs1_25_full_matrix_20260615_0_results.csv`
- `olmo3_7b_lab21_tierb_both_labs1_25_full_matrix_20260615_0_metrics.json`
- `smollm_lab21_tiera_lora_labs1_25_full_matrix_20260615_0_lora_layer_atlas.png`
- `smollm_lab21_tiera_lora_labs1_25_full_matrix_20260615_0_training_depth_evidence_dashboard.png`
- `smollm_lab21_tiera_lora_labs1_25_full_matrix_20260615_0_results.csv`
- `smollm_lab21_tiera_lora_labs1_25_full_matrix_20260615_0_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
