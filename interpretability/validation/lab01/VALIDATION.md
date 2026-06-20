# Lab 01 Validation

## Lab 1: Residual Stream and Logit Lens

Residual stream and logit lens: how a prediction emerges over depth.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/run6/C/lab1` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `n_examples`=43, `evidence_level`=OBS, `kept_prompt_set_sha256`=70560b2b93f83f66a24e82db71569379933cdc499f3557b9a4d764e1b7974bab, `n_dropped_tokenization`=3, `n_layers`=64, `selected_prompt_set_sha256`=d7cd94ac6851160f92ad4e3c3c55cf9251d566d7e01173446cb463772ebfb31c
  - model: `allenai/Olmo-3-1125-32B` (64 blocks, d_model 5120)
  - primary device: `cuda` | input device: `cuda:0` | lens device: `cuda:0`
  - dtype: `bfloat16` | quantization: `none` | top-k: 5
- `interpret/run6/B/lab1` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `n_examples`=43, `evidence_level`=OBS, `kept_prompt_set_sha256`=70560b2b93f83f66a24e82db71569379933cdc499f3557b9a4d764e1b7974bab, `n_dropped_tokenization`=3, `n_layers`=32, `selected_prompt_set_sha256`=d7cd94ac6851160f92ad4e3c3c55cf9251d566d7e01173446cb463772ebfb31c
  - model: `allenai/Olmo-3-1025-7B` (32 blocks, d_model 4096)
  - primary device: `cuda` | input device: `cuda:0` | lens device: `cuda:0`
  - dtype: `bfloat16` | quantization: `none` | top-k: 5
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab01_tierc_labs1_25_full_matrix_20260615_000508/lab01_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `n_examples`=43, `evidence_level`=OBS, `kept_prompt_set_sha256`=70560b2b93f83f66a24e82db71569379933cdc499f3557b9a4d764e1b7974bab, `n_dropped_tokenization`=3, `n_layers`=64, `selected_prompt_set_sha256`=d7cd94ac6851160f92ad4e3c3c55cf9251d566d7e01173446cb463772ebfb31c
  - model: `allenai/Olmo-3-1125-32B` (64 blocks, d_model 5120)
  - primary device: `cuda` | input device: `cuda:0` | lens device: `cuda:0`
  - dtype: `bfloat16` | quantization: `none` | top-k: 5
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab01_gemma4e4b_labs1_25_full_matrix_20260615_000508/lab01_gemma4e4b_labs1_25_full_matrix_20260615_000508` (google/gemma-4-E4B-it, tier b)
  - Metrics: `n_examples`=44, `evidence_level`=OBS, `kept_prompt_set_sha256`=3a33328446faf8f45594bcba6297d036f6ad715d7645a0f2323e81885c33574d, `n_dropped_tokenization`=2, `n_layers`=42, `selected_prompt_set_sha256`=d7cd94ac6851160f92ad4e3c3c55cf9251d566d7e01173446cb463772ebfb31c
  - model: `google/gemma-4-E4B-it` (42 blocks, d_model 2560)
  - primary device: `cuda` | input device: `cuda:0` | lens device: `cuda:0`
  - dtype: `bfloat16` | quantization: `none` | top-k: 5

## What This Lab Teaches

- The lab is best read through its run summary and dashboard artifacts: inspect the measured claim before trusting the intuition.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/run6/C/lab1` | `allenai/Olmo-3-1125-32B` | `c` | `n_examples`=43; `evidence_level`=OBS; `kept_prompt_set_sha256`=70560b2b93f83f66a24e82db71569379933cdc499f3557b9a4d764e1b7974bab |
| `interpret/run6/B/lab1` | `allenai/Olmo-3-1025-7B` | `b` | `n_examples`=43; `evidence_level`=OBS; `kept_prompt_set_sha256`=70560b2b93f83f66a24e82db71569379933cdc499f3557b9a4d764e1b7974bab |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab01_tierc_labs1_25_full_matrix_20260615_000508/lab01_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-1125-32B` | `c` | `n_examples`=43; `evidence_level`=OBS; `kept_prompt_set_sha256`=70560b2b93f83f66a24e82db71569379933cdc499f3557b9a4d764e1b7974bab |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab01_gemma4e4b_labs1_25_full_matrix_20260615_000508/lab01_gemma4e4b_labs1_25_full_matrix_20260615_000508` | `google/gemma-4-E4B-it` | `b` | `n_examples`=44; `evidence_level`=OBS; `kept_prompt_set_sha256`=3a33328446faf8f45594bcba6297d036f6ad715d7645a0f2323e81885c33574d |
| `interpret/run6/olmo32bInstruct/lab1` | `allenai/Olmo-3.1-32B-Instruct` | `b` | `n_examples`=43; `evidence_level`=OBS; `kept_prompt_set_sha256`=70560b2b93f83f66a24e82db71569379933cdc499f3557b9a4d764e1b7974bab |

## Curated Artifacts

- `olmo3_32b_run6c_readout_dashboard.png`
- `olmo3_32b_run6c_relation_event_matrix.png`
- `olmo3_32b_run6c_results.csv`
- `olmo3_32b_run6c_metrics.json`
- `olmo3_1025_7b_run6b_readout_dashboard.png`
- `olmo3_1025_7b_run6b_relation_event_matrix.png`
- `olmo3_1025_7b_run6b_results.csv`
- `olmo3_1025_7b_run6b_metrics.json`
- `olmo3_32b_lab01_tierc_labs1_25_full_matrix_20260615_000508_readout_dashboard.png`
- `olmo3_32b_lab01_tierc_labs1_25_full_matrix_20260615_000508_readout_phase_heatmap.png`
- `olmo3_32b_lab01_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `olmo3_32b_lab01_tierc_labs1_25_full_matrix_20260615_000508_metrics.json`
- `gemma4e4b_lab01_gemma4e4b_labs1_25_full_matrix_20260615_00_readout_dashboard.png`
- `gemma4e4b_lab01_gemma4e4b_labs1_25_full_matrix_20260615_00_readout_phase_heatmap.png`
- `gemma4e4b_lab01_gemma4e4b_labs1_25_full_matrix_20260615_00_results.csv`
- `gemma4e4b_lab01_gemma4e4b_labs1_25_full_matrix_20260615_00_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
