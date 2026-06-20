# Lab 08 Validation

## Lab 8: Superposition, Sparse Autoencoders, and Transcoders

Superposition, SAEs, and transcoders: find, label, and validate features.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/run6/C/lab8` (allenai/Olmo-3-1025-7B, tier c)
  - Metrics: `clamp_causal`=True, `n_survived`=0, `n_killed`=21, `reconstruction_fvu`=0.3761, `transcoder_fvu`=0.3943, `atlas_size`=25, `per_token_l0`=113.5, `ranking_overlap_topN`=0
  - model: `allenai/Olmo-3-1025-7B` (base model; SAE/transcoder are pretrained, pinned)
  - SAE layer 16, d_sae 65536; transcoder on gpt2
  - evidence level: OBS/DECODE at the feature level, CAUSAL for the one clamped feature
- `interpret/run6/B/lab8` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `clamp_causal`=True, `n_survived`=0, `n_killed`=21, `reconstruction_fvu`=0.3761, `transcoder_fvu`=0.3943, `atlas_size`=25, `per_token_l0`=113.5, `ranking_overlap_topN`=0
  - model: `allenai/Olmo-3-1025-7B` (base model; SAE/transcoder are pretrained, pinned)
  - SAE layer 16, d_sae 65536; transcoder on gpt2
  - evidence level: OBS/DECODE at the feature level, CAUSAL for the one clamped feature
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab08_tiera_labs1_25_full_matrix_20260615_000508/lab08_tiera_labs1_25_full_matrix_20260615_000508` (gpt2, tier a)
  - Metrics: `clamp_causal`=False, `n_survived`=0, `n_killed`=23, `reconstruction_fvu`=0.0019, `transcoder_fvu`=0.3943, `atlas_size`=25, `per_token_l0`=74.57, `ranking_overlap_topN`=2
  - model: `gpt2` (base model; SAE/transcoder are pretrained, pinned)
  - SAE layer 8, d_sae 24576; transcoder on gpt2
  - evidence level: OBS/DECODE at the feature level, CAUSAL for the one clamped feature
- `interpret/validate_part2_plots/lab8` (unknown model)
  - model: `gpt2` (base model; SAE/transcoder are pretrained, pinned)
  - SAE layer 8, d_sae 24576; transcoder on gpt2
  - evidence level: OBS/DECODE at the feature level, CAUSAL for the one clamped feature

## What This Lab Teaches

- The SAE evidence is strongest as a validation-discipline lesson: reconstruction can work while most tempting feature labels die.
- Read narrowed/survived features together with clamp controls; decodability alone is not a causal claim.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/run6/C/lab8` | `allenai/Olmo-3-1025-7B` | `c` | `clamp_causal`=True; `n_survived`=0; `n_killed`=21 |
| `interpret/run6/B/lab8` | `allenai/Olmo-3-1025-7B` | `b` | `clamp_causal`=True; `n_survived`=0; `n_killed`=21 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab08_tiera_labs1_25_full_matrix_20260615_000508/lab08_tiera_labs1_25_full_matrix_20260615_000508` | `gpt2` | `a` | `clamp_causal`=False; `n_survived`=0; `n_killed`=23 |
| `interpret/validate_part2_plots/lab8` | `unknown` | `` | see copied summaries |

## Curated Artifacts

- `olmo3_1025_7b_run6c_feature_validation_matrix.png`
- `olmo3_1025_7b_run6c_sae_activity_dashboard.png`
- `olmo3_1025_7b_run6c_tables_domain_validation_summary.csv`
- `olmo3_1025_7b_run6c_tables_feature_evidence_matrix.csv`
- `olmo3_1025_7b_run6b_feature_validation_matrix.png`
- `olmo3_1025_7b_run6b_sae_activity_dashboard.png`
- `olmo3_1025_7b_run6b_tables_domain_validation_summary.csv`
- `olmo3_1025_7b_run6b_tables_feature_evidence_matrix.csv`
- `gpt2_lab08_tiera_labs1_25_full_matrix_20260615_000508_sae_activity_dashboard.png`
- `gpt2_lab08_tiera_labs1_25_full_matrix_20260615_000508_feature_evidence_dashboard.png`
- `gpt2_lab08_tiera_labs1_25_full_matrix_20260615_000508_tables_domain_validation_summary.csv`
- `gpt2_lab08_tiera_labs1_25_full_matrix_20260615_000508_results.csv`
- `unknown_lab8_feature_validation_matrix.png`
- `unknown_lab8_sae_activity_dashboard.png`
- `unknown_lab8_run_summary.md`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
