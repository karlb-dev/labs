# Lab 12 Validation

## Lab 12: Relation Geometry and Method Validation

Relation geometry and method validation: the intro toolkit re-run on 12 controlled relation families.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab12_tierc_labs1_25_full_matrix_20260615_000508/lab12_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `n_items`=244, `best_depth`=9, `data_manifest_ok`=True, `data_source`=frozen_csv, `n_depths`=65, `normalization`=row_unit_norm, `relation_set`=full
  - model: `allenai/Olmo-3-1125-32B` (64 blocks, d_model 5120)
  - items: 244 across 12 relation families (relation-set 'full', data source `frozen_csv`, data sha256 1126f54728e0626c)
  - direction depths: {'relword': 41, 'subject': 8, 'final': 9}
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab12_olmo32bthink_labs1_25_full_matrix_20260615_000508/lab12_olmo32bthink_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `n_items`=244, `best_depth`=9, `data_manifest_ok`=True, `data_source`=frozen_csv, `n_depths`=65, `normalization`=row_unit_norm, `relation_set`=full
  - model: `allenai/Olmo-3-32B-Think` (64 blocks, d_model 5120)
  - items: 244 across 12 relation families (relation-set 'full', data source `frozen_csv`, data sha256 1126f54728e0626c)
  - direction depths: {'relword': 41, 'subject': 8, 'final': 9}
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab12_tierb_labs1_25_full_matrix_20260615_000508/lab12_tierb_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `n_items`=244, `best_depth`=9, `data_manifest_ok`=True, `data_source`=frozen_csv, `n_depths`=33, `normalization`=row_unit_norm, `relation_set`=full
  - model: `allenai/Olmo-3-1025-7B` (32 blocks, d_model 4096)
  - items: 244 across 12 relation families (relation-set 'full', data source `frozen_csv`, data sha256 1126f54728e0626c)
  - direction depths: {'relword': 21, 'subject': 8, 'final': 9}
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab12_gemma4e4b_labs1_25_full_matrix_20260615_000508/lab12_gemma4e4b_labs1_25_full_matrix_20260615_000508` (google/gemma-4-E4B-it, tier b)
  - Metrics: `n_items`=244, `best_depth`=16, `data_manifest_ok`=True, `data_source`=frozen_csv, `n_depths`=43, `normalization`=row_unit_norm, `relation_set`=full
  - model: `google/gemma-4-E4B-it` (42 blocks, d_model 2560)
  - items: 244 across 12 relation families (relation-set 'full', data source `frozen_csv`, data sha256 1126f54728e0626c)
  - direction depths: {'relword': 41, 'subject': 40, 'final': 16}

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab12_tierc_labs1_25_full_matrix_20260615_000508/lab12_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-1125-32B` | `c` | `n_items`=244; `best_depth`=9; `data_manifest_ok`=True |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab12_olmo32bthink_labs1_25_full_matrix_20260615_000508/lab12_olmo32bthink_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-32B-Think` | `c` | `n_items`=244; `best_depth`=9; `data_manifest_ok`=True |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab12_tierb_labs1_25_full_matrix_20260615_000508/lab12_tierb_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-1025-7B` | `b` | `n_items`=244; `best_depth`=9; `data_manifest_ok`=True |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab12_gemma4e4b_labs1_25_full_matrix_20260615_000508/lab12_gemma4e4b_labs1_25_full_matrix_20260615_000508` | `google/gemma-4-E4B-it` | `b` | `n_items`=244; `best_depth`=16; `data_manifest_ok`=True |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab12_tiera_labs1_25_full_matrix_20260615_000508/lab12_tiera_labs1_25_full_matrix_20260615_000508` | `gpt2` | `a` | `n_items`=96; `best_depth`=7; `data_manifest_ok`=True |

## Curated Artifacts

- `olmo3_32b_lab12_tierc_labs1_25_full_matrix_20260615_000508_relation_geometry_dashboard.png`
- `olmo3_32b_lab12_tierc_labs1_25_full_matrix_20260615_000508_relation_patch_heatmap.png`
- `olmo3_32b_lab12_tierc_labs1_25_full_matrix_20260615_000508_method_validation_card.md`
- `olmo3_32b_lab12_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `olmo3_32b_lab12_olmo32bthink_labs1_25_full_matrix_20260615_relation_geometry_dashboard.png`
- `olmo3_32b_lab12_olmo32bthink_labs1_25_full_matrix_20260615_relation_patch_heatmap.png`
- `olmo3_32b_lab12_olmo32bthink_labs1_25_full_matrix_20260615_method_validation_card.md`
- `olmo3_32b_lab12_olmo32bthink_labs1_25_full_matrix_20260615_results.csv`
- `olmo3_1025_7b_lab12_tierb_labs1_25_full_matrix_20260615_0005_relation_geometry_dashboard.png`
- `olmo3_1025_7b_lab12_tierb_labs1_25_full_matrix_20260615_0005_relation_patch_heatmap.png`
- `olmo3_1025_7b_lab12_tierb_labs1_25_full_matrix_20260615_0005_method_validation_card.md`
- `olmo3_1025_7b_lab12_tierb_labs1_25_full_matrix_20260615_0005_results.csv`
- `gemma4e4b_lab12_gemma4e4b_labs1_25_full_matrix_20260615_00_relation_geometry_dashboard.png`
- `gemma4e4b_lab12_gemma4e4b_labs1_25_full_matrix_20260615_00_relation_patch_heatmap.png`
- `gemma4e4b_lab12_gemma4e4b_labs1_25_full_matrix_20260615_00_method_validation_card.md`
- `gemma4e4b_lab12_gemma4e4b_labs1_25_full_matrix_20260615_00_results.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
