# Lab 33 Validation

## Lab 33: Multimodal Mechanistic Interpretability

Multimodal mechanistic interpretability: synthetic connector audit with visual, OCR, background, caption, text, alignment, and patch controls.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/lab33_tierc_full_verify_20260615_0012/lab33_tierc_full_verify_20260615_0012` (gpt2, tier c)
  - Metrics: `n_items`=28, `mean_connector_auc`=1, `mean_control_floor`=0.0237, `mean_specificity_gap`=0.5637, `mean_text_query_auc`=0.5, `mean_visual_auc`=1, `mean_visual_region_recovery`=0.5873, `n_counterexamples`=8
  - data rows: 28 selected from `multimodal_concept_pairs.jsonl`
  - families: `{'color': 4, 'shape': 4, 'count': 4, 'spatial': 4, 'chart': 4, 'ocr_control': 4, 'background_control': 4}`
  - science_ready_for_real_vlm: `False`
- `interpret/verify_part3/lab33_tierb_full_verify_20260615_0009/lab33_tierb_full_verify_20260615_0009` (gpt2, tier b)
  - Metrics: `n_items`=28, `mean_connector_auc`=1, `mean_control_floor`=0.0237, `mean_specificity_gap`=0.5637, `mean_text_query_auc`=0.5, `mean_visual_auc`=1, `mean_visual_region_recovery`=0.5873, `n_counterexamples`=8
  - data rows: 28 selected from `multimodal_concept_pairs.jsonl`
  - families: `{'color': 4, 'shape': 4, 'count': 4, 'spatial': 4, 'chart': 4, 'ocr_control': 4, 'background_control': 4}`
  - science_ready_for_real_vlm: `False`
- `interpret/verify_part3/lab33_tiera_full_verify_20260615_0005/lab33_tiera_full_verify_20260615_0005` (gpt2, tier a)
  - Metrics: `n_items`=16, `mean_connector_auc`=1, `mean_control_floor`=0.0248, `mean_specificity_gap`=0.5646, `mean_text_query_auc`=0.5, `mean_visual_auc`=1, `mean_visual_region_recovery`=0.5894, `n_counterexamples`=4
  - data rows: 16 selected from `multimodal_concept_pairs.jsonl`
  - families: `{'color': 3, 'shape': 3, 'count': 2, 'spatial': 2, 'chart': 2, 'ocr_control': 2, 'background_control': 2}`
  - science_ready_for_real_vlm: `False`

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/lab33_tierc_full_verify_20260615_0012/lab33_tierc_full_verify_20260615_0012` | `gpt2` | `c` | `n_items`=28; `mean_connector_auc`=1; `mean_control_floor`=0.0237 |
| `interpret/verify_part3/lab33_tierb_full_verify_20260615_0009/lab33_tierb_full_verify_20260615_0009` | `gpt2` | `b` | `n_items`=28; `mean_connector_auc`=1; `mean_control_floor`=0.0237 |
| `interpret/verify_part3/lab33_tiera_full_verify_20260615_0005/lab33_tiera_full_verify_20260615_0005` | `gpt2` | `a` | `n_items`=16; `mean_connector_auc`=1; `mean_control_floor`=0.0248 |

## Curated Artifacts

- `gpt2_lab33_tierc_full_verify_20260615_0012_multimodal_evidence_dashboard.png`
- `gpt2_lab33_tierc_full_verify_20260615_0012_concept_specificity_matrix.png`
- `gpt2_lab33_tierc_full_verify_20260615_0012_results.csv`
- `gpt2_lab33_tierc_full_verify_20260615_0012_metrics.json`
- `gpt2_lab33_tierb_full_verify_20260615_0009_multimodal_evidence_dashboard.png`
- `gpt2_lab33_tierb_full_verify_20260615_0009_concept_specificity_matrix.png`
- `gpt2_lab33_tierb_full_verify_20260615_0009_results.csv`
- `gpt2_lab33_tierb_full_verify_20260615_0009_metrics.json`
- `gpt2_lab33_tiera_full_verify_20260615_0005_multimodal_evidence_dashboard.png`
- `gpt2_lab33_tiera_full_verify_20260615_0005_concept_specificity_matrix.png`
- `gpt2_lab33_tiera_full_verify_20260615_0005_results.csv`
- `gpt2_lab33_tiera_full_verify_20260615_0005_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
