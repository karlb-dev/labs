# Lab 19 Validation

## Lab 19: Model Diffing With Crosscoders

Model diffing with crosscoders: shared/base-only/instruct-only feature atlas and controls.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab19_tierc_labs1_25_full_matrix_20260615_000508/lab19_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-1025-7B, tier c)
  - Metrics: `audit_status`=template_control_dominates, `causal_marker_verdict`=skipped, `d_model_a`=4096, `d_model_b`=4096, `depth_a`=21, `depth_b`=21, `eval_fvu_model_a`=0.2138, `eval_fvu_model_b`=0.2565
  - model A: `allenai/Olmo-3-1025-7B` (model_a)
  - model B: `allenai/Olmo-3-7B-Instruct` (instruct)
  - identity-pair smoke: False
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab19_tierb_labs1_25_full_matrix_20260615_000508/lab19_tierb_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `audit_status`=template_control_dominates, `causal_marker_verdict`=skipped, `d_model_a`=4096, `d_model_b`=4096, `depth_a`=21, `depth_b`=21, `eval_fvu_model_a`=0.2138, `eval_fvu_model_b`=0.2565
  - model A: `allenai/Olmo-3-1025-7B` (model_a)
  - model B: `allenai/Olmo-3-7B-Instruct` (instruct)
  - identity-pair smoke: False
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab19_tiera_labs1_25_full_matrix_20260615_000508/lab19_tiera_labs1_25_full_matrix_20260615_000508` (EleutherAI/pythia-160m, tier a)
  - Metrics: `audit_status`=identity_pair_failed_or_dictionary_unstable, `causal_marker_verdict`=skipped, `d_model_a`=768, `d_model_b`=768, `depth_a`=8, `depth_b`=8, `eval_fvu_model_a`=0.8439, `eval_fvu_model_b`=0.8474
  - model A: `EleutherAI/pythia-160m` (model_a)
  - model B: `EleutherAI/pythia-160m` (model_b)
  - identity-pair smoke: True

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab19_tierc_labs1_25_full_matrix_20260615_000508/lab19_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-1025-7B` | `c` | `audit_status`=template_control_dominates; `causal_marker_verdict`=skipped; `d_model_a`=4096 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab19_tierb_labs1_25_full_matrix_20260615_000508/lab19_tierb_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-1025-7B` | `b` | `audit_status`=template_control_dominates; `causal_marker_verdict`=skipped; `d_model_a`=4096 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab19_tiera_labs1_25_full_matrix_20260615_000508/lab19_tiera_labs1_25_full_matrix_20260615_000508` | `EleutherAI/pythia-160m` | `a` | `audit_status`=identity_pair_failed_or_dictionary_unstable; `causal_marker_verdict`=skipped; `d_model_a`=768 |

## Curated Artifacts

- `olmo3_1025_7b_lab19_tierc_labs1_25_full_matrix_20260615_0005_model_diffing_evidence_dashboard.png`
- `olmo3_1025_7b_lab19_tierc_labs1_25_full_matrix_20260615_0005_feature_context_atlas.png`
- `olmo3_1025_7b_lab19_tierc_labs1_25_full_matrix_20260615_0005_tables_causal_feature_validation_summary.csv`
- `olmo3_1025_7b_lab19_tierc_labs1_25_full_matrix_20260615_0005_tables_causal_feature_validation.csv`
- `olmo3_1025_7b_lab19_tierb_labs1_25_full_matrix_20260615_0005_model_diffing_evidence_dashboard.png`
- `olmo3_1025_7b_lab19_tierb_labs1_25_full_matrix_20260615_0005_feature_context_atlas.png`
- `olmo3_1025_7b_lab19_tierb_labs1_25_full_matrix_20260615_0005_tables_causal_feature_validation_summary.csv`
- `olmo3_1025_7b_lab19_tierb_labs1_25_full_matrix_20260615_0005_tables_causal_feature_validation.csv`
- `pythia-160m_lab19_tiera_labs1_25_full_matrix_20260615_000508_model_diffing_evidence_dashboard.png`
- `pythia-160m_lab19_tiera_labs1_25_full_matrix_20260615_000508_identity_smoke_scorecard.png`
- `pythia-160m_lab19_tiera_labs1_25_full_matrix_20260615_000508_tables_causal_feature_validation_summary.csv`
- `pythia-160m_lab19_tiera_labs1_25_full_matrix_20260615_000508_tables_causal_feature_validation.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.

## 2026-06-20 Fair-Shot Update

The Lab 19 instrument was updated after audit:

- same-dimensional model pairs now use shared side initialization, so identity-pair smoke runs no longer invent side-specific features from unrelated random initialization;
- random-direction controls now distinguish `matched_shared_direction` from the deliberately harsher `independent_side_directions` diagnostic;
- prompt groups now split into train/dev/test, with test used for reported reconstruction and stability;
- a deterministic v2 prompt inventory was added at `data/model_diffing_prompt_inventory_v2.csv` with 96 raw prompt groups and runtime paired chat variants.

Current fair-shot summary: `fairshot_20260620_summary.csv`.

Key current runs:

| Run | Read |
|---|---|
| `lab19_fairshot_tiera_identity_v2_plots_20260620` | Identity smoke passes the repaired specificity control: all 48 features are shared; matched-random false-specific rate is 0.0. |
| `lab19_fairshot_olmo3_medium_edit_v2_20260620` | OLMo base/instruct medium run reconstructs well enough, but no candidate instruct-only handles survive first-pass controls; template control dominates; optional edit is not validated by marker controls. |
| `lab19_fairshot_olmo3_v2_full_edit_20260620` | Balanced v2 inventory gives the same scientific read: no defended instruct-only feature handle, template residue dominates, activation norm shifts are large, and causal marker specificity is 0.0. |
| `lab19_fairshot_olmo3_v2_full_plots_20260620` | Plotted v2 evidence pack for dashboard and audit matrices. |

Current conclusion: Lab 19 is now a cleaner negative/teaching result. The paired crosscoder and identity smoke are healthier, but the OLMo base-vs-instruct runs still do not license a robust model-B/instruct-only feature claim under matched controls. The strongest positive result is methodological: the repaired controls stop a false identity-pair failure and make the negative easier to trust.
