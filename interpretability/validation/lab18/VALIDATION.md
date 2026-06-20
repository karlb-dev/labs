# Lab 18 Validation

## Lab 18 - Humor as Incongruity

Humor as incongruity: surprisal, joke-vs-control directions, setup routing, and steering audits.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

## 2026-06-20 Fair-Shot Update

The fair-shot update expands the frozen corpus from 20 rows to 80 rows, adds a dataset card, supports `--corpus-path`, and changes depth selection to train/dev/test: train fits directions, dev selects the stream depth, and test provides headline probe metrics.

Headline read: Lab 18 is now a **partial positive** for controlled joke-structure decodability on `allenai/Olmo-3-7B-Instruct`, but not a causal steering result. Seed 0 reached test AUC `1.0` with selectivity over the best null of `0.2172`; seed 1 dropped to test AUC `0.9272` and selectivity `0.0722`. Both seeds failed steering specificity. Family-heldout transfer remains mixed.

New committed summary:

- `fairshot_20260620_summary.csv`
- `fairshot_20260620_lab18_fairshot_smolm_v2_full_s0_20260620_*`
- `fairshot_20260620_lab18_fairshot_olmo3_7b_v2_full_s0_20260620_*`
- `fairshot_20260620_lab18_fairshot_olmo3_7b_v2_full_s1_20260620_*`

Full raw run directories were backed up to Drive under:

```text
/content/drive/MyDrive/interpret/lab18_humor_fairshot_20260620/
```

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab18_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab18_olmo32bthink_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `best_depth`=49, `family_heldout_mean_control_gap`=0.0615, `family_heldout_mean_real_auc`=0.995, `humor_positive_cosine`=0.0402, `humor_silly_cosine`=0.0113, `humor_surprise_cosine`=0.0434, `injection_layer`=48, `joke_minus_literal_attention_to_setup`=0.0227
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab18_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab18_gemma4e4b_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `best_depth`=40, `family_heldout_mean_control_gap`=0.11, `family_heldout_mean_real_auc`=0.9975, `humor_positive_cosine`=0.327, `humor_silly_cosine`=0.0862, `humor_surprise_cosine`=0.0375, `injection_layer`=39, `joke_minus_literal_attention_to_setup`=0.0803
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab18_tierc_labs1_25_full_matrix_20260615_000508/lab18_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `best_depth`=12, `family_heldout_mean_control_gap`=0.087, `family_heldout_mean_real_auc`=0.9925, `humor_positive_cosine`=0.0297, `humor_silly_cosine`=0.0132, `humor_surprise_cosine`=0.0366, `injection_layer`=11, `joke_minus_literal_attention_to_setup`=0.0245
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab18_tiera_labs1_25_full_matrix_20260615_000508/lab18_tiera_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `best_depth`=3, `family_heldout_mean_control_gap`=0.0625, `family_heldout_mean_real_auc`=1, `humor_positive_cosine`=0.0119, `humor_silly_cosine`=0.0156, `humor_surprise_cosine`=0.0441, `injection_layer`=2, `joke_minus_literal_attention_to_setup`=0.0161

## What This Lab Teaches

- The central lesson is decodability with controls: useful probes must survive selectivity, held-out data, and confound checks.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab18_olmo32bthink_labs1_25_local_reruns_20260615_101609/lab18_olmo32bthink_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `best_depth`=49; `family_heldout_mean_control_gap`=0.0615; `family_heldout_mean_real_auc`=0.995 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab18_gemma4e4b_labs1_25_local_reruns_20260615_101609/lab18_gemma4e4b_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `best_depth`=40; `family_heldout_mean_control_gap`=0.11; `family_heldout_mean_real_auc`=0.9975 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab18_tierc_labs1_25_full_matrix_20260615_000508/lab18_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `best_depth`=12; `family_heldout_mean_control_gap`=0.087; `family_heldout_mean_real_auc`=0.9925 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab18_tiera_labs1_25_full_matrix_20260615_000508/lab18_tiera_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `best_depth`=3; `family_heldout_mean_control_gap`=0.0625; `family_heldout_mean_real_auc`=1 |

## Curated Artifacts

- `olmo3_32b_lab18_olmo32bthink_labs1_25_local_reruns_2026061_humor_evidence_dashboard.png`
- `olmo3_32b_lab18_olmo32bthink_labs1_25_local_reruns_2026061_humor_evidence_matrix.png`
- `olmo3_32b_lab18_olmo32bthink_labs1_25_local_reruns_2026061_results.csv`
- `olmo3_32b_lab18_olmo32bthink_labs1_25_local_reruns_2026061_metrics.json`
- `gemma4e4b_lab18_gemma4e4b_labs1_25_local_reruns_20260615_1_humor_evidence_dashboard.png`
- `gemma4e4b_lab18_gemma4e4b_labs1_25_local_reruns_20260615_1_humor_evidence_matrix.png`
- `gemma4e4b_lab18_gemma4e4b_labs1_25_local_reruns_20260615_1_results.csv`
- `gemma4e4b_lab18_gemma4e4b_labs1_25_local_reruns_20260615_1_metrics.json`
- `olmo3_7b_lab18_tierc_labs1_25_full_matrix_20260615_000508_humor_evidence_dashboard.png`
- `olmo3_7b_lab18_tierc_labs1_25_full_matrix_20260615_000508_setup_dependence_atlas.png`
- `olmo3_7b_lab18_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `olmo3_7b_lab18_tierc_labs1_25_full_matrix_20260615_000508_metrics.json`
- `smollm_lab18_tiera_labs1_25_full_matrix_20260615_000508_humor_evidence_dashboard.png`
- `smollm_lab18_tiera_labs1_25_full_matrix_20260615_000508_setup_dependence_atlas.png`
- `smollm_lab18_tiera_labs1_25_full_matrix_20260615_000508_results.csv`
- `smollm_lab18_tiera_labs1_25_full_matrix_20260615_000508_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
