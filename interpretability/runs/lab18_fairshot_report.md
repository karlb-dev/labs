# Lab 18 Humor Fair-Shot Report

Date: 2026-06-20

## Verdict

Overall result: **partial positive**.

The v2 Lab 18 corpus and train/dev/test protocol produce a stronger result than the old 20-row lab. On `allenai/Olmo-3-7B-Instruct`, seed 0 showed a controlled joke-structure decode handle: test AUC `1.0`, best-null selectivity `0.2172`, and low cosine with surprise/silly/positive controls. Seed 1 was weaker: test AUC `0.9272`, best-null selectivity `0.0722`. Both seeds failed causal steering specificity, and family-heldout transfer stayed mixed.

Allowed claim: Lab 18 can show a scoped joke-structure/register decodability handle on the course 7B instruct model. It should not claim robust causal humor steering or a general humor mechanism.

## What Changed

- Expanded `data/humor_incongruity_pairs.csv` from 20 to 80 rows.
- Added `data/humor_incongruity_pairs_card.md`.
- Added Lab 18 `--corpus-path` support.
- Switched Lab 18 from train/eval to train/dev/test: train fits, dev selects depth, test reports.
- Updated depth-selection, metrics, plots, docs, and validation wording.
- Fixed the family generalization summary to report probe rows and held-out joke counts clearly.

## Commands Run

```bash
python interpretability/data/make_humor_incongruity.py
python -m py_compile interpretability/interp_bench.py interpretability/labs/lab18_humor_incongruity.py interpretability/data/make_humor_incongruity.py

cd interpretability
python interp_bench.py --lab lab18 --tier a --prompt-set small --corpus-path data/humor_incongruity_pairs.csv --no-plots --run-name lab18_smoke_v2_20260620
python interp_bench.py --lab lab18 --tier a --prompt-set full --corpus-path data/humor_incongruity_pairs.csv --max-examples 0 --run-name lab18_fairshot_smolm_v2_full_s0_20260620
python interp_bench.py --lab lab18 --tier b --prompt-set full --corpus-path data/humor_incongruity_pairs.csv --max-examples 0 --run-name lab18_fairshot_olmo3_7b_v2_full_s0_20260620
python interp_bench.py --lab lab18 --tier b --prompt-set full --corpus-path data/humor_incongruity_pairs.csv --max-examples 0 --seed 1 --run-name lab18_fairshot_olmo3_7b_v2_full_s1_20260620
```

## Run Summary

| Run | Model | Seed | Rows | Best Depth | Verdict | Test AUC | Best-Null Gap | Family Gap | Steering Gap |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `lab18_fairshot_smolm_v2_full_s0_20260620` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 0 | 80 | 5 | `weak_or_confounded_joke_register_handle` | 1.0000 | 0.0824 | 0.0230 | -0.8000 |
| `lab18_fairshot_olmo3_7b_v2_full_s0_20260620` | `allenai/Olmo-3-7B-Instruct` | 0 | 80 | 2 | `decodable_joke_structure_handle_not_causally_separated_by_this_run` | 1.0000 | 0.2172 | 0.0643 | -0.3000 |
| `lab18_fairshot_olmo3_7b_v2_full_s1_20260620` | `allenai/Olmo-3-7B-Instruct` | 1 | 80 | 6 | `weak_or_confounded_joke_register_handle` | 0.9272 | 0.0722 | 0.0566 | -0.6000 |

## Corpus

- File: `data/humor_incongruity_pairs.csv`
- Rows: 80
- SHA256: `d7165cf41e67f248f70cd9691dbda24860c3431701a8003e818c3fff88f2fdb0`
- Families: 8 families with 10 rows each
- Conditions: `joke`, `literal`, `surprise`, `silly`, `positive`

## Artifacts

Committed validation summary:

- `validation/lab18/fairshot_20260620_summary.csv`
- `validation/lab18/fairshot_20260620_*_metrics.json`
- `validation/lab18/fairshot_20260620_*_humor_evidence_dashboard.png`
- `validation/lab18/fairshot_20260620_*_humor_evidence_matrix.csv`
- `validation/lab18/fairshot_20260620_*_family_generalization_summary.csv`

Drive backup:

```text
/content/drive/MyDrive/interpret/lab18_humor_fairshot_20260620/
```

## Remaining Work

- Add a targeted causal mode with more prompts for the best decoded family instead of broad all-family steering.
- Add human labels or a held-out judge/probe for generated joke structure; current steering still uses marker scaffolds.
- Expand family-heldout rows further if this lab needs a stronger family-transfer claim.
