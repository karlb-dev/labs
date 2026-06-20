# Lab 17 Persona Fair-Shot Report

Date: 2026-06-20

## Verdict

Overall result: **partial positive**.

Lab 17 now gives a strong controlled `DECODE` result on the expanded corpus, especially on `allenai/Olmo-3-7B-Instruct`: both full-corpus 7B seeds reached mean test AUC `1.0` while random controls stayed around `0.64-0.66`. The `CAUSAL` rail is negative under stronger matched controls. A pilot 7B run produced one apparent `honest_disagreement` controlled-style handle, but the five-random-control rerun removed it.

Practical read: Lab 17 is good as a skeptical persona/register decodability lab. It should not claim robust causal persona steering yet.

## What Changed

- Expanded `data/persona_register_pairs.csv` from 24 rows to 256 rows: 8 traits by 32 matched tasks.
- Added `data/persona_register_pairs_card.md` and updated `data/MANIFEST.json` with the new SHA256.
- Reworked Lab 17 split hygiene to train/dev/test by `trait:topic`.
- Changed depth selection to dev control-adjusted score; test is report-only.
- Added bootstrap AUC intervals and permutation-null summaries for real held-out probe rows.
- Added `--corpus-path`, `--persona-steering-prompts`, and `--persona-steering-controls` support.
- Strengthened steering controls by allowing multiple random steering directions.
- Fixed the evidence dashboard decode panel and improved dense plot labels.
- Updated the Lab 17 handout and committed a new validation summary pack.

## Exact Commands Run

```bash
python interpretability/data/make_persona_register_pairs.py
python -m py_compile interpretability/interp_bench.py interpretability/labs/lab17_persona_voice_register.py interpretability/data/make_persona_register_pairs.py

cd interpretability
python interp_bench.py --lab lab17 --tier a --prompt-set small --corpus-path data/persona_register_pairs.csv --persona-steering-prompts 1 --persona-steering-controls 1 --no-plots --run-name lab17_smoke_v2_20260620
python interp_bench.py --lab lab17 --tier a --prompt-set small --corpus-path data/persona_register_pairs.csv --persona-steering-prompts 1 --persona-steering-controls 1 --run-name lab17_smoke_v2_plots2_20260620
python interp_bench.py --lab lab17 --tier a --prompt-set full --corpus-path data/persona_register_pairs.csv --max-examples 0 --persona-steering-prompts 4 --persona-steering-controls 3 --run-name lab17_fairshot_smolm_v2_full_s0b_20260620
python interp_bench.py --lab lab17 --tier b --prompt-set full --corpus-path data/persona_register_pairs.csv --max-examples 0 --persona-steering-prompts 2 --persona-steering-controls 2 --run-name lab17_fairshot_olmo3_7b_v2_full_s0_20260620
python interp_bench.py --lab lab17 --tier b --prompt-set full --corpus-path data/persona_register_pairs.csv --max-examples 0 --persona-steering-prompts 4 --persona-steering-controls 5 --run-name lab17_fairshot_olmo3_7b_v2_full_s0_controls5_20260620
python interp_bench.py --lab lab17 --tier b --prompt-set full --corpus-path data/persona_register_pairs.csv --max-examples 0 --persona-steering-prompts 4 --persona-steering-controls 5 --seed 1 --run-name lab17_fairshot_olmo3_7b_v2_full_s1_controls5_20260620
```

Backups:

```bash
rsync -a runs/<run_name>/ /content/drive/MyDrive/interpret/lab17_persona_fairshot_20260620/<run_name>/
```

## Run Summary

| Run | Model | Seed | Rows | Best Depth | Verdict | Test AUC | Random AUC | Steering Gap | Content Delta |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `lab17_fairshot_smolm_v2_full_s0b_20260620` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 0 | 256 | 29 | `not_validated_by_controls` | 0.9847 | 0.5888 | 1.4688 | -0.8125 |
| `lab17_fairshot_olmo3_7b_v2_full_s0_20260620` | `allenai/Olmo-3-7B-Instruct` | 0 | 256 | 5 | `decodable_but_not_steerable_by_this_run` | 1.0000 | 0.6439 | 0.0000 | 0.0000 |
| `lab17_fairshot_olmo3_7b_v2_full_s0_controls5_20260620` | `allenai/Olmo-3-7B-Instruct` | 0 | 256 | 5 | `decodable_but_not_steerable_by_this_run` | 1.0000 | 0.6439 | -0.1062 | 0.0000 |
| `lab17_fairshot_olmo3_7b_v2_full_s1_controls5_20260620` | `allenai/Olmo-3-7B-Instruct` | 1 | 256 | 10 | `decodable_but_not_steerable_by_this_run` | 1.0000 | 0.6577 | -0.1875 | 0.0312 |

## Corpus

- File: `data/persona_register_pairs.csv`
- Rows: 256
- SHA256: `fdafd17a9be0ee4d7a1e46fb2dbc6f2243354a40e5e9f701560cb9f04db795bc`
- Traits: `character_museum_guide`, `technical_register`, `warm_supportive_voice`, `honest_disagreement`, `socratic_teacher`, `concise_executive_register`, `cautious_uncertainty_voice`, `stepwise_coach`
- Families: persona, register, voice, agreement
- Split protocol: deterministic train/dev/test by `trait:topic`

## Validation Read

Probe validation is robust after the update. The 7B model had test AUC `1.0` for every trait in both full-corpus seeds. Controls remain nontrivial: random controls are high enough that the control-adjusted gap, not raw AUC, is the relevant number.

Steering validation is not robust. The small model can move style markers, but it badly damages content. The 7B model preserves content, but trait steering does not beat random/shuffled controls under the stronger sweep. The apparent pilot `honest_disagreement` causal handle was not stable when steering prompts and random controls were increased.

## Artifacts

Committed validation summary:

- `validation/lab17/fairshot_20260620_summary.csv`
- `validation/lab17/fairshot_20260620_*_metrics.json`
- `validation/lab17/fairshot_20260620_*_persona_evidence_dashboard.png`
- `validation/lab17/fairshot_20260620_*_persona_trait_evidence_matrix.csv`

Drive backup:

```text
/content/drive/MyDrive/interpret/lab17_persona_fairshot_20260620/
```

## Remaining Work

- Add a causal mode that targets one validated trait at a time with larger prompt/control grids, instead of rerunning all traits.
- Replace marker-count steering scores with a learned held-out style/content classifier.
- Test whether lower steering doses or depth-specific injection windows preserve content better on SmolLM2.
- Optionally rerun on Gemma and 32B Think with the v2 corpus if compute and licenses allow.
