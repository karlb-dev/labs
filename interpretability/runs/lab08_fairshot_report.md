# Lab 8 SAE Fair-Shot Report

Date: 2026-06-20

Verdict: partial positive.

The updated Lab 8 can now find robustly labelable SAE features under split-aware validation, and one Olmo feature passes the new matched-control causal suite. The strongest positive is Olmo feature 7849 for `finance`: train/dev/test AUC 0.9303/0.9538/1.0000, test confusable AUC 1.0000, fire fraction 0.012879, and a matched-control causal probe increase from 17.4229 to 21.0273 at 0.5x peak activation. This is not a clean positive because the top-20 context purity is 0.6, some high activations are legal/filing text, keyword-hit gains are small, and only the GPT-2 jbloom and Olmo decoderesearch SAE families were fully run.

## What Changed

- Added a historical Lab 8 audit script and regenerated `interpretability/runs/lab08_existing_run_audit.md` plus `.csv`.
- Added an explicit SAE registry and CLI SAE selection: `--sae-id`, `--sae-repo`, `--sae-subdir`, `--sae-weights`, `--sae-layer`, `--sae-site`, `--sae-center-input`, `--sae-sub-b-dec`, `--sae-jumprelu`, `--skip-transcoder`, `--atlas-budget`, `--corpus-path`, and `--feature-search`.
- Added SAE loading reports and calibration sweeps over centering, decoder-bias subtraction, ReLU vs JumpReLU, and nearby layer/site checks. Runs now stop on dimensionality mismatch or broken reconstruction.
- Added `--skip-transcoder` so SAE sweeps do not spend time on the Lab 9-style transcoder section.
- Rebuilt the SAE corpus as deterministic v3 with 1,200 rows, 20 families, 60 rows per family, and a 720/240/240 train/dev/test split. The CSV schema is `row_id,domain,family,split,text,hard_negative_group,lexical_markers,notes`.
- Added supervised targeted feature search with train discovery, dev selection, single test reporting, bootstrap AUC intervals, permutation null AUC, held-out confusable validation where available, subset stability, label types, and explicit claim grades.
- Added feature cards for selected family features.
- Added a matched causal suite with 20 neutral prompts, clamp-on dose sweep, suppression prompts, 10 controls matched by decoder norm/fire fraction/peak activation, a simple corpus-trained lexical probe, keyword hits, distinct ratio, plots, summary JSON, and a causal feature card.
- Updated the Lab 8 markdown to describe the fair-shot path and artifacts.

## Exact Commands Run

Run roots are under `/content/labs/interpretability/runs`. Useful runs were backed up under `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/<run_name>/`.

```bash
cd /content/labs/interpretability
python temp/lab08_audit_existing_runs.py --input-root runs/lab08_existing_comparison --out-dir runs
python data/make_sae_corpus.py
python -m py_compile interp_bench.py labs/lab08_sae_transcoders.py data/make_sae_corpus.py

python interp_bench.py --lab lab8 --tier a --run-name lab08_smoke_sae_registry --skip-transcoder
python interp_bench.py --lab lab8 --tier a --run-name lab08_smoke_v3_corpus --skip-transcoder --corpus-path data/sae_feature_corpus_v3.csv
python interp_bench.py --lab lab8 --tier a --run-name lab08_smoke_targeted_v3 --skip-transcoder --corpus-path data/sae_feature_corpus_v3.csv --feature-search both
python interp_bench.py --lab lab8 --tier a --run-name lab08_smoke_targeted_v3_fix --skip-transcoder --corpus-path data/sae_feature_corpus_v3.csv --feature-search both
python interp_bench.py --lab lab8 --tier a --run-name lab08_smoke_targeted_grade --skip-transcoder --corpus-path data/sae_feature_corpus_v3.csv --feature-search both
python interp_bench.py --lab lab8 --tier a --run-name lab08_smoke_causal_suite_gpt2 --skip-transcoder --corpus-path data/sae_feature_corpus_v3.csv --feature-search both --causal-suite --causal-controls 10 --causal-prompts 20

python interp_bench.py --lab lab8 --tier a --model gpt2 --dtype float32 --prompt-set small --max-examples 4 --skip-transcoder --corpus-path data/sae_feature_corpus_v3.csv --feature-search both --atlas-budget 25 --sae-layer 4 --sae-subdir blocks.4.hook_resid_pre --run-name lab08_fairshot_gpt2_jbloom_l4_v3_both_s0
python interp_bench.py --lab lab8 --tier a --model gpt2 --dtype float32 --prompt-set small --max-examples 4 --skip-transcoder --corpus-path data/sae_feature_corpus_v3.csv --feature-search both --atlas-budget 25 --sae-layer 8 --sae-subdir blocks.8.hook_resid_pre --run-name lab08_fairshot_gpt2_jbloom_l8_v3_both_s0
python interp_bench.py --lab lab8 --tier a --model gpt2 --dtype float32 --prompt-set small --max-examples 4 --skip-transcoder --corpus-path data/sae_feature_corpus_v3.csv --feature-search both --atlas-budget 25 --sae-layer 11 --sae-subdir blocks.11.hook_resid_pre --run-name lab08_fairshot_gpt2_jbloom_l11_v3_both_s0

python interp_bench.py --lab lab8 --tier b --model allenai/Olmo-3-1025-7B --dtype bfloat16 --skip-transcoder --corpus-path data/sae_feature_corpus_v3.csv --feature-search both --atlas-budget 25 --run-name lab08_fairshot_olmo3_1025_7b_decoderesearch_l16_v3_both_s0
python interp_bench.py --lab lab8 --tier b --model allenai/Olmo-3-1025-7B --dtype bfloat16 --skip-transcoder --corpus-path data/sae_feature_corpus_v3.csv --feature-search both --atlas-budget 25 --causal-suite --causal-controls 10 --causal-prompts 20 --run-name lab08_fairshot_olmo3_1025_7b_l16_v3_both_causal_s0
```

Each useful run was backed up with:

```bash
rsync -a runs/<run_name>/ /content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/<run_name>/
```

## Historical Audit

The raw historical artifacts confirm the starting-point summary.

| run | model | FVU | L0 | silent | atlas | best handle | clamp |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `lab8_run6_A` | `gpt2` | 0.0019 | 74.57 | 0.2480 | 0 survived, 2 narrowed | F9303 `code`, AUC 0.7105 | 0 -> 0 hits |
| `lab8_run6_B` | `allenai/Olmo-3-1025-7B` | 0.3761 | 113.54 | 0.6098 | 0 survived, 4 narrowed | F1265 `law`, AUC 0.7716, confusable AUC 0.8247 | 0 -> 5 hits |
| `lab8_run6_C` | `allenai/Olmo-3-1025-7B` | 0.3761 | 113.54 | 0.6098 | 0 survived, 4 narrowed | F1265 `law`, AUC 0.7716, confusable AUC 0.8247 | 0 -> 5 hits |
| `lab08_tierb_validate` | `allenai/Olmo-3-1025-7B` | 0.3736 | 113.54 | 0.6572 | 1 survived, 3 narrowed | survived F1204 `emotion`, AUC 0.8790, fire 0.85515 | 0 -> 5 hits |

The older causal evidence is treated as weak because it used keyword hits and one random control.

## Corpus

Corpus: `data/sae_feature_corpus_v3.csv`

- Rows: 1,200
- Families: 20
- Rows per family: 60
- Split: 720 train, 240 dev, 240 test
- Semantic families preserved: chemistry, code, cooking, emotion, finance, history, law, medicine, sports, weather
- SAE-native families added: capitalization/acronyms, citations/legal references, code indentation/whitespace, dates/numbers/measurements, markdown/list formatting, named entities, Python syntax, quotes/dialogue, sentiment/emotion, URLs/emails/paths
- Dataset card: `data/sae_feature_corpus_v3_card.md`

## Model And SAE Specs

| run family | model | SAE repo | layer/site | activation | dims | chosen convention |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2 sweep | `gpt2` | `jbloom/GPT2-Small-SAEs-Reformatted` | blocks 4, 8, 11 `hook_resid_pre` | ReLU | d_model 768, d_sae 24576 | center input, subtract `b_dec` |
| Olmo sweep | `allenai/Olmo-3-1025-7B` | `decoderesearch/olmo-3-saes` | layer 16 `resid_post` | JumpReLU | d_model 4096, d_sae 65536 | no centering, subtract `b_dec`, apply threshold |

The exact resolved specs are stored in each `run_config.json`; loader health is stored in each `sae_loading_report.json` and `sae_loading_calibration.json`.

## Reconstruction Health

| run | model | layer | FVU | L0 | silent | calibration chosen FVU | calibration best FVU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lab08_fairshot_gpt2_jbloom_l4_v3_both_s0` | GPT-2 | 4 | 0.0021 | 42.54 | 0.3376 | 0.000868 | 0.000868 |
| `lab08_fairshot_gpt2_jbloom_l8_v3_both_s0` | GPT-2 | 8 | 0.0048 | 74.57 | 0.1258 | 0.001697 | 0.001697 |
| `lab08_fairshot_gpt2_jbloom_l11_v3_both_s0` | GPT-2 | 11 | 0.0171 | 58.73 | 0.1203 | 0.005601 | 0.005601 |
| `lab08_fairshot_olmo3_1025_7b_l16_v3_both_causal_s0` | Olmo | 16 | 0.3602 | 113.49 | 0.4728 | 0.437008 | 0.409444 |

For Olmo, the numerically best calibration omitted the documented JumpReLU threshold and produced much higher L0. The chosen convention keeps the documented thresholded JumpReLU path, accepting slightly worse reconstruction for plausible sparsity.

## Blind Atlas Results

| run | atlas rows | verdicts |
| --- | ---: | --- |
| GPT-2 layer 4 | 30 | 1 survived, 2 narrowed, 5 polysemantic, 22 killed |
| GPT-2 layer 8 | 30 | 2 survived, 3 narrowed, 2 polysemantic, 1 token-feature, 22 killed |
| GPT-2 layer 11 | 30 | 4 survived, 1 polysemantic, 25 killed |
| Olmo layer 16 | 30 | 3 survived, 9 narrowed, 4 polysemantic, 1 token-feature, 13 killed |

The blind atlas is now more useful, but it still misses many good low-peak targeted features and remains a discovery aid rather than the final claim protocol.

## Targeted Search Results

| run | claim grades over 20 families |
| --- | --- |
| GPT-2 layer 4 | 9 survived_strong, 1 survived_weak, 7 lexical_valid, 1 narrowed, 1 token_feature_mislabeled, 1 killed |
| GPT-2 layer 8 | 6 survived_strong, 5 survived_weak, 7 lexical_valid, 1 narrowed, 1 killed |
| GPT-2 layer 11 | 8 survived_strong, 2 survived_weak, 6 lexical_valid, 3 narrowed, 1 killed |
| Olmo layer 16 | 6 survived_strong, 5 survived_weak, 5 lexical_valid, 3 narrowed, 1 killed |

Key Olmo selected features from `lab08_fairshot_olmo3_1025_7b_l16_v3_both_causal_s0/tables/best_feature_per_family.csv`:

| family | feature | grade | train/dev/test AUC | test CI | confusable AUC | fire |
| --- | ---: | --- | --- | --- | ---: | ---: |
| finance | 7849 | survived_strong | 0.9303 / 0.9538 / 1.0000 | [1.0000, 1.0000] | 1.0000 | 0.012879 |
| chemistry | 8207 | survived_strong | 1.0000 / 0.9982 / 1.0000 | [1.0000, 1.0000] | 1.0000 | 0.016066 |
| history | 986 | survived_strong | 0.9743 / 0.9996 / 0.9971 | [0.9898, 1.0000] | n/a | 0.023214 |
| sentiment_emotion | 59236 | survived_strong | 0.9968 / 1.0000 / 0.9927 | [0.9803, 1.0000] | n/a | 0.005134 |
| law | 494 | survived_weak | 0.9554 / 0.9649 / 0.9635 | [0.9287, 0.9879] | 1.0000 | 0.019562 |
| medicine | 1451 | survived_weak | 0.9500 / 0.9942 / 0.9788 | [0.9514, 0.9956] | 1.0000 | 0.009316 |
| code | 3721 | narrowed | 0.9036 / 0.9748 / 0.8830 | [0.7575, 0.9792] | n/a | 0.015645 |

Feature cards are in `feature_cards/`. The best causal feature card is `feature_cards/7849_finance.md`.

## Causal Tests

Matched causal suite outputs are in:

- `tables/causal_feature_tests.csv`
- `causal_feature_tests_summary.json`
- `plots/causal_operating_window.png`
- `causal_feature_card.md`

Results:

| run | selected feature | family | matched controls | best dose | real probe | control max at same dose | suppression | causal |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- | --- |
| `lab08_smoke_causal_suite_gpt2` | 12871 | sentiment_emotion | 10 | 0.0x | 3.2180 -> 3.2180 | 3.2180 | 17.1700 -> 9.3123 | false |
| `lab08_fairshot_olmo3_1025_7b_l16_v3_both_causal_s0` | 7849 | finance | 10 | 0.5x | 17.4229 -> 21.0273 | 17.3332 | 36.8935 -> 35.0584 | true |

The Olmo causal claim passes the current matched suite: the real feature beats matched controls before fluency collapse and suppression reduces the feature's domain probe on positive prompts. The keyword-hit effect is small, so the causal evidence is probe-led rather than sample-led.

The legacy Olmo clamp remains notable: feature 1265/494-style law features can move law keyword hits from 0 to 5 or 6 while the random control remains 0. This is now treated as supporting evidence, not a standalone causal claim.

## Negative And Failure Cases

- Historical GPT-2 run6/A remains a clean negative under the old setup: no survivors and code clamp 0 -> 0.
- GPT-2 layer 8 matched causal suite did not find a usable sentiment/emotion clamp; best dose was baseline and `causal=false`.
- GPT-2 layer 8 legacy code clamp was still negative in the fair-shot run.
- Olmo `code` improved to `narrowed` but did not meet the stronger semantic claim grade on test.
- Olmo finance F7849 is strong on AUC and causal probe score, but top contexts include legal/filing rows; top-20 purity is 0.6 and entropy is 1.5332. This supports partial positive, not a fully clean semantic concept claim.

## External SAE Search

Additional public SAE candidates were checked live and recorded as available or deferred:

- `jbloom/GPT2-Small-SAEs-Reformatted`: available and run for GPT-2 residual layers 4, 8, and 11.
- `decoderesearch/olmo-3-saes`: available and run for Olmo-3-1025-7B layer 16.
- `Solshine/deception-v4-saes-pythia-160m`: public files exist for Pythia-160M TopK/JumpReLU layers 3, 6, and 9, but the current loader expects the W_enc/W_dec/b_enc/b_dec style and this was not fully calibrated in this pass.
- `google/gemma-scope` and `google/gemma-scope-2`: public Gemma Scope releases exist, but available model sizes and SAELens-style `params.safetensors` layouts need a separate loader path and were not run.
- `decoderesearch/gemma-4-saes`: public SAE files exist, but matching model availability/license and loader compatibility were not smoke-tested.

This is the main remaining limitation of the sweep.

## Run Directories

Local:

- `runs/lab08_existing_run_audit.md`
- `runs/lab08_existing_run_audit.csv`
- `runs/lab08_smoke_sae_registry`
- `runs/lab08_smoke_v3_corpus`
- `runs/lab08_smoke_targeted_v3`
- `runs/lab08_smoke_targeted_v3_fix`
- `runs/lab08_smoke_targeted_grade`
- `runs/lab08_smoke_causal_suite_gpt2`
- `runs/lab08_fairshot_gpt2_jbloom_l4_v3_both_s0`
- `runs/lab08_fairshot_gpt2_jbloom_l8_v3_both_s0`
- `runs/lab08_fairshot_gpt2_jbloom_l11_v3_both_s0`
- `runs/lab08_fairshot_olmo3_1025_7b_decoderesearch_l16_v3_both_s0`
- `runs/lab08_fairshot_olmo3_1025_7b_l16_v3_both_causal_s0`

Drive backups:

- `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_smoke_sae_registry`
- `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_smoke_v3_corpus`
- `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_smoke_targeted_v3`
- `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_smoke_targeted_v3_fix`
- `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_smoke_targeted_grade`
- `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_smoke_causal_suite_gpt2`
- `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_fairshot_gpt2_jbloom_l4_v3_both_s0`
- `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_fairshot_gpt2_jbloom_l8_v3_both_s0`
- `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_fairshot_gpt2_jbloom_l11_v3_both_s0`
- `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_fairshot_olmo3_1025_7b_decoderesearch_l16_v3_both_s0`
- `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_fairshot_olmo3_1025_7b_l16_v3_both_causal_s0`

The final report is also copied to `/content/drive/MyDrive/interpret/lab08_sae_fairshot_20260620/lab08_fairshot_report.md`.

## Final Assessment

The original weak spots are materially improved: Lab 8 now has explicit SAE selection, loading calibration, a richer corpus, split-aware targeted discovery, stronger validation, and a matched causal suite. The strongest answer to the lab question is yes, partially: a pretrained SAE on a course-accessible base model can recover robustly labelable features, and at least one feature is causally usable under matched controls. The claim should remain partial because the best causal feature is not perfectly pure, the causal effect is stronger in probe score than keyword counts, and the broader public-SAE sweep was only partially executed due loader and model compatibility constraints.
