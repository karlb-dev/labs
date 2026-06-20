# Lab 8 Final Validation Report

Date: 2026-06-20

Verdict: partial positive.

The current Lab 8 implementation can find robustly labelable SAE features under
split-aware validation, and one Olmo feature passes the matched-control causal
suite. The strongest positive is Olmo feature 7849 for `finance`:

- train/dev/test AUC: 0.9303 / 0.9538 / 1.0000
- test confusable AUC: 1.0000
- fire fraction: 0.012879
- matched causal probe: 17.4229 -> 21.0273 at 0.5x peak activation
- matched-control max at the same dose: 17.3332
- suppression probe: 36.8935 -> 35.0584

This is not a fully clean semantic-concept proof. The top-20 context purity is
0.6, some high activations are legal/filing text, and keyword-hit gains are
small. The result is still useful: the lab now gives SAEs a fair validation
path and produces a real causal handle while preserving the caveats.

## Validation Design

The current Lab 8 code adds the pieces needed for a stronger SAE result:

- explicit SAE registry and CLI SAE selection;
- loader reports and calibration sweeps over centering, decoder-bias
  subtraction, ReLU vs JumpReLU, and layer/site checks;
- deterministic v3 SAE corpus with 1,200 rows, 20 families, 60 rows per family,
  and a 720/240/240 train/dev/test split;
- supervised targeted feature search with train discovery, dev selection, one
  held-out test report, bootstrap AUC intervals, permutation null AUC,
  confusable validation, subset stability, label types, and claim grades;
- feature cards for selected family features;
- matched causal suite with neutral prompts, dose sweep, suppression prompts,
  10 matched controls, a corpus-trained lexical probe, keyword hits, plots,
  summary JSON, and causal feature cards.

## Corpus

Corpus: `data/sae_feature_corpus_v3.csv`

- Rows: 1,200
- Families: 20
- Rows per family: 60
- Split: 720 train, 240 dev, 240 test
- Semantic families: chemistry, code, cooking, emotion, finance, history, law,
  medicine, sports, weather
- SAE-native families: capitalization/acronyms, citations/legal references,
  code indentation/whitespace, dates/numbers/measurements, markdown/list
  formatting, named entities, Python syntax, quotes/dialogue,
  sentiment/emotion, URLs/emails/paths
- Dataset card: `data/sae_feature_corpus_v3_card.md`

## Model And SAE Specs

| Run family | Model | SAE repo | Layer/site | Activation | Dims | Chosen convention |
|---|---|---|---|---|---|---|
| GPT-2 sweep | `gpt2` | `jbloom/GPT2-Small-SAEs-Reformatted` | blocks 4, 8, 11 `hook_resid_pre` | ReLU | d_model 768, d_sae 24576 | center input, subtract `b_dec` |
| Olmo sweep | `allenai/Olmo-3-1025-7B` | `decoderesearch/olmo-3-saes` | layer 16 `resid_post` | JumpReLU | d_model 4096, d_sae 65536 | no centering, subtract `b_dec`, apply threshold |

Resolved specs and loader health are kept in the copied metrics/loading report
artifacts.

## Reconstruction Health

| Run family | Model | Layer | FVU | L0 | Silent | Calibration chosen FVU | Calibration best FVU |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-2 layer 4 | GPT-2 | 4 | 0.0021 | 42.54 | 0.3376 | 0.000868 | 0.000868 |
| GPT-2 layer 8 | GPT-2 | 8 | 0.0048 | 74.57 | 0.1258 | 0.001697 | 0.001697 |
| GPT-2 layer 11 | GPT-2 | 11 | 0.0171 | 58.73 | 0.1203 | 0.005601 | 0.005601 |
| Olmo layer 16 | Olmo | 16 | 0.3602 | 113.49 | 0.4728 | 0.437008 | 0.409444 |

For Olmo, the numerically best calibration omitted the documented JumpReLU
threshold and produced much higher L0. The chosen convention keeps the
documented thresholded JumpReLU path, accepting slightly worse reconstruction
for plausible sparsity.

## Blind Atlas Results

| Run family | Atlas rows | Verdicts |
|---|---:|---|
| GPT-2 layer 4 | 30 | 1 survived, 2 narrowed, 5 polysemantic, 22 killed |
| GPT-2 layer 8 | 30 | 2 survived, 3 narrowed, 2 polysemantic, 1 token-feature, 22 killed |
| GPT-2 layer 11 | 30 | 4 survived, 1 polysemantic, 25 killed |
| Olmo layer 16 | 30 | 3 survived, 9 narrowed, 4 polysemantic, 1 token-feature, 13 killed |

The blind atlas is a discovery aid rather than the final claim protocol. It
still misses some good low-peak targeted features.

## Targeted Search Results

| Run family | Claim grades over 20 families |
|---|---|
| GPT-2 layer 4 | 9 `survived_strong`, 1 `survived_weak`, 7 `lexical_valid`, 1 `narrowed`, 1 `token_feature_mislabeled`, 1 `killed` |
| GPT-2 layer 8 | 6 `survived_strong`, 5 `survived_weak`, 7 `lexical_valid`, 1 `narrowed`, 1 `killed` |
| GPT-2 layer 11 | 8 `survived_strong`, 2 `survived_weak`, 6 `lexical_valid`, 3 `narrowed`, 1 `killed` |
| Olmo layer 16 | 6 `survived_strong`, 5 `survived_weak`, 5 `lexical_valid`, 3 `narrowed`, 1 `killed` |

Key Olmo selected features:

| Family | Feature | Grade | Train/dev/test AUC | Test CI | Confusable AUC | Fire |
|---|---:|---|---|---|---:|---:|
| finance | 7849 | `survived_strong` | 0.9303 / 0.9538 / 1.0000 | [1.0000, 1.0000] | 1.0000 | 0.012879 |
| chemistry | 8207 | `survived_strong` | 1.0000 / 0.9982 / 1.0000 | [1.0000, 1.0000] | 1.0000 | 0.016066 |
| history | 986 | `survived_strong` | 0.9743 / 0.9996 / 0.9971 | [0.9898, 1.0000] | n/a | 0.023214 |
| sentiment_emotion | 59236 | `survived_strong` | 0.9968 / 1.0000 / 0.9927 | [0.9803, 1.0000] | n/a | 0.005134 |
| law | 494 | `survived_weak` | 0.9554 / 0.9649 / 0.9635 | [0.9287, 0.9879] | 1.0000 | 0.019562 |
| medicine | 1451 | `survived_weak` | 0.9500 / 0.9942 / 0.9788 | [0.9514, 0.9956] | 1.0000 | 0.009316 |
| code | 3721 | `narrowed` | 0.9036 / 0.9748 / 0.8830 | [0.7575, 0.9792] | n/a | 0.015645 |

The copied per-family tables are the compact source for the full selected
feature list.

## Causal Tests

| Run family | Selected feature | Family | Matched controls | Best dose | Real probe | Control max at same dose | Suppression | Causal |
|---|---:|---|---:|---:|---|---:|---|---|
| GPT-2 layer 8 | 12871 | sentiment_emotion | 10 | 0.0x | 3.2180 -> 3.2180 | 3.2180 | 17.1700 -> 9.3123 | false |
| Olmo layer 16 | 7849 | finance | 10 | 0.5x | 17.4229 -> 21.0273 | 17.3332 | 36.8935 -> 35.0584 | true |

The Olmo causal claim passes the current matched suite: the real feature beats
matched controls before fluency collapse, and suppression reduces the feature's
domain probe on positive prompts. The keyword-hit effect is small, so the
causal evidence is probe-led rather than sample-led.

## Negative And Failure Cases

- GPT-2 layer 8 did not produce a matched-suite causal handle for
  sentiment/emotion; the best dose was baseline and `causal=false`.
- Olmo `code` improved to `narrowed` but did not meet the stronger semantic
  claim grade on test.
- Olmo finance F7849 is strong on AUC and causal probe score, but top contexts
  include legal/filing rows; top-20 purity is 0.6 and entropy is 1.5332.

## External SAE Search

Additional public SAE candidates were checked and recorded as available or
deferred:

- `jbloom/GPT2-Small-SAEs-Reformatted`: available and run for GPT-2 residual
  layers 4, 8, and 11.
- `decoderesearch/olmo-3-saes`: available and run for Olmo-3-1025-7B layer 16.
- `Solshine/deception-v4-saes-pythia-160m`: public files exist for Pythia-160M
  TopK/JumpReLU layers 3, 6, and 9, but the current loader expects the
  W_enc/W_dec/b_enc/b_dec style and this was not fully calibrated in this pass.
- `google/gemma-scope` and `google/gemma-scope-2`: public Gemma Scope releases
  exist, but available model sizes and SAELens-style `params.safetensors`
  layouts need a separate loader path and were not run.
- `decoderesearch/gemma-4-saes`: public SAE files exist, but matching model
  availability/license and loader compatibility were not smoke-tested.

## Final Assessment

The current Lab 8 result is yes, partially: a pretrained SAE on a
course-accessible base model can recover robustly labelable features, and at
least one feature is causally usable under matched controls. The claim should
remain partial because the best causal feature is not perfectly pure, the
causal effect is stronger in probe score than keyword counts, and the broader
public-SAE sweep still has loader/model compatibility gaps.
