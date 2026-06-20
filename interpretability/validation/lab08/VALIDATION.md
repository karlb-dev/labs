# Lab 08 Validation

## Lab 8: Superposition, Sparse Autoencoders, and Transcoders

Superposition, SAEs, and transcoders: find, label, and validate features.

## Validation Read

The 2026-06-20 fair-shot rerun materially changes the Lab 8 read. Earlier
run6/verify_part3 artifacts made the lab look mostly like a skepticism lesson:
reasonable reconstruction, many tempting labels killed, and only weak keyword
clamp evidence. The new fair-shot suite is a partial positive.

The current best result is:

- `interpretability/runs/lab08_sae_fairshot_20260620/lab08_fairshot_olmo3_1025_7b_l16_v3_both_causal_s0`
  - Model: `allenai/Olmo-3-1025-7B`
  - SAE: `decoderesearch/olmo-3-saes`, layer 16 `resid_post`, d_sae 65536
  - Reconstruction: FVU 0.3602, L0 113.49, silent fraction 47.28%
  - Blind atlas: 3 survived, 9 narrowed, 4 polysemantic, 1 token-feature, 13 killed
  - Targeted search over 20 families: 6 `survived_strong`, 5 `survived_weak`, 5 `lexical_valid`, 3 `narrowed`, 1 `killed`
  - Best causal feature: feature 7849, `finance`
  - Train/dev/test AUC: 0.9303 / 0.9538 / 1.0000
  - Test confusable AUC: 1.0000
  - Fire fraction: 0.012879
  - Matched causal suite: `causal=true`
  - Causal probe: 17.4229 -> 21.0273 at 0.5x peak activation
  - Matched-control max at same dose: 17.3332
  - Suppression probe: 36.8935 -> 35.0584

This is not a clean victory lap. The strongest Olmo feature has top-20 context
purity 0.6, some high activations include legal/filing text, and the causal
effect is stronger in the learned probe than in keyword-hit counts. The right
claim is: Lab 8 can now find robustly labelable SAE features under split-aware
validation, and at least one Olmo feature passes the matched-control causal
suite.

## What Changed In The New Run

- Added an explicit SAE registry and CLI SAE selection.
- Added SAE loading reports and calibration sweeps for centering,
  decoder-bias subtraction, ReLU vs JumpReLU, and layer/site checks.
- Rebuilt the SAE corpus as deterministic v3: 1,200 rows, 20 families, 60 rows
  per family, with 720/240/240 train/dev/test split.
- Added supervised targeted feature search with train discovery, dev
  selection, one held-out test report, bootstrap AUC intervals, permutation
  null AUC, confusable validation, subset stability, and explicit claim grades.
- Added feature cards for selected family features.
- Added a matched causal suite with neutral prompts, dose sweep, suppression
  prompts, 10 matched controls, a corpus-trained lexical probe, keyword hits,
  plots, and causal feature cards.

## New Result Summary

| Source | Model / SAE | Main result | Causal read |
|---|---|---|---|
| `lab08_fairshot_olmo3_1025_7b_l16_v3_both_causal_s0` | Olmo-3-1025-7B, decoderesearch L16 JumpReLU SAE | 6 strong + 5 weak semantic/feature claims across 20 families | Positive for finance F7849 under matched controls |
| `lab08_smoke_causal_suite_gpt2` | GPT-2, jbloom L8 SAE | sentiment/emotion F12871 is labelable | Negative: best causal dose is 0.0x and probe does not move |
| `lab08_fairshot_gpt2_jbloom_l4_v3_both_s0` | GPT-2, jbloom L4 SAE | 9 strong, 1 weak, 7 lexical-valid claims | No headline causal pass in this pack |
| `lab08_fairshot_gpt2_jbloom_l8_v3_both_s0` | GPT-2, jbloom L8 SAE | 6 strong, 5 weak, 7 lexical-valid claims | Matched causal test negative in the smoke causal suite |
| `lab08_fairshot_gpt2_jbloom_l11_v3_both_s0` | GPT-2, jbloom L11 SAE | 8 strong, 2 weak, 6 lexical-valid claims | Legacy clamp plot exists, but no matched-suite positive |

## Historical Baseline

The historical audit confirms why the old validation read was skeptical.

| Source | Model | Old read |
|---|---|---|
| `run6/A` | `gpt2` | FVU 0.0019, L0 74.57, no survived atlas rows, code clamp 0 -> 0 |
| `run6/B` / `run6/C` | `allenai/Olmo-3-1025-7B` | FVU 0.3761, L0 113.54, no survived atlas rows, law feature clamp 0 -> 5 keyword hits |
| `lab08_tierb_validate` | `allenai/Olmo-3-1025-7B` | 1 survived feature and a weak law/emotion-style clamp, but with old keyword-only causal evidence |

The fair-shot run does not erase those negatives; it explains them. The old
blind atlas was too brittle as the main discovery mechanism, and the old causal
check did not have matched controls.

## What This Lab Teaches

- SAE validation depends heavily on loader correctness, activation convention,
  corpus design, and split-aware search.
- Blind atlas labeling is useful for discovery, but targeted search with heldout
  validation is much better at finding real candidate features.
- A feature can be decodable and labelable without being causally useful.
- A causal claim should compare the real feature against matched controls before
  treating a clamp as meaningful.
- Negative results still matter: GPT-2 has labelable features in this run, but
  the matched causal suite did not find a usable sentiment/emotion clamp.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpretability/runs/lab08_sae_fairshot_20260620/lab08_fairshot_olmo3_1025_7b_l16_v3_both_causal_s0` | `allenai/Olmo-3-1025-7B` | b | strongest new result; finance F7849 causal positive |
| `interpretability/runs/lab08_sae_fairshot_20260620/lab08_smoke_causal_suite_gpt2` | `gpt2` | a | matched causal negative for sentiment/emotion F12871 |
| `interpretability/runs/lab08_sae_fairshot_20260620/lab08_fairshot_gpt2_jbloom_l4_v3_both_s0` | `gpt2` | a | best GPT-2 layer by strong targeted-search count |
| `interpretability/runs/lab08_sae_fairshot_20260620/lab08_fairshot_gpt2_jbloom_l8_v3_both_s0` | `gpt2` | a | layer used for matched causal negative |
| `interpretability/runs/lab08_sae_fairshot_20260620/lab08_fairshot_gpt2_jbloom_l11_v3_both_s0` | `gpt2` | a | deeper GPT-2 comparison layer |
| `interpretability/runs/lab08_sae_fairshot_20260620/lab08_existing_run_audit.md` | mixed | mixed | historical comparison against run6/verify artifacts |

## Curated Artifacts

Fair-shot reports and audit:

- `fairshot_20260620_report.md`
- `fairshot_20260620_existing_run_audit.md`
- `fairshot_20260620_existing_run_audit.csv`

Olmo causal-positive package:

- `olmo3_1025_7b_l16_fairshot_causal_run_summary.md`
- `olmo3_1025_7b_l16_fairshot_causal_metrics.json`
- `olmo3_1025_7b_l16_fairshot_sae_loading_report.json`
- `olmo3_1025_7b_l16_feature7849_finance_card.md`
- `olmo3_1025_7b_l16_finance7849_causal_feature_card.md`
- `olmo3_1025_7b_l16_finance7849_causal_summary.json`
- `olmo3_1025_7b_l16_finance7849_causal_feature_tests.csv`
- `olmo3_1025_7b_l16_finance7849_causal_operating_window.png`
- `olmo3_1025_7b_l16_fairshot_best_feature_per_family.csv`
- `olmo3_1025_7b_l16_fairshot_feature_evidence_matrix.csv`
- `olmo3_1025_7b_l16_fairshot_feature_evidence_dashboard.png`
- `olmo3_1025_7b_l16_fairshot_feature_validation_matrix.png`
- `olmo3_1025_7b_l16_fairshot_domain_validation_summary.png`

GPT-2 causal-negative package:

- `gpt2_l8_fairshot_causal_negative_run_summary.md`
- `gpt2_l8_fairshot_causal_negative_metrics.json`
- `gpt2_l8_sentiment12871_causal_negative_feature_card.md`
- `gpt2_l8_sentiment12871_causal_negative_summary.json`
- `gpt2_l8_sentiment12871_causal_feature_tests.csv`
- `gpt2_l8_sentiment12871_causal_negative_operating_window.png`

GPT-2 fair-shot layer comparisons:

- `gpt2_l4_fairshot_best_feature_per_family.csv`
- `gpt2_l8_fairshot_best_feature_per_family.csv`
- `gpt2_l11_fairshot_best_feature_per_family.csv`
- `gpt2_l4_fairshot_feature_validation_matrix.png`
- `gpt2_l8_fairshot_feature_validation_matrix.png`
- `gpt2_l11_fairshot_feature_validation_matrix.png`

Older artifacts retained for comparison:

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

- This validation directory is curated; the raw run directory remains the source
  of truth for exact configs, logs, and full tables.
- The broader public-SAE sweep was only partial: GPT-2 jbloom and Olmo
  decoderesearch SAEs were fully run, while Gemma Scope / Gemma-4 /
  Pythia-family candidates need loader compatibility work.
- The best Olmo causal feature is not perfectly pure. Treat it as a partial
  positive causal handle, not a complete semantic concept proof.
- The causal evidence is probe-led; keyword-hit gains are small.
