# Lab 08 Validation

## Lab 8: Superposition, Sparse Autoencoders, and Transcoders

Superposition, SAEs, and transcoders: find, label, and validate features.

## Validation Read

This directory is the final Lab 8 validation pack for the current repository
code. It keeps only the latest validation artifacts so students and reviewers
see one coherent result set.

The result is a partial positive. Lab 8 now finds robustly labelable SAE
features under split-aware validation, and one Olmo feature passes a
matched-control causal suite. The cleanest current claim is not "SAEs find
perfect concepts"; it is that the lab can identify candidate features, grade
their labels on held-out data, and separate decodable features from features
that are causally useful.

## Headline Result

The strongest result is the Olmo layer-16 finance feature:

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

This is still a scoped claim. The finance feature has top-20 context purity
0.6, some high activations include legal/filing text, and the causal effect is
stronger in the learned probe than in keyword-hit counts. The best wording is:

```text
The current Lab 8 validation finds robustly labelable SAE features and one
matched-control causal handle, but the best feature remains imperfectly pure.
```

## Current Result Summary

| Source label | Model / SAE | Main result | Causal read |
|---|---|---|---|
| Olmo L16 validation | Olmo-3-1025-7B, decoderesearch L16 JumpReLU SAE | 6 strong + 5 weak semantic/feature claims across 20 families | Positive for finance F7849 under matched controls |
| GPT-2 L8 causal check | GPT-2, jbloom L8 SAE | sentiment/emotion F12871 is labelable | Negative: best causal dose is 0.0x and probe does not move |
| GPT-2 L4 comparison | GPT-2, jbloom L4 SAE | 9 strong, 1 weak, 7 lexical-valid claims | No headline causal pass in this pack |
| GPT-2 L8 comparison | GPT-2, jbloom L8 SAE | 6 strong, 5 weak, 7 lexical-valid claims | Matched causal test negative |
| GPT-2 L11 comparison | GPT-2, jbloom L11 SAE | 8 strong, 2 weak, 6 lexical-valid claims | No matched-suite positive in this pack |

## What This Lab Teaches

- SAE validation depends heavily on loader correctness, activation convention,
  corpus design, and split-aware search.
- Blind atlas labeling is useful for discovery, but targeted search with
  held-out validation is better at finding real candidate features.
- A feature can be decodable and labelable without being causally useful.
- A causal claim should compare the real feature against matched controls before
  treating a clamp as meaningful.
- Negative controls still matter: GPT-2 has labelable features in this pack,
  but the matched causal suite did not find a usable sentiment/emotion clamp.

## Curated Artifacts

Summary:

- `lab08_validation_report.md`

Olmo causal-positive package:

- `olmo3_1025_7b_l16_causal_run_summary.md`
- `olmo3_1025_7b_l16_causal_metrics.json`
- `olmo3_1025_7b_l16_sae_loading_report.json`
- `olmo3_1025_7b_l16_feature7849_finance_card.md`
- `olmo3_1025_7b_l16_finance7849_causal_feature_card.md`
- `olmo3_1025_7b_l16_finance7849_causal_summary.json`
- `olmo3_1025_7b_l16_finance7849_causal_feature_tests.csv`
- `olmo3_1025_7b_l16_finance7849_causal_operating_window.png`
- `olmo3_1025_7b_l16_best_feature_per_family.csv`
- `olmo3_1025_7b_l16_feature_evidence_matrix.csv`
- `olmo3_1025_7b_l16_feature_evidence_dashboard.png`
- `olmo3_1025_7b_l16_feature_validation_matrix.png`
- `olmo3_1025_7b_l16_domain_validation_summary.png`

GPT-2 causal-negative package:

- `gpt2_l8_causal_negative_run_summary.md`
- `gpt2_l8_causal_negative_metrics.json`
- `gpt2_l8_sentiment12871_causal_negative_feature_card.md`
- `gpt2_l8_sentiment12871_causal_negative_summary.json`
- `gpt2_l8_sentiment12871_causal_feature_tests.csv`
- `gpt2_l8_sentiment12871_causal_negative_operating_window.png`

GPT-2 layer comparisons:

- `gpt2_l4_best_feature_per_family.csv`
- `gpt2_l8_best_feature_per_family.csv`
- `gpt2_l11_best_feature_per_family.csv`
- `gpt2_l4_feature_validation_matrix.png`
- `gpt2_l8_feature_validation_matrix.png`
- `gpt2_l11_feature_validation_matrix.png`

## Caveats

- This validation directory is curated; full raw runs remain outside this pack.
- The broader public-SAE sweep was only partial: GPT-2 jbloom and Olmo
  decoderesearch SAEs were fully run, while Gemma Scope / Gemma-4 /
  Pythia-family candidates need loader compatibility work.
- The best Olmo causal feature is not perfectly pure. Treat it as a partial
  positive causal handle, not a complete semantic concept proof.
- The causal evidence is probe-led; keyword-hit gains are small.
