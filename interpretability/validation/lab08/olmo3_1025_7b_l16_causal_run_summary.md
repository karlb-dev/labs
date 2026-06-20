# Lab 8 run summary: superposition, SAEs, and transcoders

## Run identity

- model: `allenai/Olmo-3-1025-7B` (base model; SAE/transcoder are pretrained, pinned)
- SAE layer 16, d_sae 65536; transcoder on gpt2
- evidence level: OBS/DECODE at the feature level, CAUSAL for the one clamped feature

## 1. Superposition, demonstrated (Part 0)

- toy model: 5 features represented when dense vs 17 when sparse, in only 5 dimensions — more features than dimensions, packed in superposition as sparsity rises.

## 2. Feature atlas (Part 1)

- reconstruction FVU 0.3602, per-token L0 ≈ 113.49, 47.3% of features silent on the corpus
- ranking overlap (max-activation vs frequency, top N): 1 — the two rankings surface largely different features (the disagreement is the lesson, not a bug)
- of 30 labeled features, 3 survived validation and 18 were killed (token-feature / polysemantic / low-AUC). The killed count is required; a clean sheet is a warning.
- targeted final validation search used train for discovery, dev for selection, and test for the selected
  feature per family across 20 families; grades: killed=1, lexical_valid=5, narrowed=3, survived_strong=6, survived_weak=5

## 3. Transcoder (Part 2)

- skipped by `--skip-transcoder` for the SAE final validation sweep.

## 4. Bridges and causal extension

- Lab 4 truth direction: no Lab 4 truth_direction.pt found
- feature clamp (CAUSAL): feature 1265 ('law') 0→6 keyword hits at 1.5× peak vs random 0; causal=True
- matched-control causal suite: feature 7849 ('finance') probe 17.4229→21.0273 at 0.5× peak; same-dose control max 17.3332; suppression 36.8935→35.0584; causal=True

## 5. Claims

- `L08-C1` OBS: A sparse autoencoder at layer 16 of allenai/Olmo-3-1025-7B reconstructs its activations at FVU 0.360 with ~113.49 active features per token out of 65536, and 30986 features stay silent on the 45189-token corpus — superposition made into a usable, sparse code.
  - falsifier: FVU is no better than reconstructing from the same number of random directions, or L0≈d_sae (no sparsity).
- `L08-C2` DECODE: SAE feature 922 is labeled 'dialogue' and the label SURVIVED under validation: held-out AUC 1.00 against domain membership. Of 30 labeled features, 3 survived and 18 were killed by the same battery.
  - falsifier: On a fresh corpus the held-out AUC collapses, or the label fires equally on the confusable domain — it tracked a token.
- `L08-C3` CAUSAL: Clamping validated feature 1265 ('law') along its decoder direction at 1.5× its peak activation pushes generations toward law vocabulary (0→6 keyword hits) while staying fluent, where a random feature's direction reaches only 0 — the feature is causally sufficient to move the behavior, not just decodable. Past ~3× peak the clamp collapses generation into repetition (see the CSV).
  - falsifier: The random-feature control matches the clamped feature — the effect was generic perturbation, not this feature.

## 6. The reading order

Diagnostics first, then the artifacts that make the distinctions visible.

1. `diagnostics/model_anatomy.json` — confirm base model + layer; loading conventions matter.
2. `feature_atlas.md` + `tables/feature_atlas.csv` — the deliverable. **Look for** the required
   dead labels (in the reference gpt2 run, high-purity 'code' features with held-out AUC ~0.57
   were the teaching case), the confusable-pair numbers that separate concept from token, and
   the explicit 'What the atlas does NOT show' section. A clean sheet of 'survived' is a
   warning sign.
3. `plots/feature_evidence_dashboard.png` — the whole lab packet on one page: toy geometry,
   SAE health, label verdicts, transcoder, truth bridge, and clamp status.
4. `plots/feature_validation_matrix.png` + `tables/feature_evidence_matrix.csv` — the label
   locks side by side: held-out AUC, confusable AUC, purity, low-polysemy score, and sparse firing.
5. `plots/toy_superposition_geometry.png` and `plots/toy_superposition_phase_diagram.png` —
   predict the geometry (dense: exactly d_hidden orthogonal; sparse: more features via
   accepted interference) before you look.
6. `plots/ranking_disagreement.png` + `tables/feature_rankings.csv` — **look for little or no
   overlap** between the red (max-act, rare high-peak outliers) and green (freq, broad basis
   vectors); the reference run had 0.
7. `plots/sae_activity_dashboard.png` and `plots/domain_validation_summary.png` — separate
   ordinary dictionary sparsity from the few features you are tempted to name.
8. `plots/atlas_verdicts.png` — count the killed bar; the lab wants dead labels.
9. `transcoder_reconstruction_report.json`, `plots/transcoder_feature_cards.png`, and
   `tables/transcoder_feature_promotes.csv` — FVU + splice-in KL + de-embedded promoted tokens.
10. `plots/truth_bridge_feature_cosines.png` — the Lab 4 truth direction is compared against
    SAE decoder atoms instead of being assumed to be one feature.
11. `plots/feature_clamp.png`, `plots/clamp_operating_window.png`, and `tables/feature_clamp.csv`
    — the single CAUSAL claim. **Read the sample generations** at each dose (not just hits).
    Expect a narrow window (reference run: induce ~1× peak, collapse by ~3×); random stays at
    or near 0; the distinct ratio flags repetition.

## 7. Caveats

- Validation is corpus-bound: a label that survives here can die on different text. The confusable
  pairs are the built-in guard against mistaking a token for a concept.
- 'Silent on corpus' ≠ dead; most of the dictionary simply never gets the inputs that fire it here.
- Decodability is not causality. Only the clamped, control-tested feature earns a CAUSAL tag.
- The SAE conventions (centering, b_dec, jumprelu) are validated, not assumed; a wrong convention
  inflates FVU silently. See the handout's debugging table.
