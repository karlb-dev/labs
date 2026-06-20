# Lab 14 Run Summary: lab14_validation_olmo3_7b_full_s1_20260620

- Model: `allenai/Olmo-3-7B-Instruct`
- Items: 80
- Verdict: `answerability_decodes_but_confounds_compete`
- Best certainty depth: 32
- Certainty eval AUC: 0.9556
- Certainty eval control gap: 0.4294
- Best hedging depth: 3
- Hedging-style eval AUC: 0.9933
- Mean family-held-out real AUC: 0.9781
- Internal/distribution correlation on eval: -0.5843
- Internal/verbal correlation on eval: 0.8083
- Verbal confidence ECE: 0.5333

Start with `certainty_instrument_card.md`, then inspect `tables/disagreement_examples.csv` before writing the SELF-REPORT claim. The disagreement case is the tiny trapdoor where the lab becomes useful.