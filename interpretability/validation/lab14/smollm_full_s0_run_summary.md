# Lab 14 Run Summary: lab14_validation_smollm_full_s0_20260620

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Items: 80
- Verdict: `usable_certainty_instrument`
- Best certainty depth: 10
- Certainty eval AUC: 0.92
- Certainty eval control gap: 0.3236
- Best hedging depth: 16
- Hedging-style eval AUC: 0.9244
- Mean family-held-out real AUC: 0.8563
- Internal/distribution correlation on eval: -0.0984
- Internal/verbal correlation on eval: -0.5319
- Verbal confidence ECE: 0.3167

Start with `certainty_instrument_card.md`, then inspect `tables/disagreement_examples.csv` before writing the SELF-REPORT claim. The disagreement case is the tiny trapdoor where the lab becomes useful.