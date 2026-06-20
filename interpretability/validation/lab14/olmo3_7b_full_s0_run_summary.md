# Lab 14 Run Summary: lab14_validation_olmo3_7b_full_s0_20260620

- Model: `allenai/Olmo-3-7B-Instruct`
- Items: 80
- Verdict: `usable_certainty_instrument`
- Best certainty depth: 11
- Certainty eval AUC: 0.8889
- Certainty eval control gap: 0.3698
- Best hedging depth: 9
- Hedging-style eval AUC: 1.0
- Mean family-held-out real AUC: 0.8282
- Internal/distribution correlation on eval: -0.1104
- Internal/verbal correlation on eval: 0.6943
- Verbal confidence ECE: 0.4533

Start with `certainty_instrument_card.md`, then inspect `tables/disagreement_examples.csv` before writing the SELF-REPORT claim. The disagreement case is the tiny trapdoor where the lab becomes useful.