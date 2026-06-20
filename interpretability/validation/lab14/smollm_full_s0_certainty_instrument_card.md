# Lab 14 Certainty Instrument Card

**Verdict:** `usable_certainty_instrument`

The run validates a downstream-usable answerability direction under the current controls.

## Headline metrics

- model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- items: 80
- certainty depth selected on train split: 10
- certainty eval AUC: 0.92
- certainty eval control gap: 0.3236
- mean family-held-out real AUC: 0.8563
- mean family-held-out control gap: 0.2944
- hedging projection answerability AUC: 0.6178
- distribution-confidence answerability AUC: 0.4511
- max length/letter/answer-frame baseline answerability AUC: 0.6667
- verbal confidence ECE: 0.3167

## Read before reuse

The saved `state/certainty_direction.pt` is an answerability direction in a fixed A/B/C/D frame. It is not a direct measurement of subjective confidence, knowledge, belief, or honesty. Downstream labs should project it only with its metadata and should carry the verdict above into their own ledger entries.

## First artifacts to inspect

1. `tables/depth_selection.csv` - make sure depth selection was not a pretty-curve pick.
2. `tables/family_heldout_generalization.csv` - check whether the direction transfers across families.
3. `tables/signal_predictiveness.csv` - compare the internal direction against entropy, verbal confidence, hedging, length, and answer-key baselines.
4. `tables/disagreement_examples.csv` - choose a concrete case before writing any SELF-REPORT claim.