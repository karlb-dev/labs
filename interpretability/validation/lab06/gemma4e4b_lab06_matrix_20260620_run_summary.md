# Lab 6 run summary: recall (heads_only)

## Run identity

- model: `google/gemma-4-E4B-it` (42 blocks x 8 heads)
- behavior `recall`, scope `heads_only`, run length 8
- 15 discovery + 5 held-out prompts
- primary intervention: resample/interchange ablation; mean ablation reported as comparison

## Verdict

- **OVERFIT / OVER-RECOVERY** -- held-out resample faithfulness 2.25 > 1.25: the complement was suppressing the metric (brake removal), not a clean transferable circuit.

## Headline

- base metric +3.185; knee circuit 6 nodes: L41H1, L40H0, L18H6, L40H3, L40H5, L38H1
- discovery faithfulness: resample 1.404, mean 1.184
- held-out faithfulness: resample 2.247, mean 2.171
- motif-core held-out (resample): 2.014
- knee-minus-floor faithfulness gap: +0.241
- mean-minus-resample gap (discovery): -0.220
- suppression heads: 12; positive-causal MLPs: 1
- edge: none claimed

## Claims

- `L06-C1` CAUSAL: On recall in google/gemma-4-E4B-it (heads_only), the knee circuit (L41H1, L40H0, L18H6, L40H3, L40H5, L38H1) has discovery faithfulness 1.404 (resample) / 1.184 (mean) and held-out 2.247 (resample). Verdict: OVERFIT / OVER-RECOVERY.
  - falsifier: Held-out resample faithfulness below the floor, a motif-core that transfers as well as the full knee, or a different off distribution changing the verdict.
- `L06-C2` CAUSAL: Mean-minus-resample faithfulness gap on discovery is -0.220 with 12 suppression heads detected: evidence on whether mean ablation inflates faithfulness via brake removal.
  - falsifier: Mean and resample agree and no suppression heads exist; then the inflation claim is refuted.

## Reading order

1. `circuit_card.md` - the deliverable and the verdict.
2. `prompt_hygiene_report.md` - which prompts the gate kept or excluded.
3. `faithfulness_completeness_minimality.json` - knee/floor/motif-core under mean and resample.
4. `tables/knee_floor_selection.json` and `plots/prune_trajectory.png` - knee vs floor.
5. `metrics.json` - verdict, suppression heads, MLP contribution, mean-vs-resample gap.
