# Lab 6 run summary: successor (heads_and_mlps)

## Run identity

- model: `gpt2` (12 blocks x 12 heads)
- behavior `successor`, scope `heads_and_mlps`, run length 8
- 12 discovery + 4 held-out prompts
- primary intervention: resample/interchange ablation; mean ablation reported as comparison

## Verdict

- **OVERFIT / NO CLEAN CIRCUIT** -- discovery passes but held-out resample faithfulness 0.40 < 0.70.

## Headline

- base metric +6.260; knee circuit 17 nodes: MLP0, MLP9, MLP10, MLP11, L9H1, MLP3, MLP7, MLP8, L10H7, MLP6, MLP1, MLP5, L5H1, L11H10, L0H1, L7H10, L3H7
- discovery faithfulness: resample 0.510, mean 1.112
- held-out faithfulness: resample 0.400, mean 0.684
- motif-core held-out (resample): -0.041
- knee-minus-floor faithfulness gap: +0.347
- mean-minus-resample gap (discovery): 0.603
- suppression heads: 2; positive-causal MLPs: 12
- edge: none claimed

## Claims

- `L06-C1` CAUSAL: On successor in gpt2 (heads_and_mlps), the knee circuit (MLP0, MLP9, MLP10, MLP11, L9H1, MLP3, MLP7, MLP8, L10H7, MLP6, MLP1, MLP5, L5H1, L11H10, L0H1, L7H10, L3H7) has discovery faithfulness 0.510 (resample) / 1.112 (mean) and held-out 0.400 (resample). Verdict: OVERFIT / NO CLEAN CIRCUIT.
  - falsifier: Held-out resample faithfulness below the floor, a motif-core that transfers as well as the full knee, or a different off distribution changing the verdict.
- `L06-C2` CAUSAL: Mean-minus-resample faithfulness gap on discovery is 0.603 with 2 suppression heads detected: evidence on whether mean ablation inflates faithfulness via brake removal.
  - falsifier: Mean and resample agree and no suppression heads exist; then the inflation claim is refuted.

## Reading order

1. `circuit_card.md` - the deliverable and the verdict.
2. `prompt_hygiene_report.md` - which prompts the gate kept or excluded.
3. `faithfulness_completeness_minimality.json` - knee/floor/motif-core under mean and resample.
4. `tables/knee_floor_selection.json` and `plots/prune_trajectory.png` - knee vs floor.
5. `metrics.json` - verdict, suppression heads, MLP contribution, mean-vs-resample gap.
