# Lab 6 run summary: taskvec (heads_and_mlps)

## Run identity

- model: `allenai/Olmo-3-1125-32B` (64 blocks x 40 heads)
- behavior `taskvec`, scope `heads_and_mlps`, run length 9
- 12 discovery + 4 held-out prompts
- primary intervention: resample/interchange ablation; mean ablation reported as comparison

## Verdict

- **OVERFIT / NO CLEAN CIRCUIT** -- no transferable subgraph: knee held-out resample 0.14 < 0.70.

## Headline

- base metric +5.797; knee circuit 46 nodes: MLP57, L34H9, L27H13, MLP56, MLP58, MLP34, L29H28, MLP28, MLP51, MLP31, L24H37, MLP55, L48H32, MLP48, MLP53, MLP37, L30H14, MLP42, MLP39, MLP29, MLP27, L54H11, L49H0, MLP16, MLP30, MLP20, MLP32, MLP50, MLP14, L44H10, L20H39, L57H35, MLP43, MLP33, MLP19, L27H16, L61H32, L25H32, L29H34, MLP26, MLP17, L35H10, MLP40, L30H38, L2H39, L20H5
- discovery faithfulness: resample 0.190, mean 0.631
- held-out faithfulness: resample 0.144, mean 0.789
- motif-core held-out (resample): 0.059
- knee-minus-floor faithfulness gap: -0.017
- mean-minus-resample gap (discovery): 0.441
- suppression heads: 1; positive-causal MLPs: 34
- edge: none claimed

## Claims

- `L06-C1` CAUSAL: On taskvec in allenai/Olmo-3-1125-32B (heads_and_mlps), the knee circuit (MLP57, L34H9, L27H13, MLP56, MLP58, MLP34, L29H28, MLP28, MLP51, MLP31, L24H37, MLP55, L48H32, MLP48, MLP53, MLP37, L30H14, MLP42, MLP39, MLP29, MLP27, L54H11, L49H0, MLP16, MLP30, MLP20, MLP32, MLP50, MLP14, L44H10, L20H39, L57H35, MLP43, MLP33, MLP19, L27H16, L61H32, L25H32, L29H34, MLP26, MLP17, L35H10, MLP40, L30H38, L2H39, L20H5) has discovery faithfulness 0.190 (resample) / 0.631 (mean) and held-out 0.144 (resample). Verdict: OVERFIT / NO CLEAN CIRCUIT.
  - falsifier: Held-out resample faithfulness below the floor, a motif-core that transfers as well as the full knee, or a different off distribution changing the verdict.
- `L06-C2` CAUSAL: Mean-minus-resample faithfulness gap on discovery is 0.441 with 1 suppression heads detected: evidence on whether mean ablation inflates faithfulness via brake removal.
  - falsifier: Mean and resample agree and no suppression heads exist; then the inflation claim is refuted.

## Reading order

1. `circuit_card.md` - the deliverable and the verdict.
2. `prompt_hygiene_report.md` - which prompts the gate kept or excluded.
3. `faithfulness_completeness_minimality.json` - knee/floor/motif-core under mean and resample.
4. `tables/knee_floor_selection.json` and `plots/prune_trajectory.png` - knee vs floor.
5. `metrics.json` - verdict, suppression heads, MLP contribution, mean-vs-resample gap.
