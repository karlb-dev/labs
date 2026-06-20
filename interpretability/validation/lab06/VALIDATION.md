# Lab 06 Validation

## Lab 6: Circuit Discovery and Validation, the Manual Way

Circuit discovery, the manual way: a faithful, complete, minimal subgraph.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/lab06_circuit_discovery-20260620_043831-e14cfd` (allenai/Olmo-3-1125-32B, tier b)
  - Metrics: `base_metric`=7.917, `meets_faithfulness_floor`=True, `minimality_all_positive`=True, `minimality_worst_marginal`=0.01924, `n_discovery`=3, `n_heldout`=2, `n_heldout_positive`=2, `prune_stop_reason`=one head remains
  - model: `allenai/Olmo-3-1125-32B` (64 blocks x 40 heads)
  - task: induction completion, 3 discovery + 2 held-out prompts (2 baseline-positive for F/C/M)
  - evidence level: `CAUSAL` at heads-only circuit scope
- `interpret/lab06_matrix_20260620/gpt2/successor_heads_and_mlps` (gpt2, tier a)
  - Metrics: `aborted`=False, `base_metric`=6.26, `behavior`=successor, `floor_n_nodes`=8, `headline_discovery_faith_resample`=0.5096, `headline_heldout_faith_resample`=0.4002, `induction_motif_present`=True, `knee_minus_floor_faithfulness`=0.3466
  - `L06-C1` CAUSAL: On successor in gpt2 (heads_and_mlps), the knee circuit (MLP0, MLP9, MLP10, MLP11, L9H1, MLP3, MLP7, MLP8, L10H7, MLP6, MLP1, MLP5, L5H1, L11H10, L0H1, L7H10, L3H7) has discovery faithfulness 0.510 (resample) / 1.112 (mean) and held-out 0.4...
  - falsifier: Held-out resample faithfulness below the floor, a motif-core that transfers as well as the full knee, or a different off distribution changing the verdict.
  - `L06-C2` CAUSAL: Mean-minus-resample faithfulness gap on discovery is 0.603 with 2 suppression heads detected: evidence on whether mean ablation inflates faithfulness via brake removal.
- `interpret/lab06_matrix_20260620/google__gemma-4-E4B-it/recall_heads_only` (google/gemma-4-E4B-it, tier b)
  - Metrics: `aborted`=False, `base_metric`=3.185, `behavior`=recall, `floor_n_nodes`=2, `headline_discovery_faith_resample`=1.404, `headline_heldout_faith_resample`=2.247, `induction_motif_present`=False, `knee_minus_floor_faithfulness`=0.241
  - `L06-C1` CAUSAL: On recall in google/gemma-4-E4B-it (heads_only), the knee circuit (L41H1, L40H0, L18H6, L40H3, L40H5, L38H1) has discovery faithfulness 1.404 (resample) / 1.184 (mean) and held-out 2.247 (resample). Verdict: OVERFIT / OVER-RECOVERY.
  - falsifier: Held-out resample faithfulness below the floor, a motif-core that transfers as well as the full knee, or a different off distribution changing the verdict.
  - `L06-C2` CAUSAL: Mean-minus-resample faithfulness gap on discovery is -0.220 with 12 suppression heads detected: evidence on whether mean ablation inflates faithfulness via brake removal.
- `interpret/lab06_matrix_20260620/allenai__Olmo-3-1125-32B/taskvec_heads_and_mlps` (allenai/Olmo-3-1125-32B, tier b)
  - Metrics: `aborted`=False, `base_metric`=5.797, `behavior`=taskvec, `floor_n_nodes`=60, `headline_discovery_faith_resample`=0.1904, `headline_heldout_faith_resample`=0.1435, `induction_motif_present`=False, `knee_minus_floor_faithfulness`=-0.01687
  - `L06-C1` CAUSAL: On taskvec in allenai/Olmo-3-1125-32B (heads_and_mlps), the knee circuit (MLP57, L34H9, L27H13, MLP56, MLP58, MLP34, L29H28, MLP28, MLP51, MLP31, L24H37, MLP55, L48H32, MLP48, MLP53, MLP37, L30H14, MLP42, MLP39, MLP29, MLP27, L54H11, L49H0,...
  - falsifier: Held-out resample faithfulness below the floor, a motif-core that transfers as well as the full knee, or a different off distribution changing the verdict.
  - `L06-C2` CAUSAL: Mean-minus-resample faithfulness gap on discovery is 0.441 with 1 suppression heads detected: evidence on whether mean ablation inflates faithfulness via brake removal.

## What This Lab Teaches

- Treat the circuit card as a stress test of circuit claims: faithfulness, completeness, minimality, and held-out transfer all matter.
- The latest Lab 6 reruns are the headline evidence for this pack; older runs are useful mainly as before/after context.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/lab06_circuit_discovery-20260620_043831-e14cfd` | `allenai/Olmo-3-1125-32B` | `b` | `base_metric`=7.917; `meets_faithfulness_floor`=True; `minimality_all_positive`=True |
| `interpret/lab06_matrix_20260620/gpt2/successor_heads_and_mlps` | `gpt2` | `a` | `aborted`=False; `base_metric`=6.26; `behavior`=successor |
| `interpret/lab06_matrix_20260620/google__gemma-4-E4B-it/recall_heads_only` | `google/gemma-4-E4B-it` | `b` | `aborted`=False; `base_metric`=3.185; `behavior`=recall |
| `interpret/lab06_matrix_20260620/allenai__Olmo-3-1125-32B/taskvec_heads_and_mlps` | `allenai/Olmo-3-1125-32B` | `b` | `aborted`=False; `base_metric`=5.797; `behavior`=taskvec |
| `interpret/lab06_matrix_20260620/google__gemma-4-E4B-it/recall_heads_and_mlps` | `google/gemma-4-E4B-it` | `b` | `aborted`=False; `base_metric`=3.185; `behavior`=recall |
| `interpret/lab06_matrix_20260620/google__gemma-4-E4B-it/induction_p3_heads_only` | `google/gemma-4-E4B-it` | `b` | `aborted`=False; `base_metric`=16.81; `behavior`=induction_p3 |
| `interpret/lab06_matrix_20260620/google__gemma-4-E4B-it/induction_p3_heads_and_mlps` | `google/gemma-4-E4B-it` | `b` | `aborted`=False; `base_metric`=16.81; `behavior`=induction_p3 |
| `interpret/lab06_matrix_20260620/google__gemma-4-E4B-it/induction_p2_heads_and_mlps` | `google/gemma-4-E4B-it` | `b` | `aborted`=False; `base_metric`=10.07; `behavior`=induction_p2 |

## Curated Artifacts

- `olmo3_32b_lab06_circuit_discovery-20260620_043831-e14cfd_causal_motif_atlas.png`
- `olmo3_32b_lab06_circuit_discovery-20260620_043831-e14cfd_circuit_discovery_dashboard.png`
- `olmo3_32b_lab06_circuit_discovery-20260620_043831-e14cfd_circuit_scorecard.png`
- `olmo3_32b_lab06_circuit_discovery-20260620_043831-e14cfd_results.csv`
- `olmo3_32b_lab06_circuit_discovery-20260620_043831-e14cfd_metrics.json`
- `olmo3_32b_lab06_circuit_discovery-20260620_043831-e14cfd_run_summary.md`
- `gpt2_lab06_matrix_20260620_causal_motif_atlas.png`
- `gpt2_lab06_matrix_20260620_circuit_discovery_dashboard.png`
- `gpt2_lab06_matrix_20260620_circuit_scorecard.png`
- `gpt2_lab06_matrix_20260620_results.csv`
- `gpt2_lab06_matrix_20260620_metrics.json`
- `gpt2_lab06_matrix_20260620_run_summary.md`
- `gemma4e4b_lab06_matrix_20260620_causal_motif_atlas.png`
- `gemma4e4b_lab06_matrix_20260620_circuit_discovery_dashboard.png`
- `gemma4e4b_lab06_matrix_20260620_circuit_scorecard.png`
- `gemma4e4b_lab06_matrix_20260620_results.csv`
- `gemma4e4b_lab06_matrix_20260620_metrics.json`
- `gemma4e4b_lab06_matrix_20260620_run_summary.md`
- `olmo3_32b_lab06_matrix_20260620_causal_motif_atlas.png`
- `olmo3_32b_lab06_matrix_20260620_circuit_discovery_dashboard.png`
- `olmo3_32b_lab06_matrix_20260620_circuit_scorecard.png`
- `olmo3_32b_lab06_matrix_20260620_results.csv`
- `olmo3_32b_lab06_matrix_20260620_metrics.json`
- `olmo3_32b_lab06_matrix_20260620_run_summary.md`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
