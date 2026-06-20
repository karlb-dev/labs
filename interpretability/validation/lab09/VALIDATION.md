# Lab 09 Validation

## Lab 9: Attribution Graphs and Circuit Tracing

Attribution graphs: a transcoder replacement model, feature-level circuit tracing, and interventions.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/run6/C/lab9` (gpt2, tier c)
  - Metrics: `baseline_gate_dropped`=1, `induction_metric_logit_diff`=0.6222, `kept_coverage_of_feature_mass`=0.4435, `metric_logit_diff`=3.013, `n_feature_nodes`=48, `n_paraphrases`=7, `n_recurring_features`=9, `node_budget`=48
  - model: `gpt2` + `jacobdunefsky/gpt2small-transcoders` (full 12-layer MLP transcoder stack)
  - primary fact: `The capital of France is` -> ` Paris` (vs ` Berlin`)
  - evidence level: ATTR for the graph, CAUSAL only for the intervention rows
- `interpret/run6/B/lab9` (gpt2, tier b)
  - Metrics: `baseline_gate_dropped`=1, `induction_metric_logit_diff`=0.6222, `kept_coverage_of_feature_mass`=0.3696, `metric_logit_diff`=3.013, `n_feature_nodes`=28, `n_paraphrases`=7, `n_recurring_features`=7, `node_budget`=28
  - model: `gpt2` + `jacobdunefsky/gpt2small-transcoders` (full 12-layer MLP transcoder stack)
  - primary fact: `The capital of France is` -> ` Paris` (vs ` Berlin`)
  - evidence level: ATTR for the graph, CAUSAL only for the intervention rows
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab09_tierc_labs1_25_full_matrix_20260615_000508/lab09_tierc_labs1_25_full_matrix_20260615_000508` (gpt2, tier c)
  - Metrics: `baseline_gate_dropped`=1, `induction_metric_logit_diff`=0.6222, `kept_coverage_of_feature_mass`=0.4435, `metric_logit_diff`=3.013, `n_feature_nodes`=48, `n_paraphrases`=7, `n_recurring_features`=9, `node_budget`=48
  - model: `gpt2` + `jacobdunefsky/gpt2small-transcoders` (full 12-layer MLP transcoder stack)
  - primary fact: `The capital of France is` -> ` Paris` (vs ` Berlin`)
  - evidence level: ATTR for the graph, CAUSAL only for the intervention rows
- `interpret/validate_part2_plots/lab9` (unknown model)
  - model: `gpt2` + `jacobdunefsky/gpt2small-transcoders` (full 12-layer MLP transcoder stack)
  - primary fact: `The capital of France is` -> ` Paris` (vs ` Berlin`)
  - evidence level: ATTR for the graph, CAUSAL only for the intervention rows

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/run6/C/lab9` | `gpt2` | `c` | `baseline_gate_dropped`=1; `induction_metric_logit_diff`=0.6222; `kept_coverage_of_feature_mass`=0.4435 |
| `interpret/run6/B/lab9` | `gpt2` | `b` | `baseline_gate_dropped`=1; `induction_metric_logit_diff`=0.6222; `kept_coverage_of_feature_mass`=0.3696 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab09_tierc_labs1_25_full_matrix_20260615_000508/lab09_tierc_labs1_25_full_matrix_20260615_000508` | `gpt2` | `c` | `baseline_gate_dropped`=1; `induction_metric_logit_diff`=0.6222; `kept_coverage_of_feature_mass`=0.4435 |
| `interpret/validate_part2_plots/lab9` | `unknown` | `` | see copied summaries |

## Curated Artifacts

- `gpt2_run6c_graph_evidence_dashboard.png`
- `gpt2_run6c_paraphrase_feature_matrix.png`
- `gpt2_run6c_tables_paraphrase_feature_matrix.csv`
- `gpt2_run6c_results.csv`
- `gpt2_run6b_graph_evidence_dashboard.png`
- `gpt2_run6b_paraphrase_feature_matrix.png`
- `gpt2_run6b_tables_paraphrase_feature_matrix.csv`
- `gpt2_run6b_results.csv`
- `gpt2_lab09_tierc_labs1_25_full_matrix_20260615_000508_graph_evidence_dashboard.png`
- `gpt2_lab09_tierc_labs1_25_full_matrix_20260615_000508_source_token_ledger.png`
- `gpt2_lab09_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `gpt2_lab09_tierc_labs1_25_full_matrix_20260615_000508_metrics.json`
- `unknown_lab9_graph_evidence_dashboard.png`
- `unknown_lab9_paraphrase_feature_matrix.png`
- `unknown_lab9_run_summary.md`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
