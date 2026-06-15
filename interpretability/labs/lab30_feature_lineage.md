# Lab 30: Cross-Layer Feature Lineage Without Feature-Identity Overclaiming

**One-sentence thesis:** A feature-like direction earns a lineage claim only when its label, activations, top contexts, and controls recur together across depth.

**Time estimate:** Tier A smoke in minutes on CPU; Tier B science on the course base model in a Colab/A100-style runtime.

**Compute tier:** Tier A uses `gpt2` on a balanced small slice; Tier B uses the course base model on the frozen corpus.

**Dependencies:** Labs 8, 19, and 29. Lab 31 can reuse the top-context and failure-specimen tables for automated interpretability audits.

**Minimum passing artifacts:** `method_card.md`, `operationalization_audit.md`, `metrics.json`, `results.csv`, `results.jsonl`, `diagnostics/warning_summary.csv`, `diagnostics/lab30_run_config_snapshot.json`, `tables/feature_lineage_nodes.csv`, `tables/feature_lineage_node_scores.csv`, `tables/feature_lineage_edges.csv`, `tables/causal_transfer_dose_response.csv`, `tables/evidence_matrix.csv`, `tables/label_stability_summary.csv`, `tables/cross_model_feature_overlap.csv`, `tables/failure_specimens.md`, `plots/plot_manifest.json`, `state/cross_layer_dictionary.pt`, `state/lineage_graph.json`, and `plots/overview_dashboard.png`.

**Main plot:** `plots/overview_dashboard.png`

**Main table:** `tables/evidence_matrix.csv`

**Evidence rung:** `DECODE + ATTR`, with a narrow marker-logit activation-addition probe.

**Forbidden claim:** "This is the same concept everywhere in the model."

**One-sentence allowed claim:** "A supervised prototype direction for domain F recurred across the selected depths above random-direction and confusable-domain controls on this frozen corpus."

**Human-label requirement:** none for the default domain-labeled corpus. Future auto-labeling extensions must add human-review columns.

## What question this lab asks

Can a feature-like direction be tracked across layers without pretending that similarity is identity?

Lab 30 uses supervised prototype directions as a cheap, inspectable first pass:

```text
direction(domain, depth) = mean(domain activations) - mean(other activations)
```

That is a teaching instrument, not a sparse autoencoder and not a crosscoder. The lab asks whether these domain directions are label-valid on held-out rows and whether adjacent-depth same-label edges beat random, off-label, and confusable-domain controls.

The visual pass adds a second rule: every figure must be traceable to a saved source table. A plot is not a postcard. It is a receipt with axes.

## Why this matters in the course progression

Lab 8 taught feature inspection at one site. Lab 19 compared model-diff features. Lab 29 asked when features or circuits appear over checkpoints. Lab 30 changes the axis: it asks whether a candidate feature family persists, splits, merges, or drifts across depth.

The danger is obvious: cosine similarity makes identity cosplay as math. This lab is built to make that overclaim expensive. A direction can recur geometrically and still fail held-out AUC. A same-label edge can look strong and still lose to a confusable domain. A marker-logit edit can move a token while telling you almost nothing about semantic behavior.

## Data

Default rows live in:

```text
data/feature_lineage_corpus.csv
```

Required columns:

```text
row_id, family, domain, source_lab, text, group_id, split, labels_json
```

The frozen starter corpus is organized into paired confusable groups:

| Group | Domains | Why the pair matters |
|---|---|---|
| `code_cooking` | `code`, `cooking` | Both are procedural; a shallow vocabulary/control failure is easy to spot. |
| `finance_sports` | `finance`, `sports` | Both contain scores, rankings, and outcomes. |
| `law_medicine` | `law`, `medicine` | Both use institutional/professional language. |
| `weather_emotion` | `weather`, `emotion` | Both contain condition/state language, one external and one internal. |

Each row's `labels_json` includes a marker token, a contrast token, and a confusable domain. Marker tokens are used only for the narrow activation-addition transfer probe.

## What the experiment measures

### Nodes: held-out decodability

A node is one `(domain, depth)` prototype direction. It records:

```text
train AUC
held-out eval AUC
random-direction AUC
random-control lift
top train/eval context IDs
```

The aggregate node table is:

```text
tables/feature_lineage_nodes.csv
```

The visual pass also writes the raw per-row projection table:

```text
tables/feature_lineage_node_scores.csv
```

That table is the antidote to aggregate fog. It lets you see whether one row, one confusable group, or one split is carrying the node AUC.

### Edges: adjacent-depth recurrence

An edge connects one domain at depth `d0` to one target domain at depth `d1`.

The edge score combines:

```text
direction cosine
activation-score correlation
top-context Jaccard
endpoint AUCs
```

The edge must beat:

```text
random-direction control
best off-label target
confusable-domain target
```

The main edge table is:

```text
tables/feature_lineage_edges.csv
```

Confusable-domain scores and gaps are recorded on the edge rows themselves as `confusable_control_score` and `confusable_gap`. A bright same-label edge that does not beat its confusable neighbor is a refinement result, not a lineage claim.

### Split/merge screens

The lab writes:

```text
tables/split_merge_candidates.csv
```

These rows look for source domains with several strong targets, target domains with several strong sources, and strongest-target label changes. They are hypothesis generators. Do not call them proof of feature splitting or merging.

### Cross-model placeholder

The default run does not compare two external pretrained model families. It writes:

```text
tables/cross_model_feature_overlap.csv
```

This is a same-model cross-layer overlap schema plus random controls. It exists so a later SAE, crosscoder, Pythia, OLMo, or model-family extension can reuse the columns. It is not external cross-model evidence.

### Marker-logit activation addition

The transfer probe adds a domain direction to a neutral prompt and measures:

```text
logit(marker_token) - logit(contrast_token)
```

The upgraded code runs a small dose sweep rather than a single dose:

```text
Tier A: scale_fraction_of_median_stream_norm in {0.0, 0.45}
Tier B/C: scale_fraction_of_median_stream_norm in {0.0, 0.15, 0.30, 0.45, 0.75}
```

The headline-by-depth and dose-response tables are:

```text
tables/causal_transfer_by_layer.csv
tables/causal_transfer_long.csv
tables/causal_transfer_dose_response.csv
```

This is a narrow marker-logit intervention. It is not semantic steering. If this plot looks impressive while the node/edge controls fail, the supported conclusion is "marker token handle," not "feature lineage."

## Depth discipline

Depth `0` is the embedding stream. The final depth is the final-norm input. Both are useful diagnostic rows, but formal lineage gates prefer interior depths when available.

The run records whether each node or edge is claimable in the depth grid. Boundary rows should stay visible in plots because they often diagnose token artifacts, but they should not be smuggled into lineage language.

## Controls and falsifiers

| Favorite story | Control or falsifier |
|---|---|
| The domain has a real recurring direction | Held-out AUC and random-direction AUC |
| Same-label edges track lineage | Random-direction edge score |
| The edge is label-specific | Best off-label edge and confusable-domain edge |
| The direction is not just vocabulary | Top activating contexts, raw projection rows, and confusable groups |
| A marker edit shows semantic control | It only moves marker-vs-contrast token logits; semantic claims require a generated-text audit |
| The run compares models | `cross_model_feature_overlap.csv` explicitly says external cross-model is not run |

## Running it

Run from `interpretability/`:

```bash
python interp_bench.py --lab lab30 --tier a --no-plots
python interp_bench.py --lab lab30 --tier a
python interp_bench.py --lab lab30 --tier b --prompt-set full
```

Useful variants:

```bash
python interp_bench.py --lab lab30 --tier b --prompt-set medium --max-examples 48
python interp_bench.py --lab lab30 --tier b --prompt-set data/feature_lineage_corpus.csv
```

Tier A proves the plumbing. Tier B is the evidence path.

## Artifact tree

```text
runs/lab30_feature_lineage-*/
  run_summary.md
  method_card.md
  operationalization_audit.md
  metrics.json
  results.csv
  results.jsonl
  ledger_suggestions.md

  diagnostics/
    data_manifest.json
    tokenization_gate.csv
    split_balance.csv
    activation_norms_by_depth.csv
    self_check_status.json
    safety_status.json
    warning_summary.csv
    warning_summary.json
    lab30_run_config_snapshot.json
    residual_addition_noop_check.json
    hook_parity.json
    logit_lens_self_check.json
    patch_noop_check.json

  tables/
    corpus_manifest.csv
    feature_lineage_nodes.csv
    feature_lineage_node_scores.csv
    edge_eval_pairs.csv
    feature_lineage_edges.csv
    split_merge_candidates.csv
    causal_transfer_by_layer.csv
    causal_transfer_long.csv
    causal_transfer_dose_response.csv
    cross_model_feature_overlap.csv
    feature_lineage_evidence_matrix.csv
    evidence_matrix.csv
    label_stability_summary.csv
    feature_lineage_counterexamples.csv
    counterexamples.csv
    failure_specimens.jsonl
    failure_specimens.md
    confusable_control_ladder.csv
    plot_reading_guide.csv

    figure_sources/
      dashboard_evidence.csv
      dashboard_evidence.csv
      target_vs_control_source.csv
      target_vs_control_aggregate.csv
      dose_response_source.csv
      layer_sweep_heatmap_source.csv
      paired_examples_source.csv
      node_auc_by_depth_source.csv
      feature_lineage_node_scores_source.csv
      cross_layer_feature_graph_source.csv
      lineage_similarity_matrix_source.csv
      confusable_control_ladder_source.csv
      feature_split_merge_atlas_source.csv
      label_stability_source.csv
      cross_model_feature_overlap_source.csv
      causal_transfer_by_layer_source.csv
      counterexamples_source.csv

  plots/
    plot_manifest.json
    plot_manifest.csv
    plot_reading_guide.csv
    overview_dashboard.png
    feature_lineage_dashboard.png
    target_vs_control.png
    dose_response.png
    layer_sweep_heatmap.png
    paired_examples.png
    node_auc_by_depth.png
    cross_layer_feature_graph.png
    lineage_similarity_matrix.png
    confusable_control_ladder.png
    feature_split_merge_atlas.png
    label_stability_ladder.png
    cross_model_feature_overlap.png
    causal_transfer_by_layer.png

  state/
    cross_layer_dictionary.pt
    cross_layer_dictionary_metadata.json
    lineage_graph.json
    domain_markers.json
```

## Reading order

Start with `method_card.md`. It says whether the run used supervised prototype directions, confirms that no SAE/crosscoder/external-model comparison was run, and gives the domain verdicts.

Then read:

1. `diagnostics/warning_summary.csv`: did any data-quality or control warnings fire?
2. `plots/plot_manifest.json`: what is each figure's source table, metric, control, and claim boundary?
3. `diagnostics/tokenization_gate.csv`: did marker and contrast tokens pass?
4. `diagnostics/split_balance.csv`: did every domain have train/eval support?
5. `tables/feature_lineage_node_scores.csv`: what do the raw per-row projections look like?
6. `tables/feature_lineage_nodes.csv`: are directions held-out decodable above controls?
7. `tables/feature_lineage_edges.csv`: do adjacent-depth same-label edges recur above random and confusable controls?
8. `tables/failure_specimens.md`: what shrinks or kills the claim?
9. `operationalization_audit.md`: what language is allowed?
10. `plots/overview_dashboard.png`: the cockpit, not the verdict machine.

## How to read the figures

Read every plot with three questions in mind:

1. **What is the target measurement?** Same-label edge, held-out node AUC, marker-transfer gap, or split/merge entropy?
2. **What control is beside it?** Random directions, confusable domains, off-label targets, boundary-depth caveats, or raw row specimens?
3. **What claim would this plot still not support?** Most Lab 30 plots can support a candidate handle. None can support feature identity.

The manifest is part of the figure. Open `plots/plot_manifest.json` before exporting or screenshotting a plot. If a figure is not traceable to `tables/figure_sources/*.csv`, treat it as decorative until fixed.

## Plot catalog

| Figure | Source artifact | Question answered | Interpretation notes |
|---|---|---|---|
| `overview_dashboard.png` | `tables/figure_sources/dashboard_evidence.csv` | Which domains survive the whole audit? | Read as a cockpit. Follow up in the tables before writing claims. |
| `feature_lineage_dashboard.png` | `tables/figure_sources/dashboard_evidence.csv` | Legacy dashboard filename for the same evidence. | Kept for continuity with earlier handout text. |
| `target_vs_control.png` | `tables/figure_sources/target_vs_control_source.csv` | Do same-label edges beat confusable and random controls? | The key specificity plot. If controls are close, the claim narrows. |
| `dose_response.png` | `tables/figure_sources/dose_response_source.csv` | Does marker-token transfer depend on scale? | A token-margin probe only. It cannot rescue weak lineage evidence. |
| `layer_sweep_heatmap.png` | `tables/figure_sources/layer_sweep_heatmap_source.csv` | Where across depth is held-out node AUC lift above random? | Boundary rows are diagnostic, not claimable. |
| `paired_examples.png` | `tables/figure_sources/paired_examples_source.csv` | Do selected source-depth and target-depth scores move together on raw eval examples? | Raw points show whether a candidate edge is broad or specimen-carried. |
| `node_auc_by_depth.png` | `tables/figure_sources/node_auc_by_depth_source.csv` | Are node directions label-valid on held-out rows? | AUC is necessary, not sufficient. |
| `cross_layer_feature_graph.png` | `tables/figure_sources/cross_layer_feature_graph_source.csv` | Do same-label scores trace across adjacent depths? | A high curve is a candidate handle, not identity. |
| `lineage_similarity_matrix.png` | `tables/figure_sources/lineage_similarity_matrix_source.csv` | Are off-label or confusable domains strong? | Bright off-diagonal cells are often the most important evidence. |
| `confusable_control_ladder.png` | `tables/figure_sources/confusable_control_ladder_source.csv` | How much stronger is same-label recurrence than confusable/random controls? | Means can hide depth-specific failures. Read paired rows. |
| `feature_split_merge_atlas.png` | `tables/figure_sources/feature_split_merge_atlas_source.csv` | Where might split, merge, or label-change hypotheses live? | Hypothesis generation only. |
| `label_stability_ladder.png` | `tables/figure_sources/label_stability_source.csv` | How often is the strongest outgoing edge same-label? | Label survival is not semantic identity. |
| `cross_model_feature_overlap.png` | `tables/figure_sources/cross_model_feature_overlap_source.csv` | What does the future cross-model schema look like on same-model data? | This is explicitly not external cross-model evidence. |
| `causal_transfer_by_layer.png` | `tables/figure_sources/causal_transfer_by_layer_source.csv` | Where does default-dose marker transfer beat random? | Marker-logit side probe only. |
| `failure_specimens.md` | `tables/failure_specimens.jsonl` | Which specimens narrow the claim? | Read before positive ledger language. |

## Expected Tier A smoke behavior versus Tier B science behavior

Tier A should produce all tables, manifests, warning artifacts, and plot files without crashing. Because it uses a tiny model and a small balanced slice, it may show strange, unstable, or negative patterns. That is fine. Tier A is the plumbing and artifact-contract check.

Tier B is the evidence path. A Tier B claim needs frozen data, enough train/eval support per domain, passing self-checks, node AUC above random, same-label edges above random and confusable controls, and failure specimens that do not overturn the posture.

## Honest negative results

An honest negative result might look like any of these:

| Pattern | Interpretation |
|---|---|
| High eval AUC but weak same-label edge lift | The direction is label-valid at individual depths but does not recur under the edge instrument. |
| High same-label score but confusable edge is close | The domain pair may share vocabulary, surface form, or group structure. |
| Random-direction control matches same-label edge | The edge score may reflect generic residual geometry rather than label-specific recurrence. |
| Marker transfer works while node/edge gates fail | The direction can move marker logits without supporting cross-depth lineage. |
| Boundary depths dominate | Embedding or final-readout artifacts may be driving the pattern. |
| Cross-model placeholder looks high | Still same-model only. Do not write external cross-model language. |

A negative result is not a lab failure. It is the experiment refusing to let the favorite story ride a tiny parade float through the ledger.

## What this lab can claim

It can claim that a supervised prototype direction for a domain was decodable and recurred across adjacent depths under named controls.

It can claim that train-fit prototype directions survived or failed held-out evaluation rows.

It can claim that a marker-token activation addition moved marker-vs-contrast logits more than controls, if the table says so.

It can claim that split/merge candidates deserve follow-up.

## What this lab cannot claim

It cannot claim feature identity across layers.

It cannot claim that the prototype direction is an SAE feature.

It cannot claim a complete circuit.

It cannot claim an external cross-model result from the default run.

It cannot claim semantic steering from marker-token movement.

It cannot claim a split or merge without a follow-up intervention.

## Common failure modes

| Symptom | Likely cause | What to inspect |
|---|---|---|
| Many rows dropped | marker/contrast token not single-token for the runtime tokenizer | `diagnostics/tokenization_gate.csv` |
| A domain has no eval AUC | split imbalance after caps | `diagnostics/split_balance.csv` |
| Same-label and confusable scores are close | domain pair is too confusable or direction is vocabulary-like | `tables/feature_lineage_edges.csv`, `plots/paired_examples.png` |
| Top contexts come from wrong domains | prototype direction is broad or polysemantic | `tables/feature_lineage_node_scores.csv`, `tables/feature_lineage_nodes.csv` |
| All depths look strong at depth 0 only | embedding/token artifact | `plots/layer_sweep_heatmap.png`, `tables/feature_lineage_edges.csv` |
| Transfer probe dominates the story | marker token moved, not necessarily behavior | `tables/causal_transfer_by_layer.csv`, `plots/dose_response.png` |
| Plot looks positive but evidence row says no | one gate failed | `tables/evidence_matrix.csv`, `diagnostics/warning_summary.csv` |

## Writeup questions

1. Which domain had the strongest held-out node AUC?
2. Which domain had the strongest held-out same-label lineage edge?
3. Which confusable domain was the hardest control?
4. Did the train-fit direction and same-label edge survive held-out controls?
5. Did top contexts support the domain label, or did they reveal vocabulary leakage?
6. Did marker-token transfer agree with the lineage table?
7. Was marker transfer dose-dependent, random-like, or absent?
8. Which counterexample most narrows the claim?
9. What is the smallest sentence you can defend from `tables/evidence_matrix.csv`?
10. What would a real external cross-model extension need to add?
11. Write one allowed claim and one forbidden overclaim.

## Ledger templates

Positive:

```text
[L30-C1] DECODE+ATTR | For domain <F>, supervised prototype directions recurred across adjacent depths <d0→d1> on <model>: eval lineage lift <x>, confusable gap <y>, and held-out AUC <z>. This is a prototype-direction lineage claim, not feature identity.
Artifact: runs/<run>/tables/evidence_matrix.csv | Falsifier: a held-out corpus where the same edge loses to random or confusable-domain controls.
```

Negative/refinement:

```text
[L30-C2] DECODE,AUDIT | Domain <F> did not earn a positive lineage claim because <failed_gate>. The supported next step is <narrower scope>.
Artifact: runs/<run>/tables/failure_specimens.md | Falsifier: a rerun on a balanced held-out corpus where the failed gate passes without changing thresholds.
```

Forbidden:

```text
This is the same concept everywhere in the model.
This prototype direction is an SAE feature.
The same-model placeholder is a cross-model result.
The marker-token probe proves semantic steering.
```

## Suggested extensions

Replace prototype directions with SAE decoder vectors, keeping the same edge schema and controls.

Add independent per-layer SAEs and compare dictionary artifacts.

Run the same corpus across Pythia or OLMo checkpoints and connect the result to Lab 29.

Add a true second-model comparison and fill `cross_model_feature_overlap.csv` with external model IDs, tokenizer notes, matching rules, and matched random controls.

Use Lab 31 to ask whether automatic labels can predict the held-out top contexts in `feature_lineage_node_scores.csv`.
