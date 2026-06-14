# Lab 30: Cross-Layer Feature Lineage Without Feature-Identity Overclaiming

**One-sentence thesis:** A feature-like direction earns a lineage claim only when its label, activations, top contexts, and controls recur together across depth.

**Time estimate:** Tier A smoke in minutes on CPU; Tier B science on the course base model in a Colab/A100-style runtime.

**Compute tier:** Tier A uses `gpt2` on a balanced small slice; Tier B uses the course base model on the frozen corpus.

**Dependencies:** Labs 8, 19, and 29. Lab 31 can reuse the top-context tables for automated interpretability audits.

**Minimum passing artifacts:** `method_card.md`, `operationalization_audit.md`, `metrics.json`, `results.csv`, `results.jsonl`, `tables/feature_lineage_nodes.csv`, `tables/feature_lineage_edges.csv`, `tables/evidence_matrix.csv`, `tables/label_stability_summary.csv`, `tables/cross_model_feature_overlap.csv`, `state/cross_layer_dictionary.pt`, `state/lineage_graph.json`, and `plots/feature_lineage_dashboard.png`.

**Main plot:** `plots/feature_lineage_dashboard.png`

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

The refactor includes a deterministic generator:

```text
data/make_feature_lineage_corpus.py
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
confusable-domain edge evidence
top train/eval context IDs
```

The node table is:

```text
tables/feature_lineage_nodes.csv
```

A direction that is stable but not held-out label-valid is not a lineage claim. It is a geometry artifact with a polite hat.

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

This is a narrow marker-logit intervention. It is not semantic steering.

The table is:

```text
tables/causal_transfer_by_layer.csv
```

## Depth discipline

Depth `0` is the embedding stream. The final depth is the final-norm input. Both are useful diagnostic rows, but formal lineage gates prefer interior depths when available.

The run records whether each node or edge is claimable in the depth grid.

## Controls and falsifiers

| Favorite story | Control or falsifier |
|---|---|
| The domain has a real recurring direction | Held-out AUC and confusable-domain AUC |
| Same-label edges track lineage | Random-direction edge score |
| The edge is label-specific | Best off-label edge and confusable-domain edge |
| The direction is not just vocabulary | Top activating contexts and confusable groups |
| A marker edit shows semantic control | It only moves marker-vs-contrast token logits; semantic claims require a later generated-text audit |
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
    hook_parity.json
    logit_lens_self_check.json
    patch_noop_check.json

  tables/
    feature_lineage_nodes.csv
    feature_lineage_edges.csv
    split_merge_candidates.csv
    causal_transfer_by_layer.csv
    cross_model_feature_overlap.csv
    feature_lineage_evidence_matrix.csv
    evidence_matrix.csv
    label_stability_summary.csv
    feature_lineage_counterexamples.csv
    counterexamples.csv
    confusable_control_ladder.csv
    plot_reading_guide.csv

  plots/
    plot_reading_guide.csv
    feature_lineage_dashboard.png
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

1. `diagnostics/data_manifest.json`: science-ready data or smoke-only fallback?
2. `diagnostics/tokenization_gate.csv`: did marker and contrast tokens pass?
3. `diagnostics/split_balance.csv`: did every domain have train/eval support?
4. `tables/feature_lineage_nodes.csv`: are directions held-out decodable above controls?
5. `tables/feature_lineage_edges.csv`: do adjacent-depth same-label edges recur?
6. `tables/feature_lineage_counterexamples.csv`: what shrinks or kills the claim?
7. `operationalization_audit.md`: what language is allowed?
8. `plots/feature_lineage_dashboard.png`: the cockpit, not the verdict machine.

## Plot guide

### `feature_lineage_dashboard.png`

Read first. It combines node decodability, edge control gaps, confusable gaps, and transfer probe behavior.

### `cross_layer_feature_graph.png`

Shows same-label lineage scores over depth. A high curve is a candidate handle, not a feature identity.

### `lineage_similarity_matrix.png`

Shows mean source-domain to target-domain scores. Off-diagonal strength is often the most important part of the plot.

### `feature_split_merge_atlas.png`

Screens for possible split/merge/label-change behavior. Treat as hypothesis generation.

### `label_stability_ladder.png`

Shows domain-level lift or survival after the selected-edge audit.

### `cross_model_feature_overlap.png`

Same-model overlap versus random. It is not external cross-model evidence.

### `causal_transfer_by_layer.png`

Marker-token activation-addition control gap by depth. It is a token probe, not semantic steering.

## Interpreting result patterns

| Pattern | Interpretation |
|---|---|
| High eval AUC, high same-label edge lift, confusable gap positive, held-out evaluation survives | Narrow positive lineage claim for this corpus and depth grid. |
| High cosine but weak eval AUC | Geometry recurs, but the label is not validated. |
| Same-label edge loses to confusable edge | Domain vocabulary or group confound may explain the lineage. |
| Node/edge evidence passes train-like screens but fails held-out controls | Discovery-split candidate only. |
| Marker transfer works but lineage fails | The direction can move marker logits without supporting cross-depth lineage. |
| Cross-model placeholder looks high | Still same-model only. Do not write external cross-model language. |

## What this lab can claim

It can claim that a supervised prototype direction for a domain was decodable and recurred across adjacent depths under named controls.

It can claim that train-fit prototype directions survived or failed held-out evaluation rows.

It can claim that a marker-token activation addition moved marker-vs-contrast logits more than controls, if the table says so.

## What this lab cannot claim

It cannot claim feature identity across layers.

It cannot claim that the prototype direction is an SAE feature.

It cannot claim a complete circuit.

It cannot claim an external cross-model result from the default run.

It cannot claim semantic steering from marker-token movement.

## Common failure modes

| Symptom | Likely cause | What to inspect |
|---|---|---|
| Many rows dropped | marker/contrast token not single-token for the runtime tokenizer | `diagnostics/tokenization_gate.csv` |
| A domain has no eval AUC | split imbalance after caps | `diagnostics/split_balance.csv` |
| Same-label and confusable scores are close | domain pair is too confusable or direction is vocabulary-like | `tables/feature_lineage_edges.csv` |
| Top contexts come from wrong domains | prototype direction is broad or polysemantic | `tables/feature_lineage_nodes.csv` |
| All depths look strong at depth 0 only | embedding/token artifact | `tables/feature_lineage_nodes.csv` |
| Transfer probe dominates the story | marker token moved, not necessarily behavior | `tables/causal_transfer_by_layer.csv` |

## Writeup questions

1. Which domain had the strongest held-out node AUC?
2. Which domain had the strongest held-out same-label lineage edge?
3. Which confusable domain was the hardest control?
4. Did the train-fit direction and same-label edge survive held-out controls?
5. Did top contexts support the domain label, or did they reveal vocabulary leakage?
6. Did marker-token transfer agree with the lineage table?
7. Which counterexample most narrows the claim?
8. What is the smallest sentence you can defend from `tables/evidence_matrix.csv`?
9. What would a real external cross-model extension need to add?
10. Write one allowed claim and one forbidden overclaim.

## Ledger templates

Positive:

```text
[L30-C1] DECODE+ATTR | For domain <F>, supervised prototype directions recurred across adjacent depths <d0→d1> on <model>: eval lineage lift <x>, confusable gap <y>, and held-out AUC <z>. This is a prototype-direction lineage claim, not feature identity.
Artifact: runs/<run>/tables/evidence_matrix.csv | Falsifier: a held-out corpus where the same edge loses to random or confusable-domain controls.
```

Negative/refinement:

```text
[L30-C2] DECODE,AUDIT | Domain <F> did not earn a positive lineage claim because <failed gate>. The supported next step is <narrower scope>.
Artifact: runs/<run>/tables/feature_lineage_counterexamples.csv | Falsifier: a rerun on a balanced held-out corpus where the failed gate passes without changing thresholds.
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

Add a true second-model comparison and fill `cross_model_feature_overlap.csv` with external model identifiers.

Feed the top-context columns from `tables/feature_lineage_nodes.csv` into Lab 31 auto-labeling, then validate those labels here.
