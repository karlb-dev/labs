# Lab 30: Cross-Layer and Cross-Model Feature Geometry

Time estimate: 60-90 minutes for the default prototype-direction run.  
Compute tier: Tier A uses `gpt2` on a small frozen corpus; Tier B can run the same workflow on the course base model or external dictionaries.  
Dependencies: Labs 8, 19, and 29 concepts.  
Minimum passing artifacts: `tables/feature_lineage_nodes.csv`, `tables/feature_lineage_edges.csv`, `tables/label_stability_summary.csv`, `tables/cross_model_feature_overlap.csv`, `state/cross_layer_dictionary.pt`, `state/lineage_graph.json`, and `plots/feature_lineage_dashboard.png`.  
Main plot: `plots/feature_lineage_dashboard.png`.  
Main table: `tables/label_stability_summary.csv`.  
Evidence rung: `DECODE + ATTR`, with a narrow activation-addition transfer probe.  
Forbidden claim: "This is the same concept everywhere in the model."  
One-sentence allowed claim: "A supervised feature direction for domain F recurred across these depths above random and confusable controls on this frozen corpus."  
Human-label requirement: none for the default domain-labeled corpus.

## Why This Lab Exists

Lab 8 treated features at one site. Lab 19 compared model-diff features. Lab 29
made checkpoint movement explicit. Lab 30 asks whether a feature-like direction
can be tracked across depth without pretending that similarity is identity.

The first pass is intentionally conservative. It does not train a sparse
autoencoder or crosscoder. Instead it builds supervised prototype directions:

```text
direction(domain, layer) = mean(domain activations) - mean(other activations)
```

That gives students a cheap, inspectable feature-lineage workflow before they
run expensive dictionary training.

## Data

Default rows live in `data/feature_lineage_corpus.csv`.

Required columns:

```text
row_id, family, domain, source_lab, text, group_id, split, labels_json
```

The corpus includes paired confusable groups:

- code / cooking;
- finance / sports;
- law / medicine;
- weather / emotion.

Each row also has marker and contrast tokens for the activation-addition
transfer probe. These are runtime-checked to be single tokens.

## What The Lab Builds

### Nodes

`tables/feature_lineage_nodes.csv` contains one node per `(domain, depth)`.

Each node records:

- direction norm;
- train and eval AUC for the domain label;
- random-direction AUC;
- top activating contexts;
- the feature kind: `supervised_prototype_direction`.

### Edges

`tables/feature_lineage_edges.csv` compares every source domain at one depth to
every target domain at the next depth.

The edge score combines:

- direction cosine;
- activation correlation across the corpus;
- top-context Jaccard;
- endpoint label AUC.

Random-direction controls are stored on the same row. Same-label edges become
claim candidates only if they beat the random score by a margin.

### Split And Merge Screens

`tables/split_merge_candidates.csv` looks for cases where:

- one source domain has several strong next-depth targets;
- several source domains point into the same target;
- the strongest target label changes.

These are screens, not proof of splitting or merging.

### Cross-Model Placeholder

`tables/cross_model_feature_overlap.csv` is honest about the first pass: it
compares same-model cross-layer overlap with deterministic random controls. It
does not claim an external cross-model result. A later Pythia, OLMo, SAE, or
crosscoder extension should reuse the same columns.

### Transfer Probe

`tables/causal_transfer_by_layer.csv` adds a domain direction to a neutral prompt
and measures:

```text
logit(marker_token) - logit(contrast_token)
```

This is a narrow marker-logit intervention. It is not semantic control of model
behavior.

## How To Run

```bash
cd interpretability
python interp_bench.py --lab lab30 --tier a
python interp_bench.py --lab lab30 --tier b --prompt-set full
```

For a fast table-only smoke:

```bash
python interp_bench.py --lab lab30 --tier a --no-plots
```

## Reading Order

1. `method_card.md`

   Confirms that the lab used supervised prototype directions, not SAE features.

2. `tables/feature_lineage_nodes.csv`

   Checks whether each domain direction is decodable at each depth.

3. `tables/feature_lineage_edges.csv`

   Checks whether same-label adjacent-depth edges beat random controls.

4. `tables/split_merge_candidates.csv`

   Looks for split, merge, and label-change candidates.

5. `tables/label_stability_summary.csv`

   Gives the domain-level verdict.

6. `tables/cross_model_feature_overlap.csv`

   Records the first-pass limitation on external cross-model claims.

7. `operationalization_audit.md`

   Lists cheap explanations before you write a feature-identity claim.

## Common Failure Modes

### High Cosine, Weak Held-Out AUC

The direction may be stable but not label-valid. Do not call it a feature
lineage.

### Strong Same-Label Edge, Strong Confusable Edge

The domain may be riding on shared vocabulary. Inspect `group_id` and top
contexts before claiming survival.

### Good Marker Transfer, Bad Lineage

The direction can move a marker token without tracking a stable feature across
depth. Keep the intervention claim separate.

### External Cross-Model Claim Sneaks In

The default run does not compare two pretrained model families. The
`cross_model_feature_overlap.csv` file is a schema placeholder plus same-model
control, not a cross-model result.

## Extension Ideas

- Replace prototype directions with SAE decoder vectors.
- Add independent per-layer SAEs and compare dictionary artifacts.
- Add Pythia checkpoints and connect this lab to Lab 29.
- Add cross-model OLMo/GPT-style comparisons through a crosscoder.
- Use Lab 31 to auto-label the top contexts, then validate those labels here.

## Claim Grammar

Allowed:

```text
DECODE + ATTR: Feature family F has a recurring supervised direction across
layers L1-L3, with label AUC, activation correlation, top-context overlap, and
random-control lift recorded in the artifact tables.
```

Forbidden:

```text
This is the same concept everywhere in the model.
```

Also forbidden:

```text
This prototype direction is an SAE feature.
This same-model control is a cross-model result.
The marker-token intervention proves semantic steering.
```

## Deliverable

Write a short lineage memo:

- Which domain had the strongest recurring direction?
- Which confusable domain was the hardest control?
- Did label stability survive held-out contexts?
- Did the activation-addition probe agree with the lineage table?
- What is the smallest feature-lineage claim you can make?
