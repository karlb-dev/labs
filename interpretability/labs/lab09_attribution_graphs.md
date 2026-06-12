# Lab 9: Attribution Graphs and Circuit Tracing

**Evidence level targeted:** `ATTR` for the attribution graph, upgraded to `CAUSAL` only for feature interventions that succeed on the real model with matched controls.

**Prerequisites:** Lab 6's `circuit_card.md`, Lab 8's understanding of transcoders, and the evidence ladder from every earlier lab. Base model, no chat template.

## The question

Can you turn a model computation into a graph of features, and how much should you trust a graph computed on a **replacement model** rather than the model itself?

Lab 6 built a circuit the slow way: screen heads, ablate them, prune them, and earn faithfulness, completeness, and minimality. Lab 9 asks for the same kind of mechanism-shaped claim, but at feature level and with automated attribution. The bargain is deliciously dangerous: you get a graph quickly, but only after freezing attention, replacing MLPs with transcoders, pruning thousands of possible edges, and hiding unreconstructed computation inside error nodes. This lab is about the graph and the receipt stapled to its back.

## Backend note: inspectable miniature, same evidence contract

The course outline names the frontier-style path: `circuit-tracer` with Gemma-family transcoders, shareable graph visualizations, and two-hop examples such as Dallas -> Texas -> Austin. This lab file implements the **inspectable miniature** path instead: `gpt2` plus the public Dunefsky full-stack MLP transcoders. That choice keeps the entire replacement-model and edge-attribution machinery visible in one Python file.

The tradeoff is scope. GPT-2 small does not reliably support the canonical two-hop demonstration, so the main behavior is one-hop factual recall:

```text
The capital of France is   ->  " Paris"   vs " Berlin"
The capital of Germany is  ->  " Berlin"  donor for counterfactual substitution
```

That is not a downgrade in epistemology. The lab still teaches the important method: build a feature graph, audit the replacement model, prune it, propose a supernode, intervene on the real model, run a random matched control, and compare the graph's blind spots against Lab 6.

A course build can swap the backend later. The artifact contract below should survive that swap.

## The core idea

### 1. Build a local replacement model

Run the real model once on the prompt. Capture:

- token embeddings and positions;
- every attention pattern;
- every LayerNorm input, so the data-dependent denominator can be frozen;
- every real MLP output;
- the final logits.

Then replay the forward pass with a local replacement model:

```text
attention pattern = frozen from the real pass
LayerNorm scale   = frozen from the real pass
MLP_k(x)          = transcoder_k(x) + error_node_k
```

The error node is not a footnote. It is the vector difference between the real MLP output and the transcoder reconstruction at that layer and position. With those error nodes included, the replacement model should reproduce the real logits up to numerical tolerance. That exactness does **not** mean the transcoders are perfect. It means the missing computation has been explicitly placed in nodes named `error`.

The first self-check is therefore:

```text
diagnostics/replacement_exactness.json
```

No exactness, no graph.

### 2. Make direct-attribution edges

Once attention patterns and LayerNorm denominators are frozen, the replacement network is linear in its source terms: embeddings, feature writes, error nodes, and transcoder output biases.

For a target scalar `t`, such as a feature pre-activation or the final logit difference,

```text
edge(feature s -> target t) = activation_s * (decoder_s dot grad_at_write_site)
```

The lab computes those edges with one backward pass per target node. The required accounting check is:

```text
bias path + embedding edges + feature edges + error edges + transcoder bias edges = target value
```

That check is saved in:

```text
diagnostics/edge_reconstruction_check.json
```

This is the little iron gate before the pretty graph garden.

### 3. Prune backward from the logit node

The full graph is too large to read. The lab starts at the metric node:

```text
logit(" Paris") - logit(" Berlin")
```

It keeps the strongest incoming feature edges, then recursively expands backward from selected feature nodes. The node budget is the compute and readability knob. The code writes both the pruned graph and a direct-logit budget curve so you can see how much feature-edge mass your chosen budget keeps.

### 4. Treat the graph as a hypothesis

A graph on the replacement model is not yet a claim about the real model. The graph proposes a mechanism such as:

```text
France-token features -> country/France supernode -> say-Paris features -> Paris logit
```

The lab then tests that hypothesis on the **real GPT-2 forward pass**:

- suppress the subject supernode by setting selected feature activations to zero;
- substitute counterfactual-country features from the Germany prompt;
- suppress a random matched set of active subject-site features as the control.

Only this intervention table can support a causal claim.

## What counts as success?

A strong run has this shape:

- the replacement exactness check passes;
- the edge reconstruction check passes;
- the graph's feature nodes cover a meaningful share of direct feature-edge mass;
- error-node share is measured and not swept into the sofa;
- subject-supernode suppression lowers the target-vs-distractor logit difference more than the random matched control;
- substitution increases probability on the counterfactual capital;
- at least some subject-site features recur across paraphrases.

A failed intervention is still a good result if diagnosed correctly. A beautiful graph whose supernode intervention behaves like a random perturbation is not a mechanism claim. It is a graph-shaped hypothesis that met the real model and lost the duel.

## The Lab 6 confrontation

The lab also runs the same attribution machinery on Lab 6's induction-style prompt:

```text
red blue green red blue green red blue -> " green" vs " red"
```

This is not because feature graphs are expected to be best at induction. It is because they are expected to reveal a blind spot. Induction is attention routing. This replacement model freezes attention into the wiring, so the routing computation cannot appear as learned feature structure in the same way the factual-recall MLP computation can.

The important comparison is not "which plot is prettier?" It is:

- Lab 6 forces you to validate heads by causal ablation, but it treats many MLP details as support structure.
- Lab 9 names MLP features and edges, but it freezes QK attention decisions and relies on dictionary fidelity.

Each microscope has a blind region. The course wants you to know the silhouette of both.

## Run commands

```bash
# Inspectable miniature path. The registry may already set these defaults.
python interp_bench.py --lab lab9 --tier a --model gpt2 --dtype float32 --attn-implementation eager
python interp_bench.py --lab lab9 --tier b --model gpt2 --dtype float32 --attn-implementation eager --max-examples 6
```

Some course builds expose:

```bash
--graph-nodes 32
```

The revised lab code also runs when that flag is absent. It falls back to the tier budget recorded in `diagnostics/graph_build_manifest.json`.

The 12 GPT-2 transcoders download once from Hugging Face. Keep this lab in float32. The replacement exactness check is deliberately strict, and low precision turns the audit into pudding.

## Artifact map

```text
runs/lab09_attribution_graphs-.../
  graph_card.md
  run_summary.md
  metrics.json
  results.csv

  diagnostics/
    graph_build_manifest.json
    tokenization_report.csv
    replacement_exactness.json
    edge_reconstruction_check.json
    feature_edit_noop_check.json
    feature_intervention_manifest.json
    transcoder_stack_report.json

  graphs/
    pruned_graph.json
    supernode_map.json
    induction_graph.json

  tables/
    baseline_gate.csv
    transcoder_reconstruction_by_layer.csv
    graph_nodes.csv
    graph_edges.csv
    logit_edge_sources.csv
    node_budget_curve.csv
    supernode_features.csv
    intervention_results.csv
    paraphrase_robustness.csv
    influence_ledger.csv

  plots/
    attribution_graph.png
    influence_composition.png
    edge_mass_shares.png
    intervention_effects.png
    paraphrase_recurrence.png
```

## First artifact-reading path

Start with `graph_card.md`. It is the Lab 9 counterpart of Lab 6's circuit card and contains the mechanism hypothesis, intervention verdict, error-node share, coverage, and non-claims.

Then read `diagnostics/replacement_exactness.json` and `diagnostics/edge_reconstruction_check.json`. These are not optional plumbing files. They decide whether the graph is an instrument or decorative spaghetti.

Next read `tables/logit_edge_sources.csv` before `plots/attribution_graph.png`. The plot is pruned for readability. The table shows the largest raw sources into the logit node before display pruning.

Then read `tables/intervention_results.csv`, `tables/supernode_features.csv`, and `diagnostics/feature_intervention_manifest.json`. These tell you whether the graph's proposed subject supernode survived contact with the real model, and whether duplicate feature assignments in substitution were collapsed correctly.

Finally compare `plots/influence_composition.png` and `plots/edge_mass_shares.png`. The signed ledger says who paid for the logit difference. The absolute-share plot says how much of the graph's direct edge mass lives in features, embeddings, or error nodes.

## How to read the main plots

`attribution_graph.png` is a display graph, not a complete graph. Dots are features, squares are embeddings, triangles are error nodes, and the star is the logit-difference node. Blue edges push the target direction, red edges push against it. Edge width tracks absolute direct attribution.

`influence_composition.png` is the signed ledger. The bars should sum to the metric. This plot is best for questions like, "Did feature writes push Paris, or did token embeddings and biases do the work?"

`edge_mass_shares.png` is the visibility audit. A high error-node share means the graph is leaning on unreconstructed computation. That does not automatically kill the explanation, but it must lower the swagger.

`intervention_effects.png` is the causal test. The graph wins only if suppression and substitution move the behavior in a way the random matched control does not.

`paraphrase_recurrence.png` is the cheap robustness screen. Recurrent subject-site features are better mechanism candidates than one-template fireworks.

## Writeup questions

1. Did replacement exactness and edge reconstruction both pass? Explain what each check means in one sentence, and what would be invalid if it failed.
2. Which source category carried the largest **signed** contribution to the factual-recall logit diff: embeddings, features, errors, bias path, or transcoder bias?
3. Which source category carried the largest **absolute edge mass**? If error nodes are large, what exactly has the graph failed to explain?
4. Did subject-supernode suppression beat the random matched control? State the suppression drop, random-control drop, and specificity gap.
5. Did counterfactual substitution increase probability on the counterfactual capital? Was that enough to call the mechanism causal?
6. Which subject-site features recur across paraphrases? Which look like template artifacts?
7. Compare your Lab 6 circuit card and this graph card. Which method made stronger assumptions? Which gave stronger controls? Which would you trust for an attention-routing claim, and which for an MLP-feature claim?

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| Lab crashes before graph construction | `diagnostics/tokenization_report.csv`, especially subject and answer single-token checks |
| no attention patterns returned | run with eager attention; frozen patterns are required |
| replacement exactness fails | dtype, GPT-2 architecture mismatch, wrong transcoder convention, or missing error-node term |
| edge reconstruction fails | a frozen quantity is live, an injection site is wrong, or feature gates were not detached |
| all top sources are error nodes | transcoder dictionary does not describe this prompt's MLP computation well |
| graph looks plausible but intervention fails | the supernode grouping is wrong, too small, or not specific |
| random control moves as much as suppression | the intervention is generic perturbation, not mechanism-specific evidence |
| paraphrase recurrence is zero | features are template-bound, or the subject position changed under tokenization |

## Extensions

**Manageable:** rerun the primary graph on Italy or Japan as the factual-recall target. Keep the same artifact contract and compare whether the same layers and feature families appear.

**Medium:** add a second random matched control and report a small bootstrap interval for the control effect. This turns "random was smaller once" into a real control estimate.

**Ambitious:** add one QK-attribution vignette for the induction prompt. Unfreeze the attention pattern for one induction head, differentiate the softmax score or attention weight, and ask which source-position features would sharpen attention to the copied token. That stitches one tendon back onto the frozen-attention skeleton.

## Interpretation and ethics: idealization

The graph describes an idealized replacement model: frozen attention, frozen norm denominators, transcoder features, and error nodes. Idealization is not cheating. Science often explains by replacing the world with something simpler. The question is whether the simplification preserves the causal handles that matter.

**Writing prompt:** Defend and attack this claim:

> An idealized model that supports successful interventions is explanation enough.

Use your own artifacts. The substitution result and random control argue for the claim. Error-node share, node-budget coverage, and the induction comparison argue against overconfidence. Two paragraphs, then a verdict you would sign.

## What goes in the ledger

Write 3 to 4 claims.

The graph-structure claim is `ATTR`. It must name the model, prompt, metric, feature share, error-node share, and kept-coverage number.

The intervention claim is `CAUSAL` if the real-model intervention beats the matched random control. If it does not, write a negative causal claim: the tested supernode was not validated.

The paraphrase claim is `ATTR`. It must distinguish recurring features from one-template artifacts.

The Lab 6 comparison claim can be `OBS` or `ATTR`, depending on what your artifacts show. Its job is to scope what Lab 6 and Lab 9 are each allowed to say.

Do not write "the graph is causal." The graph proposes. The intervention disposes.

## Reading

- Ameisen et al., "Circuit Tracing: Revealing Computational Graphs in Language Models".
- Lindsey et al., "On the Biology of a Large Language Model".
- Dunefsky et al., "Transcoders Find Interpretable LLM Feature Circuits".
- Marks et al., "Sparse Feature Circuits".
- Potochnik, *Idealization and the Aims of Science*, one chapter or equivalent.
