# Lab 6: Circuit Discovery and Validation, the Manual Way

**Evidence level targeted:** causal evidence at circuit scope.

**Prerequisites:** Labs 2, 3, and 5 (plus the residual-stream and self-check habits from Lab 1). Lab 2 gives direct-logit attribution (the cheap screen), Lab 3 gives attention motifs (previous-token and induction patterns), and Lab 5 gives the intervention habit (mean-ablation instead of zero-ablation). Keep the `circuit_card.md` from this lab: Lab 9 will place your hand-built circuit next to an automated attribution graph so you can compare the epistemic standards of the two methods.

## The question

Can you reduce a behavior to a small computational subgraph and say, with measurements attached, that the graph is **faithful**, **complete**, and **minimal**?

Those words are not decorative labels:

| Word | Operational test | Read it as |
|---|---|---|
| faithful | mean-ablate every non-circuit head | the circuit alone preserves the behavior |
| complete | mean-ablate the circuit heads | the circuit is necessary for the behavior |
| minimal | remove each kept head from the final circuit | every kept node earns its rent |

**Make the concept pop (F/C/M + held-out):** In `plots/circuit_scorecard.png` and `faithfulness_completeness_minimality.json` you will see two rows: discovery vs held-out. Faithfulness on held-out is often *higher* than on discovery (the circuit is not over-fit to the exact tokens). Completeness effect is usually larger on held-out. Minimality (worst marginal value in `tables/pruned_circuit.csv`) tells you whether the last head you kept was actually pulling its weight.

The lab is a little circuit courtroom. The cheap screens nominate suspects. Mean-ablation cross-examines them. The circuit card is the verdict, including all awkward caveats.

**Make the concept pop:** After your run, open `plots/screen_vs_causal.png`. The left panel shows cheap screen rank vs actual causal drop; many high-ranking suspects (by attribution or motif) have near-zero causal drop. The right panel does the same for absolute attribution. These off-diagonal points are the central lesson: screening is a hypothesis generator, not a circuit claim. Then open `plots/per_prompt_faithfulness.png` — the lowest bars are the specific prompts your final circuit explains least well. Do not hide them.

**Headline numbers note:** Faithfulness/completeness/minimality are computed on a small set of 8-token repeating prompts (17 discovery+heldout after expansion for robustness, minus any the tokenization gate drops on a given tokenizer). The per-prompt table, held-out generalization, and edge-interaction controls are what make the scoped circuit claim legible; the headline F/C/M ratios on this synthetic task should be read as qualitative + one-significant-figure evidence.

## The task

The behavior is induction completion on fixed-length 8-token repeating patterns:

```text
red blue green red blue green red blue ->  green
```

The distractor is the cycle restart:

```text
red blue green red blue green red blue ->  red
```

The metric is:

```text
logit(target) - logit(distractor)
```

Discovery prompts use one set of vocabulary families. Held-out prompts use fresh families such as metals, compass points, and beasts. Held-out evaluation asks whether the circuit is about the induction pattern or merely about these exact tokens.

The baseline gate matters. A prompt is used for discovery only if the unablated model already prefers the target over the distractor. You cannot trace a circuit for a behavior the model is not doing.

## Scope: heads-only routing graph

The course outline describes circuit discovery broadly as heads and MLPs. This executable lab deliberately narrows the claim: the validated circuit nodes are **attention heads only**.

MLP layers are still causally ranked and plotted as supporting infrastructure. They are not part of the faithfulness complement. That means the claim is not "this is the whole mechanism in the transformer." The claim is:

```text
For this prompt family and metric, these attention heads form a heads-only routing subgraph that preserves and disrupts the behavior under dataset-mean ablation.
```

That sentence is longer than "we found the circuit," but it has a spine.

## Why mean-ablation is the off switch

Zero-ablating hundreds of heads creates a model that never appears during normal inference. It can prove that your hooks are powerful without proving that the model uses the circuit you named.

This lab uses **dataset-mean ablation**. For each discovery prompt, it captures every head's out-projection input at every position. Because the prompt length is fixed at 8 tokens, it can replace a head's prompt-specific output with the mean output for that same layer, head, and position.

Mean-ablation removes prompt-specific computation while staying closer to the data manifold. The cost is that the circuit is relative to that off distribution. A different off switch can define a different circuit.

The run writes `diagnostics/ablation_manifest.json` so this choice is not trapped in a comment.

## Method spine

1. **Validate the task.** Confirm the prompt length and answer tokenization. Drop discovery prompts the model does not solve.
2. **Screen cheaply.** Rank heads by final-position attribution, induction motif score, and previous-token motif score. Rank MLP layers by direct-logit attribution as support candidates.
3. **Rank causally.** Mean-ablate each screened candidate alone and measure the drop in the logit-difference metric.
4. **Prune greedily.** Start from every screened head with positive causal drop. Repeatedly remove the head whose removal gives the highest remaining faithfulness. Stop before crossing the faithfulness floor, or stop if the candidate set never reaches it.
5. **Earn F/C/M.** Measure faithfulness, completeness, and minimality on discovery and held-out families.
6. **Try one edge.** Test ordered previous-token -> induction pairs by ablation interaction. The source head must be in an earlier layer than the target head. If no ordered pair passes the checks, the lab says no edge was earned. If the best pair is weak, report it as weak rather than inflating it into a crisp path claim.
7. **Write the circuit card.** The card is the deliverable. The plots and tables are evidence for it.

## The edge claim, carefully

The edge test asks whether a previous-token head's effect shrinks when an induction head is already ablated.

```text
interaction = effect(previous head alone) - effect(previous head | induction head already ablated)
```

A positive interaction supports the idea that part of the previous-token head's effect routes through the induction head. In the executable lab, a pair is reportable at 2% routed fraction and labeled **weak** until 5%. This keeps the Olmo 7B target honest: redundancy can make the strongest single edge small without making the interaction uninteresting.

It does **not** show whether the route is through keys, values, queries, residual writes, or another subpath. That stronger claim needs path patching.

The revised code also refuses impossible arrows: a later previous-token head cannot route through an earlier induction head in an ordinary forward computation. If the layer order is wrong, the pair is not eligible for the edge claim.

## Running it

```bash
python interp_bench.py --lab lab6 --tier a
python interp_bench.py --lab lab6 --tier b --prompt-set full
```

Tier A is a plumbing smoke test on `gpt2`. Tier B is the course run. The bench auto-enables eager attention because motif scores need attention patterns.

## First artifact-reading path

**Instrument health + baseline gate first (exactly as in Labs 1-5):**
1. `diagnostics/tokenization_and_baseline.csv` — the model must actually do the induction task on the discovery prompts before you claim a circuit for it.
2. `diagnostics/ablation_manifest.json` — the off distribution (dataset mean, fixed length 8) is part of the claim.

**Then the science (the manual workflow):**
3. `circuit_card.md` — the deliverable. Read this first; everything else is evidence for it.
4. `plots/screen_vs_causal.png` — the central "cheap screening is a hypothesis generator" plot (off-diagonal points are the lesson).
5. `plots/prune_trajectory.png` — how the circuit shrinks and where it stops relative to the faithfulness floor.
6. `plots/circuit_scorecard.png` and `faithfulness_completeness_minimality.json` — F/C/M on discovery vs held-out (the held-out row is the generalization test).
7. `plots/per_prompt_faithfulness.png` and `tables/per_prompt_faithfulness.csv` — the specific prompts your circuit explains least well (the anti-cherry-pick evidence).
8. `plots/circuit_graph.png` — the visual summary with motif labels and any claimed edge.
9. `tables/pruned_circuit.csv` — minimality: the worst marginal value tells you whether the last kept head was earning its place.
10. `plots/edge_interactions.png`, `tables/edge_claim.json`, and `tables/edge_interactions.csv` — the (carefully scoped) edge claim or the reason none was earned.

## How to read the main plots

### `screen_vs_causal.png`

The left panel plots cheap screen rank against causal drop. The right panel plots attribution magnitude against causal drop. Points above zero mattered under mean-ablation. Points below zero were attractive suspects that did not help the behavior.

A good writeup names at least one disagreement. For example: "Head X had a high induction motif score but little causal drop, so the motif was behavior-adjacent but not necessary under this metric."

### `prune_trajectory.png`

Read right to left. The rightmost point is the full positive-causal screened set. Each step removes one head. The red line is the faithfulness floor.

If the first point is below the floor, the screen did not collect enough of the behavior. That is not a crash. It is an experimental result with a caveat attached.

### `circuit_scorecard.png`

Faithfulness is preservation. Higher is better.

Completeness effect is `1 - completeness_ratio`. Higher means circuit ablation destroyed more of the behavior. A circuit can be faithful but not complete if redundant paths exist. It can be complete but not minimal if extra heads hitchhiked into the final set.

### `per_prompt_faithfulness.png`

This is the anti-cherry-pick plot. The lowest bars become the failure cases in the circuit card. Do not hide them. Ask what they share.

## What the revised code improves

The updated lab makes several evidence-quality changes:

- It writes a combined tokenization and baseline report before any circuit claims.
- It adds the component decomposition self-check, so attention and MLP contribution bookkeeping is verified before screening.
- It records an ablation manifest that defines the off distribution.
- It broadens and documents the screen budgets instead of relying on a tiny fixed candidate set.
- It separates cheap screen rank, attribution score, motif scores, and causal rank in `candidate_components.csv`.
- It prevents overclaiming when the pruned circuit fails the faithfulness floor.
- It labels weak versus strong edge interactions instead of treating every positive interaction as equally crisp.
- It always writes an edge diagnostic, even when no edge is claimed.
- It requires edge direction to respect layer order.
- It adds per-prompt faithfulness artifacts so failure cases are visible rather than prose-only.

## Writeup questions

1. What is sufficient? Cite faithfulness and say whether the final circuit passed the floor.
2. What is necessary? Cite completeness ratio and explain whether circuit ablation destroyed the behavior or merely dented it.
3. What is minimal? Cite the worst marginal value in `tables/pruned_circuit.csv`. Did every kept node have a positive marginal value?
4. Where did cheap screening and causal ranking disagree most? Was the misleading signal attribution, induction motif, or previous-token motif?
5. What do the two weakest per-prompt faithfulness cases share?
6. Did the held-out families preserve the result? If held-out prompts were filtered because the base model missed them, say how many ratio-defined prompts remain. If faithfulness is above 1.0, explain why that is over-recovery, not magic.
7. If an edge was claimed, was it weak or strong? What exactly does the interaction fraction license you to say? If no edge was claimed, which requirement failed?
8. Map your card onto Machamer, Darden, and Craver's entities-and-activities schema. Mark every activity you named but did not directly test.

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| most discovery prompts dropped | `diagnostics/tokenization_and_baseline.csv`; the model may not do the task |
| `head_means seq mismatch` | prompt length changed; mean-ablation requires the fixed-length contract |
| starting faithfulness below 0.70 | the screen missed important heads or MLP support is doing more than expected |
| pruning stops immediately | the candidate set is fragile, redundant, or already below the floor |
| completeness ratio stays high | ablating the circuit leaves another path for the behavior |
| minimality worst marginal is negative | at least one kept head hurts faithfulness under this pruning rule |
| no edge claimed | check `tables/edge_interactions.csv`; motif alone is not an edge |

## What goes in the ledger

The ledger should contain two or three `CAUSAL` claims. Each claim must name:

```text
model, prompt population, metric, intervention, circuit scope, artifact, falsifier
```

A good claim is scoped enough to survive contact with a skeptical reader:

```text
For 8-token induction prompts on this model, a heads-only routing circuit of [nodes]
preserves X of the target-vs-distractor logit gap when every non-circuit head is
replaced by its discovery-set mean. This does not claim natural-text induction,
MLP sufficiency, or invariance under zero-ablation.
```

A circuit card without scope is a treasure map with no scale bar: exciting, foldable, and treacherous in the field.
