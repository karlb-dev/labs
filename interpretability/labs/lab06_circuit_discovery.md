# Lab 6: Circuit Discovery and Validation, the Manual Way

**Evidence level targeted:** causality, composed into a subgraph claim.
**Prerequisites:** Labs 2, 3, 5 — this lab is their composition, and it reuses
your Lab 3 motif map directly. **Keep the circuit card you produce: Lab 9
will hold it next to an automated attribution graph.**

## The question

Can you reduce a behavior to a small computational subgraph — built from
heads by hand — that is **faithful** (the circuit alone preserves the
behavior), **complete** (removing the circuit destroys it), and **minimal**
(every kept node earns its place)? And what does each of those words cost?

## The task

Induction completion on fixed-length 8-token repeating patterns
(`red blue green red blue green red blue` → ` green`, distractor ` red` —
the cycle restart). Chosen because: you already hold candidate heads from
Lab 3; the model does it reliably (baseline gate verifies per prompt); and
**held-out vocabulary families** (metals, compass points, beasts…) let you
test whether your circuit is about *induction* or about *these tokens*.

## Method spine

1. **Screen cheaply**: rank heads by final-position attribution (Lab 2's
   frozen-norm convention, per-head as in Lab 3) and by motif scores.
   Screening is allowed to lie — that's why step 2 exists.
2. **Rank causally**: mean-ablate each candidate alone (all positions) and
   measure the metric drop over the discovery set.
3. **Prune greedily**: starting from the causally-confirmed set, repeatedly
   remove the head whose removal costs the least faithfulness, stopping at
   the floor (0.7). The trajectory plot shows what every node was worth.
4. **Earn the three numbers**: faithfulness (complement of the circuit
   mean-ablated), completeness (circuit mean-ablated), minimality (marginal
   value per kept node) — on discovery AND held-out families.
5. **Earn one edge**: ablation interaction. If the previous-token head's
   effect vanishes when the induction head is already ablated, the effect
   routed through it. (Path patching would localize the edge to keys vs
   values; that's named future work, not this lab.)

### Why mean-ablation

Zero-ablating hundreds of heads tests a model that never exists — activations
far off-distribution prove nothing about the real computation. Replacing each
head's output with its **dataset mean** (well-defined because every prompt is
exactly 8 tokens) removes prompt-specific computation while staying near the
data manifold. Write this in the card's scope section: *a different "off"
defines a different circuit*. That sentence is the most transferable thing in
this lab.

### Scope decision you must understand

The circuit's node set is **attention heads only**. MLP layers are causally
ranked and reported as supporting infrastructure, but the faithfulness
complement never ablates them. A subgraph claim should say what it is a
subgraph *of* — ours is the routing graph. (Lab 5 told you where the MLP
"recall" work happens; the card links the two.)

## Running it

```bash
python interp_bench.py --lab lab6 --tier a               # gpt2, 6 discovery prompts
python interp_bench.py --lab lab6 --tier b --prompt-set full
```

The bench auto-sets eager attention (motif screen needs patterns).

## First artifact-reading path

1. `circuit_card.md` — the deliverable. Everything else is its evidence.
2. `plots/circuit_graph.png` — the subgraph, motif-labeled, with the one edge.
3. `plots/prune_trajectory.png` — faithfulness vs circuit size; read it
   right-to-left and you watch the circuit assemble.
4. `plots/screen_vs_causal.png` — every point off the trend is a place the
   cheap ranking lied. This is Syed et al.'s finding at course scale.
5. `plots/circuit_scorecard.png` — discovery vs held-out: does your circuit
   survive fresh vocabulary?
6. `tables/pruned_circuit.csv` — minimality: each node's marginal value.

## Writeup questions

1. What is necessary? What is sufficient? Cite faithfulness, completeness,
   and the minimality table — and note where they pull apart.
2. Your two failure prompts (in the card): what do they share? What would
   you add to the circuit to cover them, and what would that cost in
   minimality?
3. Where did screening and causal ranking disagree most (`screen_vs_causal`)?
   Was the liar an attribution score or a motif score, and why might that be?
4. The edge claim: what does an X% routed fraction actually license you to
   say? If it came out near 0% on a model with several induction heads,
   what does THAT mean? (Hint: redundancy is not absence.)
   If no edge is claimed, check whether any previous-token head had a
   positive single-node causal drop; motif alone is not enough for an edge.
5. MDC mapping: list your entities and activities. Mark every activity you
   named but did not show. The card's filler-terms section is graded prose.

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| baseline gate drops most prompts | the model can't do the task; try tier b or simpler 2-cycles |
| faithfulness > 1.0 | fine — the complement was mildly hurting the behavior; say so, don't hide it |
| pruning stops immediately | your candidates are redundant; check whether two induction heads carry the same work |
| completeness ratio stays high | circuit too small, or the behavior has a path your screen never saw (MLPs?) |
| `head_means seq mismatch` | you added a prompt that isn't 8 tokens; the dataset contract is fixed-length |

## What goes in the ledger

2–3 claims, `CAUSAL` at circuit scope. The drafted claims name the
intervention (dataset-mean ablation), the population (8-token patterns,
listed vocabularies), and the falsifier (a different off-distribution; longer
prompts; natural text). A circuit card without its scope section is a map
drawn without a legend — pretty, portable, and wrong somewhere you can't see.
