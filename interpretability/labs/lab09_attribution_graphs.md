# Lab 9: Attribution Graphs and Circuit Tracing

**Evidence level targeted:** attribution at the feature level, upgraded to
causality only where the interventions (with controls) succeed.
**Prerequisites:** Lab 6 (keep your `circuit_card.md` on the desk — this lab
ends by confronting it) and Lab 8 (transcoders: this lab is built on the full
12-layer stack of the same Dunefsky gpt2 transcoders whose loading convention
Lab 8 validated). Base model, no chat template.

## The question

Can you read off the intermediate steps of a model's computation as a *graph
of features* — and how much should you trust a graph that was computed on a
**replacement model** rather than the model itself?

Lab 6 built a circuit by hand: weeks of screening, ablating, pruning, and
honest bookkeeping, for one behavior, in heads. This lab gets a circuit-shaped
object in seconds, automatically, in features. The price is hidden in the
small print — frozen attention, a transcoder stand-in for every MLP, error
nodes absorbing what the dictionary missed — and the lab's real subject is
that small print.

## Why this matters

Attribution graphs are how frontier interpretability is actually done in
2025–26. When Anthropic traced multi-step reasoning, multilingual circuits,
planned rhymes in poetry, and the anatomy of a hallucination in "On the
Biology of a Large Language Model," *this pipeline* — transcoders, a local
replacement model, pruned attribution graphs, feature interventions — was
the instrument. A course that stops at manual head ablation teaches circuit
discovery as it stood in 2023; a student who can only *use* circuit-tracer
as a library learns where the buttons are but not where the bodies are
buried. This lab threads the needle: you build the whole instrument from
parts you have already validated (Lab 8's transcoders, the bench's verified
hooks), so when a graph hands you a beautiful mechanism, you know precisely
which three idealizations it is standing on — and you have measured all
three.

The lab also completes the course's central arc. Lab 5 asked *where* a fact
is recovered (a patch of residual stream); Lab 6 asked *which components*
implement a behavior (heads, earned by ablation); Lab 8 asked *what units*
are worth naming (validated features). This lab composes all of it: the same
factual recall, now as a graph of named features with signed, audited edges —
plus the honest discovery that the instrument goes blind exactly where Lab
6's method shone. Methods are lenses; the capstone audit needs students who
know what each lens cannot see.

## What you build (the whole pipeline, no black boxes)

This is the method behind "Circuit Tracing" and "On the Biology of a Large
Language Model" (2025), implemented from scratch in ~400 lines so every step
is inspectable:

1. **The local replacement model.** Run the real model once on the prompt and
   freeze three things at their observed values: every attention pattern,
   every LayerNorm denominator, and — after substituting each MLP with its
   transcoder — a per-(layer, position) **error node** equal to whatever the
   transcoder failed to reconstruct. The resulting network reproduces the real
   logits **exactly** (the bench asserts it, `diagnostics/replacement_exactness.json`),
   and because everything nonlinear is now a constant, it is **linear** in its
   inputs: token embeddings, feature outputs, error vectors.

2. **Direct-attribution edges.** Linearity makes "the direct effect of feature
   s on feature t" well-defined: detach all intermediate feature gates, put a
   zero injection leaf at every MLP-write site, backprop t's pre-activation,
   and read off `edge(s→t) = activation_s · (w_dec_s · grad at s's site)`. One
   backward pass per target node yields its complete incoming-edge set — and
   the books must balance: bias path + every edge must sum to the target's
   value (`diagnostics/edge_reconstruction_check.json`, abort on failure).

3. **Backward-flow pruning.** Start at the logit node (the course metric,
   `logit(target) − logit(distractor)`), keep the strongest feature edges,
   then backward from each kept node to pull in the multi-hop sources a
   single logit pass would miss. The node budget is the compute knob
   (`--graph-nodes`; one backward per node).

4. **Interventions on the REAL model.** The graph is a *hypothesis generator*.
   Each kept feature can be edited on the real gpt2 by adding
   `(new_act − real_act) · w_dec` to the MLP output at its site. Suppress the
   subject supernode; substitute the counterfactual country's features;
   run a **random suppression of matched size**. Only these rows are claims
   about the model anyone actually runs.

## The behavior (and why it is not Dallas→Austin)

The canonical demonstration of this method is the two-hop fact
"the capital of the state containing Dallas is → Austin" on gemma-2-2b.
**This lab deviates from the course outline on purpose.** Gemma weights are
license-gated; the `circuit-tracer` library drags in the TransformerLens
dependency this course deliberately avoids; and a course rule is that nobody
runs code they can't explain. gpt2 + the ungated full-stack Dunefsky
transcoders support *the entire method* — replacement model, edges, pruning,
interventions, error accounting — at the price of a **one-hop** fact, because
gpt2-small cannot do the two-hop:

```text
The capital of France is        →  " Paris"   (distractor " Berlin")
The capital of Germany is       →  " Berlin"  (the substitution donor)
```

This is Lab 5's domain, on purpose: Lab 5 localized factual recall by patching
the residual stream wholesale; this lab re-describes the same recall at
feature granularity. What is lost relative to the published demo is the
latent intermediate entity (there is no "Texas" that never appears in the
prompt); what is kept is everything epistemically interesting — including the
substitution test, which still flips the output to the counterfactual capital.
A baseline gate (Lab 6's rule) drops any prompt the model does not already
solve, with a recorded count: you cannot trace a mechanism for a behavior the
model is not doing.

## The signed ledger (read this before the plots)

Because edges + bias reconstruct the metric *exactly*, the logit node has an
accounting identity: **bias path + embeddings + features + error nodes +
transcoder bias = logit diff**. `plots/influence_composition.png` shows this
ledger twice:

- **The fact:** features pay almost the whole bill (≈ +2.3 of +3.0). The
  recall lives in the MLP dictionary, and the graph can see it.
- **Lab 6's induction prompt, same instrument:** features pay little; copied
  **token embeddings** and **error nodes** dominate. Induction is attention
  routing, and the replacement model froze attention into the wiring — the
  behavior arrives as embedding mass moved by edges the graph cannot show you.

That second panel is the confrontation the course promised: the automated
graph is structurally blind exactly where Lab 6's manual method had to do its
work (QK pattern-matching), and Lab 6's heads-only circuit treated MLPs as
support exactly where this graph lives. Neither artifact is "the mechanism."

## Running it

```bash
python interp_bench.py --lab lab9 --tier a    # gpt2, CPU-ok, small node budget
python interp_bench.py --lab lab9 --tier b     # same model, bigger graph + battery
```

gpt2 on **every** tier (the registry pins it): it is the only ungated model
with a public all-layers MLP-transcoder set. Tiers raise the node budget and
the paraphrase battery, not the model. The 12 transcoders download once
(~2 GB). Float32 everywhere — the exactness check is calibrated for it.
`--graph-nodes N` overrides the budget; `--max-examples` caps the paraphrase
battery.

## First artifact-reading path

1. `graph_card.md` — the deliverable, mirroring Lab 6's circuit card.
2. `plots/attribution_graph.png`, then `graphs/pruned_graph.json` and
   `tables/graph_nodes.csv` — the object itself; check the de-embeddings
   (the L0 France feature promotes " Alps", " Marse…" — read a few).
3. `tables/intervention_results.csv` + `plots/intervention_effects.png` —
   the causal test: baseline / suppress / substitute / random control.
4. `plots/influence_composition.png` — the two-panel signed ledger.
5. `tables/paraphrase_robustness.csv` — which subject-site features recur
   across surface variants, and which were template artifacts.
6. `graphs/supernode_map.json` — **auto-proposed** groupings; edit before
   citing. Membership must be defended from top contexts, not de-embeddings
   alone.
7. `diagnostics/` — exactness, edge-reconstruction, and feature-edit no-op
   checks. If any of these failed, nothing above would mean anything.

## Writeup questions

1. Did the substitution flip the model toward the counterfactual capital, and
   did the random control leave the behavior alone? State exactly which rung
   of the evidence ladder each outcome puts your mechanism claim on.
2. What fraction of the logit's direct |edge| mass routes through error
   nodes, and what do you tell a reviewer who says "then the graph isn't an
   explanation"? (You will want the ethics reading below.)
3. Which subject-site features recurred across ≥ n−1 paraphrases, and which
   appeared once? What does a single-template feature *mean*?
4. The signed ledger for induction: who pays for the +0.6 logit diff, and why
   is the feature column small? Where did that computation go?
5. **Lab 6 vs Lab 9, honestly.** What did the manual method force you to
   verify that the graph hands you for free? Where does each hide its
   assumptions (metric and template choice vs. replacement-model fidelity,
   node budget, frozen attention)? Which artifact do you trust more, *for
   what kind of claim*?

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| replacement exactness check fails | wrong dtype (must be fp32), or attention patterns missing (must be eager — the registry forces it) |
| edge reconstruction off by a lot | a frozen quantity is live: LN sigma recomputed instead of captured, or feature gates not detached |
| all top edges are error nodes | the transcoders don't fit this prompt's computation — check `transcoder_stack_report.json` FVU per layer first |
| suppression does nothing | supernode too small or at the wrong position — check `subject` token matching in `tables/graph_nodes.csv` |
| random control moves as much as suppression | your "supernode" was generic perturbation; the claim dies (and that is a result, not a bug) |
| baseline gate drops the primary fact | the model can't do the behavior at this size/precision; there is nothing to trace |

## Extensions

- **Manageable:** a second vignette in miniature — run the pipeline on
  `"The capital of Italy is"` as the primary fact (one flag-free code edit)
  and report whether the *same* L0/L8 feature families appear with Italy
  features in place of France's. Artifact: a second `graph_card.md`.
- **Ambitious:** add QK-attribution. The frozen patterns are constants; for
  one induction head, differentiate the *pattern* (unfreeze the softmax for
  that head only) and ask which source-position features sharpen attention to
  the answer token. This is exactly the piece the replacement model amputates,
  and grafting it back — even for one head — earns the right to criticize the
  method from above rather than below.

## Interpretation & ethics — idealization

**Reading:** a chapter of Potochnik, *Idealization and the Aims of Science*
(2017) — or any serious treatment of idealized models in science.

The graph describes a **replacement model** that imitates the real one, with
error nodes absorbing the residue, attention frozen into wiring, and a
dictionary deciding what can be seen at all. Your own artifacts quantify the
idealization: the error-node share, the kept-coverage number, the induction
panel where the instrument goes blind.

**Writing prompt:** Defend or attack — *"an idealized model that supports
successful interventions is explanation enough."* You must use your own
numbers: the substitution flip and its random control argue one way; your
error-node share and the induction ledger argue the other. Two paragraphs,
one per side, then a verdict you are willing to sign.

## What goes in the ledger

3–4 claims. The graph-structure claim is `ATTR` and must carry the signed
shares and the kept-coverage number. The intervention claim is `CAUSAL` and
must name the supernode size, both effect sizes, and the random control. The
Lab 6 confrontation claim is `OBS` and should be written so that it *retires
or scopes* something you believed after Lab 6 — that is the ledger working.
The paraphrase claim is `ATTR`. Do not upgrade the graph itself to CAUSAL;
the graph proposed, the interventions disposed.

## Reading

- Ameisen et al., "Circuit Tracing: Revealing Computational Graphs in
  Language Models" (2025) — the method this lab reimplements in miniature.
- Lindsey et al., "On the Biology of a Large Language Model" (2025) — what
  the method finds at frontier scale; read the Dallas→Austin section and
  compare with your one-hop graph.
- Dunefsky et al., "Transcoders Find Interpretable LLM Feature Circuits"
  (2024) — the transcoders you are using, doing this job at gpt2 scale.
- Marks et al., "Sparse Feature Circuits" (2024) — the SAE-based sibling.
- Potochnik, *Idealization and the Aims of Science* (2017), one chapter.
