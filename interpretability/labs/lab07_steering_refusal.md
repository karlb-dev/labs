# Lab 7: Steering Vectors, Representation Engineering, and the Refusal Direction

**Evidence level targeted:** causal intervention for generation steering, plus a forward-pass monitor for refusal.

**Prerequisites:** Labs 1-6. Lab 4's `truth_direction.pt` is useful provenance when present, but this lab recomputes the truth direction on the current instruct model because directions are model-specific.

**Model family:** instruct models only. Every prompt goes through the tokenizer chat template before any activation is read or changed.

## The question

Can a direction in activation space, computed from contrast pairs, be used to monitor or change behavior? If a refusal direction is predictive and steerable, what does that show, and what does it still fail to explain?

This lab has three linked experiments:

1. **Track A:** build a sentiment steering direction and measure a dose-response curve with controls.
2. **Track B:** build a refusal direction with forward passes only, use it as a monitor, then steer benign prompts toward refusal.
3. **Bridge:** revisit Lab 4's truth probe by injecting a recomputed truth direction and separating answer bias from truthfulness.

The unit of evidence is not a single spicy generation. The unit of evidence is a curve, a control, an artifact, and a caveat.

## Why adding a vector can work at all

Labs 1-6 read the residual stream. This lab writes to it. The bet is the
linear representation hypothesis: if a feature rides the stream as a
direction, then `activations + scale * direction` moves every downstream
reader of that feature at once — the way patching moved a fact in Lab 5,
except you authored the edit instead of borrowing it from a clean run.
Difference-in-means over matched pairs is the cheapest estimator of such a
direction: the pairing cancels topic and syntax, leaving the one thing you
varied. None of this is guaranteed. The dose-response curve tests the bet;
the two controls test the estimator.

## Safety wall for Track B

Track B is safety-relevant and intentionally constrained. The constraints are part of the experiment, not classroom decoration:

- Direction extraction uses **forward passes only**.
- The refusal monitor uses **held-out projections only**.
- The lab **never samples a completion from a refusal-eliciting prompt**.
- Steering is only **toward refusal**, and only on **benign prompts**.
- Refusal ablation is **not implemented**.

The frozen `data/refusal_elicitation_set.csv` exists only to produce the internal contrast for a forward pass. Students do not author, extend, or generate from that file. The run writes `diagnostics/lab07_safety_audit.json` so the safety wall leaves a footprint in the artifacts.

## The load-bearing implementation detail: where the vector lives

The bench's steering hook adds a vector to a decoder block's **output**. Under the course stream convention, decoder block `k` writes the residual stream called `streams[k + 1]`. The revised lab therefore extracts directions from `streams[injection_layer + 1]` at the final prompt position and injects them into `bundle.blocks[injection_layer]`.

That off-by-one detail is easy to miss and expensive to ignore. Reading from `streams[k]` while injecting into block `k` means you measured one residual stream and changed the next one. The code now names both fields in the tables:

```text
injection_layer = decoder block whose output receives the vector
stream_depth    = residual-stream depth where the direction is read
```

Dose is also normalized. A scale of `1.0` means "one median activation norm at the chosen injection site," not "add a raw unit vector." This makes the sweep more comparable across a 135M smoke model and a 7B course model.

## Track A: dose-response steering, not before/after theater

Track A computes a positive-minus-negative sentiment direction from frozen contrast pairs, chooses an injection layer by actually steering generations, and sweeps through negative, zero, and positive doses.

At every dose it records four measurements:

| Metric | Why it matters |
|---|---|
| sentiment score | target behavior: did the generation move? |
| mean token logprob | fluency side effect: did the text become unlikely or degenerate? |
| KL from the unsteered next-token distribution | distribution shift: how hard did the vector push? |
| unrelated fact accuracy | collateral damage: did steering leak into another task? |

It compares the real direction against two controls on the same axes:

| Control | What it rules out |
|---|---|
| random unit direction | generic activation perturbation |
| shuffled-label direction | contrast-set structure without the concept label |

The main plot is `plots/dose_response_sentiment.png`. The updated version has four panels: target score, fluency, KL, and drift. Do not claim success from the target panel alone. A vector that moves sentiment only after fluency collapses is not a clean concept intervention. It is a smoke machine wearing a lab coat.

The sentiment score itself is a transparent word-list meter, and the list
shares adjectives with the contrast pairs. A generation that parrots
contrast-set vocabulary maxes the meter without expressing sentiment about
anything. Before claiming a dose, read `tables/steered_examples.csv` at that
dose: in the reference run the max-dose example is visibly degenerating while
the target panel still rises — the fluency panel exists because the target
panel can be gamed.

Expect asymmetry on an RLHF'd assistant. In the reference tier B run the
positive swing is +0.31 while the negative swing is only -0.12: the model's
positivity prior resists downward steering. The 135M smoke model shows the
opposite shape (+0.18 up, -0.40 down). Asymmetry is a measurement of the
model's prior, not a bug in your direction — and it is invisible in any
single before/after pair, which is the argument for the sweep.

### Layer selection

The lab no longer talks about a cheap next-token proxy. It chooses the layer by generation-based sweep and writes:

- `plots/layer_sweep.png`
- `tables/layer_sweep.csv`
- `tables/layer_sweep_by_prompt.csv`

The chosen block is the one with the largest positive-minus-negative sentiment spread at the probe dose. This is slower than a proxy, but it measures the object students are actually claiming: generated behavior.

One honesty note the claims carry forward: the sweep prompts are a subset of
the evaluation prompts, so the layer is selected on data that also produces
the headline number. With 4-12 eval prompts a disjoint split would cost more
power than the bias is worth, but the C1 falsifier names it — re-selecting
the layer on fresh prompts should not move the effect materially.

## Track B: refusal direction, predict versus cause

Track B builds a refusal direction from training contrast pairs:

```text
mean(refusal-eliciting forward-pass activations) - mean(matched-benign forward-pass activations)
```

Then it asks two separate questions.

### 1. Does the direction monitor held-out prompt category?

Held-out refusal-eliciting and benign prompts are projected onto the direction. This produces an AUC and a small ROC-style table. No held-out refusal-eliciting prompt is generated.

Artifacts:

- `plots/refusal_monitor.png`: projection histograms plus ROC curve
- `tables/refusal_monitor_table.csv`: threshold, true positive rate, false positive rate
- `tables/refusal_monitor_examples.csv`: projection values, with text marked as not generated

This is a forward-pass monitor. It predicts held-out category. It is not evidence about sampled harmful completions.

### 2. Does the direction cause refusal when added to benign prompts?

The lab then steers only benign evaluation prompts toward the refusal direction and measures induced-refusal rate with a simple string classifier. A random direction is plotted as a control.

Artifacts:

- `plots/induced_refusal.png`: induced-refusal rate versus dose
- `tables/induced_refusal_curve.csv`: refusal counts, rates, and standard errors
- `tables/induced_refusal_generations.csv`: benign generations and matched refusal markers

Hand-audit the generations whenever this curve supports a claim. The classifier is a transparent meter, not an oracle.

Two reading rules for this curve. First, the dose-0 rate is the classifier
floor, not steering: markers like "as an AI" fire on benign disclaimers
(reference tier B floor: 0.17 = 2/12 prompts). Read the curve relative to its
own floor and quote the random direction at the same dose. Second, with 12
eval prompts the rate moves in steps of 1/12 — quote counts next to rates:
100% means 12/12 here, which licenses a claim about these prompts, not a
population.

## Bridge: Lab 4's truth direction, split into two claims

Lab 4 showed truth was linearly decodable. Lab 7 asks whether a truth direction steers a True/False readout on the instruct model. The direction is fit on one split of the frozen truth-pair set and evaluated on held-out pairs from that same family.

The old shortcut was tempting: plot `logit('True') - logit('False')` and call the direction steerable. The revised bridge is more careful. It plots two quantities:

| Quantity | Meaning |
|---|---|
| answer bias | mean `logit('True') - logit('False')` across all statements |
| signed truth margin | `True-False` for true statements, `False-True` for false statements |

This distinction matters. A vector can make the model say "True" more often without making it more truthful. The bridge verdict is one of:

- `decodable-but-inert`: steering barely moves the True/False readout.
- `decodable-and-steers-True-assent`: steering mainly changes answer bias.
- `decodable-and-improves-truth-margin`: steering improves the signed correctness margin on the held-out truth split.

Artifacts:

- `plots/truth_direction_bridge.png`: answer bias and signed truth margin, real versus random direction
- `tables/truth_direction_bridge.csv`: aggregate bridge metrics
- `tables/truth_direction_bridge_by_statement.csv`: held-out per-statement readout values

This is the probe lesson in miniature: decodable, steerable, and explanatory are three different words with three different jobs.

## Running it

CPU smoke path:

```bash
python interp_bench.py --lab lab7 --tier a --prompt-set small
```

Standard course run:

```bash
python interp_bench.py --lab lab7 --tier b --prompt-set full
```

Useful debugging runs:

```bash
python interp_bench.py --lab lab7 --tier b --prompt-set small --max-examples 4
python interp_bench.py --lab lab7 --tier b --prompt-set medium --no-plots
```

`--prompt-set` controls the default dataset cap. `--max-examples` is then applied as a hard cap, so the smoke path remains quick and the full path remains full.

One tier A footgun: the global tier default `--max-examples 4` also caps the
refusal pairs to 4, so the smoke monitor AUC is computed on 2 held-out pairs.
It checks plumbing, not the monitor.

## First artifact-reading path

Read the artifacts in this order:

1. `steering_claim_card.md`: the shortest defensible interpretation.
2. `plots/dose_response_sentiment.png`: Track A effect and side effects in one figure.
3. `tables/dose_response_by_prompt.csv` and `tables/steered_examples.csv`: the actual generations behind the score.
4. `plots/refusal_monitor.png`: forward-pass monitoring, not generation.
5. `plots/induced_refusal.png`: benign prompt steering, with random control.
6. `diagnostics/lab07_safety_audit.json`: what was and was not generated.
7. `plots/truth_direction_bridge.png`: answer bias versus signed truth margin.
8. `ledger_suggestions.md`: drafted claims with measured numbers.

## What a good writeup says

A good writeup keeps three separations alive:

1. **Real direction versus controls:** quote the target effect and the random/shuffled gaps.
2. **Effect versus side effect:** quote the fluency, KL, and drift costs at the dose you want to claim.
3. **Predict versus cause:** the refusal AUC and the induced-refusal rate are different evidence.

A good bridge answer does not say "the truth direction works" until it checks the held-out signed truth-margin panel. If only the answer-bias panel moves, write that down. That is not a failed lab. It is a sharper claim.

## Writeup questions

1. At what positive dose does the sentiment direction first beat both controls? What happens to fluency, KL, and drift at that dose?
2. Is the negative side of the sentiment sweep symmetric with the positive side? Give the measured swings rather than a vibe report.
3. How well does the refusal direction monitor held-out prompt category? How well does it cause refusal on benign prompts? Why are those not the same claim?
4. Did the random direction induce refusal too? If yes, how much of the curve might be generic disruption rather than a refusal feature?
5. For the truth bridge, which verdict did you get: inert, True-assent steering, or improved truth margin? What does that imply about Lab 4's probe evidence?
6. Hacking's entity-realism challenge: you intervened on a direction and behavior moved. Is the direction real? What extra evidence would turn steering success into a mechanistic explanation?
7. Dual-use prompt: the refusal-ablation result was published with methods. Argue both sides of whether that was justified, using your own Track B safety audit and curves as evidence.

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| effect equals controls at every dose | check `tables/layer_sweep.csv` and confirm the real direction is not being read from the wrong stream depth |
| strong target effect but bad text | check fluency and KL panels before claiming clean steering |
| random direction induces refusal too | audit `tables/induced_refusal_generations.csv`; the classifier may be catching disruption |
| induced-refusal rate is nonzero at dose 0 | that is the classifier floor, not steering: markers like "as an AI" fire on benign disclaimers. Read the curve relative to its floor and quote the random direction at the same dose |
| monitor AUC near 0.5 | inspect pair matching and try a different injection block from `layer_sweep.csv` |
| truth bridge looks strong but false statements get worse | the vector is steering True-assent, not truthfulness; use the signed-margin panel |
| smoke run is too slow | use `--prompt-set small --max-examples 4 --no-plots` for plumbing only |

## What goes in the ledger

Write 2-3 claims with dose, control gaps, and side effects in the claim text. Avoid cloud-shaped claims like "the direction controls sentiment." Use measured claims:

```text
[L07-C1] CAUSAL | Injecting the sentiment direction at decoder block L changes score X to Y at dose D, beating random by R and shuffled by S, with fluency cost F and drift cost G.
Artifact: runs/.../plots/dose_response_sentiment.png | Falsifier: controls match the real curve or the effect appears only after fluency collapse.
```

Tag with care. The monitor result is `DECODE` evidence: a projection
predicted a held-out category — exactly Lab 4's evidence class with a refusal
label instead of a truth label. Only the induced-refusal curve is `CAUSAL`.
Split Track B into two claims, one per tag. AUC 1.00 next to induced refusal
100% on the same direction is the whole course in one row — but only if the
tags keep them apart.

The refusal claim must carry the safety scope in its own words. The bridge claim must state whether the direction changed answer bias, signed truth margin, or neither.
