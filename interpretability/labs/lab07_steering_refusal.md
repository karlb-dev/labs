# Lab 7: Steering Vectors, Representation Engineering, and the Refusal Direction

**Evidence level targeted:** causal intervention for generation steering, plus a forward-pass monitor for refusal.

**Prerequisites:** Labs 1-6. You already know the residual-stream indexing and "readout is an instrument" caution from Lab 1, the frozen-norm linearization and "attribution is a ledger, not causation" discipline from Lab 2, the routing-vs-contribution distinction from Lab 3, the "decodable does not mean used" skepticism and truth direction from Lab 4, patching/interchange from Lab 5 (here you author the edit instead of borrowing a clean activation), and manual circuit scope/evidence from Lab 6. Lab 4's `truth_direction.pt` is useful provenance when present, but this lab recomputes the truth direction on the current instruct model because directions are model-specific. The bridge re-uses the Lab 4 truth-pair family but splits the "decodable" claim into answer bias vs signed truth margin.

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

**Make the concept pop:** the safety wall is measured and auditable. After the run, open the JSON: it records zero refusal-eliciting generations and confirms the monitor/steering boundaries. This is the lab's way of making "we did not do the dangerous thing" a reproducible artifact rather than a promise. The same discipline (forward-only for the monitor, controls, scoped claims) appears in Labs 4–6; here it is the central object of study.

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

**Make the concept pop:** open `layer_sweep.png` (or the csv). The early layers often show near-zero spread on the sweep prompts; mid-stack (here block 20 / stream 21) wins. The proxy would have picked a late layer with almost no generation effect. Layer choice is now itself an intervention measurement, exactly as Lab 5 forces you to confront where the fact actually lives before you patch.

One honesty note the claims carry forward: the sweep prompts are a subset of
the evaluation prompts, so the layer is selected on data that also produces
the headline number. With the current eval set size a disjoint split would cost more
power than the bias is worth for some analyses, but the C1 falsifier names it — re-selecting
the layer on fresh prompts should not move the effect materially.

**Headline numbers note:** In full runs this lab uses all 28 sentiment pairs and 28 refusal pairs in the shipped sets, 24 eval prompts, and 12 drift facts (across categories). The qualitative story (dose-response shape, real-vs-control gaps, monitor/induced dissociation, bias-vs-margin split, safety wall) is supported by structure, multiple controls, and per-prompt artifacts; none of the percentages or rates should be treated as having more than one significant figure of confidence.

**Battery overlap caveat:** the drift battery is not 12 independent trials —
7 of the 12 facts are capital-city completions (and they overlap the
capital-fact sets Labs 5 and 11 probe). A steering side effect that
specifically disturbed geographic recall would move 7 facts at once and read
as a 0.58 drift drop, while a same-sized side effect on arithmetic would read
as 0.17. Treat drift accuracy as a coarse canary with a capitals bias, not a
balanced capability battery; if you extend the lab, balancing the battery
across 4+ unrelated topic clusters is the first cheap improvement.

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


## Visualization upgrade: make the handle, cost, and safety scope visible

The revised plotting code keeps the original Track A / Track B / Bridge spine, then adds synthesis artifacts that make the evidence ladder harder to blur:

| Artifact | What it teaches |
|---|---|
| `plots/steering_evidence_dashboard.png` | the whole lab in one cockpit: target movement, side-effect cost, refusal predict-vs-cause, and truth bridge verdict |
| `tables/steering_evidence_matrix.csv` | one row per claim-bearing object: evidence rung, headline metric, controls, side effects, and caveat |
| `tables/dose_operating_points.csv` | a transparent operating-point table: target movement, control gaps, KL, fluency, drift, and a claimability flag |
| `plots/prompt_steering_response_heatmap.png` + `tables/dose_response_by_prompt.csv` | whether the aggregate sentiment curve is broad or carried by one excitable prompt row |
| `plots/dose_operating_frontier.png` | target movement versus side-effect cost: the "steering handle or smoke machine?" plot |
| `plots/refusal_safety_dashboard.png` | paired monitor separation, benign-only induced refusal, classifier floor, and the safety wall footprint |
| `plots/induced_refusal.png` + `tables/induced_refusal_generations.csv` | which benign prompts and doses tripped the transparent refusal marker classifier |
| `plots/truth_bridge_statement_atlas.png` + `tables/truth_bridge_statement_summary.csv` | which held-out truth statements improved or worsened under the truth direction |
| `plots/steering_direction_cosines.png` + `tables/steering_direction_cosines.csv` | whether sentiment, refusal, truth, and control vectors are distinct handles or nearly the same axis wearing different nametags |
| `plots/refusal_safety_dashboard.png` + `diagnostics/lab07_safety_audit.json` | a visual and machine-readable footprint of what was forward-only, what was benign-only generation, and what was not implemented |
| `tables/plot_reading_guide.csv` | the artifact map for students and future report generation |

The new plots do not add a new scientific rung. They make the existing rungs legible: steering is a causal handle, monitor AUC is decode evidence, and the truth bridge remains an operationalization audit rather than a declaration that the model became truthful.

## Running it

Always run Tier A smoke first (instrument checks + end-to-end plumbing on the small instruct model).

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
It checks plumbing, not the monitor. The refusal direction and the truth bridge
both require at least 4 pairs for a train/held-out split; Tier A smoke still
exercises the full pipeline and writes the safety audit and claim card.

## First artifact-reading path

Instrument health first (as in every prior lab), then the payload artifacts that separate the claims.

1. `diagnostics/hook_parity.json`, `logit_lens_self_check.json`, and `model_anatomy.json` (instrument hygiene; the chat template and residual streams must line up before any steering claim).
2. `plots/steering_evidence_dashboard.png` + `tables/steering_evidence_matrix.csv`: start here. The dashboard is the cockpit; the evidence matrix is the flight recorder.
3. `steering_claim_card.md`: the shortest defensible interpretation, with safety wall and "what it does not show" for every track.
4. `plots/layer_sweep.png`, `plots/layer_selection_detail.png`, and `tables/layer_sweep.csv`: generation-based (not proxy) choice of injection site. The spread column is the actual steering effect you later claim.
5. `plots/dose_response_sentiment.png` + `plots/dose_operating_frontier.png` + `tables/dose_operating_points.csv`: Track A target, fluency, KL, drift, and effect/cost tradeoff. Look for the first dose where real beats both controls before KL or fluency collapse starts eating the furniture.
6. `plots/prompt_steering_response_heatmap.png` + `tables/dose_response_by_prompt.csv` + `tables/steered_examples.csv`: prompt-level heterogeneity and generation previews. If one prompt carries the curve, the claim must say so.
7. `plots/refusal_monitor.png` + `plots/refusal_safety_dashboard.png` + `tables/refusal_monitor_table.csv` + `tables/refusal_monitor_examples.csv`: forward-pass DECODE monitor on held-out pairs. This predicts category labels from activations; it is not generated harmful behavior.
8. `plots/induced_refusal.png` + `plots/refusal_safety_dashboard.png` + `tables/induced_refusal_curve.csv` + `tables/induced_refusal_generations.csv`: CAUSAL steering on benign prompts only, with random control and classifier-floor audit. Hand-audit the generations at the dose you claim.
9. `diagnostics/lab07_safety_audit.json` + `plots/refusal_safety_dashboard.png`: machine-checkable safety wall (zero refusal-eliciting generations, forward-only monitor, refusal ablation not implemented).
10. `plots/truth_direction_bridge.png` + `plots/truth_bridge_statement_atlas.png` + `tables/truth_direction_bridge.csv` + `tables/truth_direction_bridge_by_statement.csv` + `tables/truth_bridge_statement_summary.csv`: Lab 4 bridge split into answer bias versus signed truth margin, with per-statement deltas. If bias moves while margin does not, the verdict is `decodable-and-steers-True-assent`, not "more truthful."
11. `plots/steering_direction_cosines.png` + `tables/steering_direction_cosines.csv`: direction-geometry confound audit; if two handles are nearly collinear, the claims should say so.
12. `tables/plot_reading_guide.csv` and `ledger_suggestions.md`: plot map plus the three drafted claims.

## What a good writeup says

A good writeup keeps three separations alive:

1. **Real direction versus controls:** quote the target effect and the random/shuffled gaps.
2. **Effect versus side effect:** quote the fluency, KL, and drift costs at the dose you want to claim.
3. **Predict versus cause:** the refusal AUC and the induced-refusal rate are different evidence. The monitor is forward-pass category prediction on held-out pairs (DECODE, Lab 4 style); the induced curve is a causal intervention on benign prompts only (CAUSAL).

A good bridge answer does not say "the truth direction works" until it checks the held-out signed truth-margin panel. If only the answer-bias panel moves (or the margin gets worse at high dose), the verdict is `decodable-and-steers-True-assent`. Write the actual spans and the verdict label. That is not a failed lab. It is a sharper claim that distinguishes "the direction moves the True/False token distribution" from "the direction is a truthfulness mechanism the model uses."

**Make the concept pop:** the unit of evidence is the curve + the control gap + the side-effect cost + the safety footprint + the bias-vs-margin split. One spicy steered sentence is an anecdote; the dashboard, prompt heterogeneity heatmap, effect/cost plot, monitor-vs-induced pair, safety-wall plot, and truth bridge are the evidence bundle. The claim card, evidence matrix, and the three ledger drafts (C1/C2/C3) are written to force you to keep the distinctions.

## Writeup questions

1. At what positive dose does the sentiment direction first beat both controls on the target score (see `dose_response_sentiment.png` and the per-prompt table)? At that same dose, what do the fluency, KL, and drift panels show? Quote the real-vs-control gaps.
2. Is the negative side of the sentiment sweep symmetric with the positive side? Give the measured swings (positive at max dose vs negative at min dose) rather than a vibe report. What does the asymmetry tell you about the model's prior (see also the layer sweep spread and steered examples at high positive dose)?
3. How well does the refusal direction monitor held-out prompt category (refusal_monitor.png + AUC + table)? How well does it cause refusal on benign prompts (induced_refusal.png + curve + generations)? Why are those two numbers not the same claim? Hand-audit the generations at the doses where the curve rises.
4. Did the random direction induce refusal too? At which doses? How much of any induced-refusal curve might be generic disruption (or the classifier floor at dose 0, e.g. "as an AI" on ordinary disclaimers) rather than a refusal feature?
5. For the truth bridge, which verdict did you get: inert, `decodable-and-steers-True-assent`, or `decodable-and-improves-truth-margin`? Quote the answer-bias span vs the signed truth-margin span from the two-panel plot and the by-statement table. What does the gap (or lack of gap) imply about Lab 4's probe evidence?
6. Hacking's entity-realism challenge: you intervened on a direction and behavior moved. Is the direction real? What extra evidence (controls, side-effect costs, held-out generalization, safety audit, re-selection of layer, different prompt family) would turn steering success into a mechanistic explanation?
7. Dual-use prompt: the refusal-ablation result was published with methods. Argue both sides of whether that was justified, using your own Track B safety audit (`lab07_safety_audit.json`), the monitor-vs-induced gap, and the random control as evidence. Keep the scope of what this lab actually implemented.

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
