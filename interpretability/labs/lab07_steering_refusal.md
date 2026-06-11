# Lab 7: Steering Vectors, Representation Engineering, and the Refusal Direction

**Evidence level targeted:** causality (representation-level control), with
explicit attention to what control does and does not explain.
**Prerequisites:** Labs 1–6, and Lab 4's saved `truth_direction.pt` (the
bridge loads it). **This is the first lab on an instruct model** — chat
templates are now load-bearing.

## The question

Can a direction in activation space, computed from contrast pairs, be used to
**monitor** or **change** behavior — and what does it mean that one direction
appears to mediate refusal?

## Read this first: the safety wall

Track B works with the **refusal direction**, the most consequential
single-direction result in the literature (Arditi et al.). This lab confronts
the dual-use question by design, with apparatus, not homilies. The constraints
are not optional and are enforced in code:

> - Direction extraction and the monitor use **forward passes only**.
> - **No completion is ever sampled from a refusal-eliciting prompt.**
> - Steering is only ever **toward** refusal, on **benign** prompts.
> - **Refusal ablation is not implemented.** The published result that
>   ablating this direction jailbreaks models is assigned as *reading* — you
>   discuss it, you do not reproduce it. Reading it teaches the science;
>   reproducing it teaches nothing extra and leaves a jailbroken artifact on
>   disk that nobody needs.

The `data/refusal_elicitation_set.csv` is frozen and instructor-provided; its
"harmful-sounding" prompts are category-level with no operational content,
and exist only to elicit the model's internal refusal representation. You
never author or extend that file.

## The method (Track A): a dose-response curve, not a screenshot

The honest unit of evidence for steering is a **dose-response curve with
controls**, never a cherry-picked before/after. You compute a
difference-in-means direction from sentiment contrast pairs, inject
`scale × direction` into the residual stream at one layer during generation,
and sweep the scale through zero. At each dose you measure:

| Metric | What it catches |
|---|---|
| target sentiment score | does the behavior actually move? |
| fluency (mean token logprob) | the side effect that breaks first |
| KL from the unsteered distribution | how far you've pushed the model |
| drift accuracy (unrelated facts) | collateral damage |

…against two controls on the same axes: a **random** direction of matched
norm, and a direction from **shuffled** pair labels. If the controls match
the real direction, your effect was generic perturbation, not the concept.

### What you will actually see (and a real finding)

Steering an **aligned instruct model**'s sentiment is *asymmetric*: positive
dose reliably raises the sentiment score, but negative dose barely moves it —
the RLHF "be positive and helpful" floor resists being pushed negative. That
asymmetry is not a bug in your direction; it is a measurable property of the
model, and it belongs in your writeup. The controls confirm the positive
effect is concept-specific (the real direction beats random and balanced-
shuffled controls on the positive side); the flat negative side is the model
pushing back. The injection **layer matters enormously** — mid-stack layers
steer generation; late layers carry the answer but resist redirection, so the
lab chooses the layer by *actually steering and scoring generations*, not by a
cheap next-token proxy (which picked a near-useless late layer in testing).

### Template discipline (the load-bearing detail)

Every prompt goes through the chat template before anything touches it. The
direction is read at the **generation position** of the *templated* prompt —
the same place steering later acts. Computing a direction on a raw
untemplated string and steering templated generation is the silent
meaning-level mismatch that the design guide warns about: the code runs
fine, the residual stream at "the same layer" is a different object, and your
result is quietly wrong. The bench's `apply_chat_template` is the only door.

## The result (Track B): predict vs cause

Two different properties, measured separately:

1. **Monitor (predicts):** project held-out prompts onto the refusal
   direction; ground truth is the prompt's category. The ROC/AUC says how
   well the direction *predicts* which prompts trigger refusal — **forward
   passes only, no generation**.
2. **Steer toward refusal (causes):** add the direction to **benign** prompts
   at increasing dose and measure the induced-refusal rate (a refusal-string
   classifier on the benign generations). This shows the direction is
   causally *sufficient* in the safe direction.

"Predicts refusal" and "causes refusal" are not the same claim. A direction
can do one without the other. Keep them apart in your writeup.

## The bridge: Lab 4's loop, closed

Lab 4 found truth **decodable**. Does intervening on a truth direction
**change behavior**? The lab loads your Lab 4 `truth_direction.pt` for
provenance, then **recomputes** the diff-in-means truth direction on *this*
instruct model from the same frozen cities data — because directions are
model-specific, and the saved one was computed on the base model. Steering it
shifts (or doesn't) the model's True/False assent. **Decodable-and-steerable**
and **decodable-but-inert** are both publishable sentences, and which one you
get tells you what a probe is worth as evidence.

## Running it

```bash
python interp_bench.py --lab lab7 --tier a    # SmolLM2-135M-Instruct (CPU-ok)
python interp_bench.py --lab lab7 --tier b     # Olmo-3-7B-Instruct
```

The bench picks the instruct model per tier and applies the chat template
automatically. Generation is greedy (frozen) so the only moving part across a
sweep is the dose.

## First artifact-reading path

1. `steering_claim_card.md` — the deliverable: effect, dose, side effects,
   and what the intervention does *not* show.
2. `plots/dose_response_sentiment.png` — real vs both controls; find where
   fluency breaks before the sentiment effect saturates.
3. `plots/refusal_monitor.png` — predict (forward-pass AUC).
4. `plots/induced_refusal.png` — cause (benign prompts only).
5. `plots/truth_direction_bridge.png` — the Lab 4 loop.
6. `tables/steered_examples.csv` — actual generations across the dose; read
   them, don't just trust the score.

## Writeup questions

1. At what dose does the sentiment effect exceed the random-direction
   control, and what breaks first as the dose rises? Quote the curves.
2. How well does the refusal direction *predict* refusal (AUC) versus *cause*
   it (induced rate)? Are those the same property? Argue from your numbers.
3. The truth direction: decodable-and-steerable, or decodable-but-inert? What
   does your answer imply about probes as evidence (back to Lab 4)?
4. **Hacking (entity realism — "if you can spray them, they're real"):** you
   intervened with this direction and the model moved. Is the direction
   *real*? What would distinguish your steering success from an *explanation*
   of refusal? What would change your mind?
5. **Dual use:** the refusal-ablation result was published with full methods.
   Argue *both sides* of whether it should have been — using your own Track B
   artifacts as evidence about how easy or hard the method is.

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| effect == control at all doses | direction computed on untemplated prompts? check `apply_chat_template` is in the path |
| fluency fine but no effect | steering layer too late/early; check `plots/layer_sweep.png` |
| induced refusal high for the random control too | you're measuring disruption, not refusal; tighten the classifier and hand-audit |
| monitor AUC ≈ 0.5 | the refusal/benign pairs aren't matched, or the layer carries no refusal feature |
| refusal classifier disagrees with your eyes | hand-audit 20 generations and fix the marker list — the design *expects* this |

## What goes in the ledger

2–3 claims, `CAUSAL`, **with dose and side effects in the claim text**. "The
direction steers sentiment" is not a claim; "injecting the sentiment
direction at layer L raises the sentiment score from X to Y at dose D, beating
the random control by Z, at a fluency cost of W" is. The refusal claim must
carry the safety scope in its own words. The bridge claim must say which of
decodable-and-steerable / decodable-but-inert you found, and what it implies.
