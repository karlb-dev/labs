# Lab 7 verification report — steering and the refusal direction

Date: 2026-06-11 · Machine: Colab A100-SXM4-80GB · Branch: `lab1_colab`

## What was built

The course's transition to **instruct models**, and its ethics unit done with
apparatus. New bench infrastructure (reused by Labs 8–10): chat-template
application (`apply_chat_template`), activation-addition steering hooks
(`steering_hooks`), frozen-decoding generation (`generate_text`,
`next_token_logits`), and per-lab instruct-model overrides in the registry.

Three tracks, one safety wall, all on `allenai/Olmo-3-7B-Instruct` (tier B) /
`SmolLM2-135M-Instruct` (tier A):

- **Track A** — sentiment dose-response with random + balanced-shuffled
  controls, measuring target sentiment, fluency, KL-to-unsteered, and
  unrelated-task drift.
- **Track B** — the refusal direction, **forward-pass extraction only**:
  monitor (projection vs category, ROC/AUC, no generation) and
  steer-toward-refusal on **benign** prompts (induced-refusal rate). No
  completion is ever sampled from a refusal-eliciting prompt; refusal
  ablation is not implemented.
- **Bridge** — recompute Lab 4's diff-in-means truth direction on this
  instruct model and test whether the *decodable* direction *steers*.

Frozen data: `refusal_elicitation_set.csv` (24 category-level harmful-sounding
/ matched-benign pairs, no operational content), `sentiment_contrast_set.csv`
(20 pairs), `steering_eval_prompts.csv` (12 benign prompts).

## Two correctness fixes found during validation (both became teaching points)

1. **Steering scale must be relative to the activation norm.** A unit
   direction times a fixed coefficient is ~6% of a typical residual stream and
   does nothing. Scales are now fractions of the median activation norm at the
   injection layer, so a "dose" transfers across a 135M and a 7B. Before the
   fix, Track A was flat; after, it steers cleanly.
2. **Layer selection must use actual generation, not a next-token proxy.** The
   cheap proxy (final-position logit nudge) picked a late layer (L22) with
   steering spread ~0.06, when mid-stack layers steer generation far better.
   Selection now steers and scores generations per candidate layer. The
   handout teaches *why* the proxy fails. This is the design guide's "silent
   layer mismatch in meaning, not in code" made concrete.

## Validation evidence

| Measurement | Tier A (135M) | Tier B (Olmo-3-7B-Instruct) |
|---|---|---|
| Layer chosen (by generation spread) | 16 (spread 0.71) | 20 (spread 0.39) |
| Track A: real vs random at max +dose | +0.58 (real) vs +0.00 | **+0.42 (real) vs +0.04** |
| Track A: positivity asymmetry | symmetric (weaker RLHF) | pos swing +0.31, neg +(−0.12) — RLHF floor |
| Track A: fluency at 0 → max dose | clean → breaks | −0.36 → −2.22 (side effect) |
| **Track B: refusal monitor AUC** | 0.92 | **1.00** (perfect forward-pass separation) |
| **Track B: induced refusal on benign** | 0% (135M lacks refusal) | **17% → 100%** (real) vs ~0% (random) |
| Bridge: truth direction | decodable-and-steerable (4.4 logits) | **decodable-and-steerable (17.8 logits)** |

The Tier B Track B result is the headline and it is clean: the refusal
direction *predicts* refusal perfectly (AUC 1.0, forward-pass projection on
held-out prompts) **and** *causes* it (steering benign prompts toward it
induces refusal in up to 100% of them, while the matched random direction
induces ~0%). "Predicts" and "causes" are measured and reported separately.

## Safety wall — verified honored

No completion is ever generated from a refusal-eliciting prompt (the monitor
is forward-pass projection; steering generates only on the 12 benign eval
prompts). Refusal ablation is not implemented anywhere. The jailbreak result
(Arditi et al.) is referenced as assigned reading, not reproduced.

## Runs included

- `lab07_tiera/` — Tier A smoke (SmolLM2-135M-Instruct, full pipeline)
- `lab07_tierb_full/` — Tier B (Olmo-3-7B-Instruct; ~440 steered generations)

Deliverable per run: `steering_claim_card.md` — effect, dose, side effects,
and what the intervention does *not* show, for all three tracks.
