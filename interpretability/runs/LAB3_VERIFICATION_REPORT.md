# Lab 3 verification report — attention routing and head motifs

Date: 2026-06-11 · Machine: Colab A100-SXM4-80GB · Branch: `lab1_colab`

## What was built

- **Bench**: head-level capture (`run_with_attention_cache` — patterns + per-head
  out-projection slices + per-block attention outputs in one forward), `HeadAnatomy`
  resolution (Conv1D vs Linear weight orientation handled), **head decomposition
  self-check** (per-head pieces + shared bias must rebuild each block's attention
  output), scoped head ablation (`final_pos` vs `all_pos`), and an eager-attention
  guard: transformers 5 returns an **empty attentions tuple silently** under
  sdpa/flash (verified empirically), so `needs_eager` labs auto-set eager and the
  capture hard-fails if patterns are missing.
- **Lab module**: motif scores (previous-token, induction, first-token sink, entropy),
  transparent labeling rule, per-head frozen-norm attribution (two norms compose on
  post-norm architectures — spelled out in code), natural-text confirmation of
  synthetic-labeled induction heads, scoped ablations with random/low-attribution
  controls, claims with scope-tagged causal evidence.
- **Prompts**: non-alphabetic synthetic cycles (avoiding the "A B C A B" confound where
  the alphabetic continuation equals the induction answer), period-2 cycles, natural
  repeated phrases, and no-repeat controls — all dual-tokenizer-verified.

## Two measurement lessons found and fixed during validation

1. **Sink mass is background, not a competing motif.** A naive argmax labeling rule
   assigned 0 induction heads on gpt2 even though the literature heads L5H1/L5H5 scored
   0.37–0.47 on induction — their 0.7 sink mass outranked it. Fixed: content motifs
   above the bar take priority; sink is a fallback (bar 0.5). The handout teaches the
   mistake explicitly; raw scores stay in the head table so students can re-derive it.
2. **Pairing granularity matters.** Pairing mean-across-prompts attribution with
   per-prompt ablation effects diluted the correlation; fixed to per-example pairing.
   Remaining dilution is real: pairs below the bf16 noise floor (|attr| < 0.1) are coin
   flips. Both correlations are reported (pooled and above-noise), and the scatter
   shades the noise band.

## Validation evidence

| Check | Tier A (gpt2, fp32) | Tier B (Olmo-3-7B, bf16, A100) |
|---|---|---|
| Hook parity / lens self-check | OK / OK | OK / OK |
| Component anatomy | module/module, 1.4e-7 | post_norm/post_norm, 1.4e-2 |
| Head decomposition check | OK, 3.7e-7 | OK, 2.1e-3 |
| Head labels | 5 induction (incl. L5H1, L5H5, L7H10 — the literature heads), 15 prev-token | 19 induction, 60 prev-token, 654 sink of 1024 |
| Natural-text confirmation | — | 5/8 induction heads confirmed |
| Spearman (attr vs direct ablation) | 0.54 pooled | 0.17 pooled, **0.70 above noise floor** (n=5) |

## The headline result (Tier B)

Previous-token head **L6H20** is composition caught in the act: zeroing its write at
the final position changes the answer gap by **+0.02**; zeroing it at all positions
changes it by **+2.36** (on `cycle_moon`) — a 100× direct-vs-total gap. Its causal role
runs through what it writes at earlier positions for later heads to read. This is the
two-head induction circuit measured with nothing but scoped zero-ablation, and it sets
up Lab 5's patching machinery as the natural next question.

## Runs included

- `lab03_attention_routing-*/` — Tier A smoke (gpt2)
- `lab03_tierb_full/` — Tier B science run (Olmo-3-7B, 11 prompts, 1024 heads scored,
  88 scoped ablations)
