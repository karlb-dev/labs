# Official reference targets (Claude-context only; gate and tune nothing)

Exact values stated in the paper's text/captions
(transformer-circuits.pub/2026/workspace · arXiv:2607.15495v1), recorded
per plan §5.2 before GPU work. Claude values are external context for
direction and rough magnitude — **never Qwen/OLMo acceptance thresholds,
never used to tune a strength after seeing open-model outcomes**. The
addendum §3.4 receipts (independently audited against paper text) are the
primary source; the paper was re-fetched 2026-08-08 and the quotes below
re-verified.

## Verbal report (paper §3.1) — model: Claude (Sonnet 4.5 primary)

| Metric | Value | Unit/n | Source |
|---|---|---|---|
| J-lens-vector swap drives target into top-5 | **88%** of trials | 14×first-ten candidate trials | §3.1 text |
| J-space component swap | **59%** | same | §3.1 text |
| non-J-space component swap | **5%** | same | §3.1 text |

Predicted direction (all lanes): coordinate swap raises swapped-in
candidate's rank; success = rank 1 (release rule) / top-5 (paper
context metric).

## Verbal introspection (§3.1)

n=100 concepts (release ships 101 — D3); metric = median reciprocal rank
of injected surface at the open quote vs steering strength; injection
over the user turn. "The model reports the injected concept in the
majority of trials." Direction: MRR increases with strength from the
strength-0 control.

## Internal reasoning / probe-swap (§3.3)

| Metric | Value | Set | Source |
|---|---|---|---|
| raw J-lens intermediate swap top-1 | **54% / 70% / 70%** (Haiku 4.5 / Sonnet 4.5 / Opus 4.5) | 50-prompt set | §3.3 text |
| raw J-lens intermediate swap top-1 | **60%** | released n=90 set | §3.3 text |
| probe J-space component | **61%** | n=90 | §3.3 text |
| non-J component | **28%**, → **6%** under J-clamp | n=90 | §3.3 text |

Study-1 arm = the raw J-lens swap (the paper's own headline §3.3 arm).
Direction: swap flips next-token answer to `swap_answer`; success =
top-1.

## Flexible generalization (§3.4, A.13)

**76/192** at α=1 → **101/192** at α=2 (Sonnet 4.5), heavy per-category
variance (paper appendix). Direction: swap moves answer to target
argument's answer; α=2 ≥ α=1 expected.

## Workspace band (§3.3/§4.1 figure text) — the pinned provenance quote

> "Beginning about a third of the way through (~L38) and ending shortly
> before the output (~L92), as the region where the J-space carries
> persistent, abstract content distinct from both the input tokens and
> the imminent output."

Band mapping to 64 layers is a preregistered cross-model convention
(LAYER_BAND_CONTRACT).

## Lens-fit recipe (§2 methods)

> "…a corpus of one thousand prompts sampled from a pretraining-like
> distribution" — 1000 × 128-token sequences (upstream README: quality
> saturates quickly; ~100 prompts usable). Anchors the n=1000 parity
> ceiling and legitimizes the n=250/500 tiers.

## Qualitative paper-text directions (no exact numbers in running text)

- Lens quality: "the two lenses agree closely in the model's last
  several layers and diverge earlier, with the J-lens recovering
  interpretable content at depths where the logit lens does not"
  (methods-comparison). Direction: J-lens pass@k ≥ logit-lens pass@k at
  mid depths.
- Ignition: share "sits near one endpoint or the other, switching
  sharply" from ~workspace onset. Direction: late-layer transition width
  < early-layer width.
- Directed modulation: focus > suppress > (not zero); "Under the ignore
  instruction, target presence is substantially lower than under the
  focus instruction, but it is not zero" (white-bear residue).
- Selectivity-language: explicit − automatic > 0 ("largely unmoved"
  controls).
- Dual-task: single − dual reachability > 0 (interference).
- Capacity quote for context: "no more than 25 … the number of J-lens
  vectors that are meaningfully active at a given time" (sparse-readout
  context, **not** the task-capacity metric; never merged with campaign
  occupancy).

## Figure-digitized appendix (non-gating)

None digitized for Study 1 — figure-only values (per-set pass@k bars,
modulation rates, dual-task interference, Q2−Q1 fractions) are left
undigitized; comparisons to them are qualitative-direction only.
