# Lab 8 verification report — superposition, SAEs, and transcoders

Date: 2026-06-11 · Machine: Colab A100-SXM4-80GB · Branch: `lab1_colab`

## What was built

The course's superposition / sparse-dictionary unit, on **base models** with
**pretrained dictionaries** loaded from the Hub (no SAE training on the hot
path; training is the COURSE.md Tier-C extension). Three parts plus two bridges:

- **Part 0 — toy superposition (CPU, model-free).** The Elhage et al. toy
  autoencoder, packing 20 sparse features into 5 dimensions. Reproduces the
  canonical collapse: **5/20 features represented when dense** (orthogonal,
  interference 0.00) rising to **17/20 when sparse** (superposition).
- **Part 1 — feature atlas.** Run the SAE over the frozen domain-tagged corpus
  (`data/sae_feature_corpus.csv`, 166 lines / 10 domains in confusable pairs),
  rank features by peak activation and by firing frequency (overlap 0 — the
  rankings disagree), label from top contexts, and **validate** each label:
  held-out AUC vs domain membership, an adversarial confusable-pair AUC
  (concept vs token feature), and polysemanticity entropy → a verdict.
- **Part 2 — transcoder.** A gpt2 MLP transcoder verified by reconstruction
  FVU **and** the downstream-logit KL of splicing its reconstruction in for the
  real MLP output, plus de-embedded features (the bridge to Lab 9). Runs on
  gpt2 on every tier (auxiliary gpt2 loaded on Tier B).
- **Bridges.** Cosine of the best SAE feature's decoder direction with Lab 4's
  truth direction; and a **CAUSAL feature-clamp** with a random-feature control
  and a fluency proxy.

New frozen data: `data/sae_feature_corpus.csv` (+ `data/make_sae_corpus.py`).
No new bench infrastructure was required — the lab reuses `steering_hooks`,
`generate_text`, and the block references resolved by `resolve_anatomy`.

## The loading conventions were validated empirically, not assumed

Each dictionary expects its activations in a specific form; the wrong form
silently inflates FVU while the code runs. Settled by measurement at authoring
time (the single most important correctness work in this lab):

1. **gpt2 resid SAE (jbloom):** TransformerLens-trained, so the residual stream
   is centered — the SAE reconstructs well only on a **per-token-demeaned**
   input. Wrong: FVU ≫ 1. Right (`center_input=True`): **FVU 0.0019**, L0 ≈ 65.
2. **gpt2 MLP transcoder (Dunefsky):** input is the **bare LayerNorm** of the
   pre-MLP residual (no affine γ/β), no `b_dec` subtraction. Using the model's
   full `ln_2` output gives FVU ≈ 1.0; the bare-LN convention gives **FVU 0.34**
   on raw HF activations (matches the published transcoders' ballpark).
3. **Olmo SAE (decoderesearch):** **jumprelu** — features below a learned
   per-feature threshold are exactly zero. ReLU instead gives FVU 0.51 at L0
   3000; jumprelu gives **FVU 0.31, L0 ≈ 100**.

## Two design fixes found during validation (both became teaching points)

1. **Clamp by multiples of the feature's peak activation, not a unit vector.**
   A unit decoder direction times a small constant does nothing in a
   residual stream whose norm is in the hundreds (the same mistake Lab 7 calls
   out). Clamping at multiples of the feature's *observed peak activation*
   gives a physically meaningful, model-transferable dose with a narrow window:
   ~1× peak induces the concept, ~3× collapses generation into repetition.
2. **Clamp the cleanest *concept* feature, not the highest-AUC one.** The lone
   "survived" feature on Olmo fires on 85% of tokens — a broadly-active basis
   vector, not a concept handle; clamping it just degrades fluency. Selection
   now maximizes `held_out_auc − fire_fraction`, which picks the low-frequency
   concept feature (id 1265, "law", fires 1.18%). Clamped at 1× peak it makes
   neutral prompts generate *"The court has ruled that the defendant is not
   guilty of the crime"* — **0 → 5 domain-keyword hits, random control 0**, with
   the fluency proxy flagging the collapse past the window.

## Validation evidence (Tier B, Olmo-3-1025-7B, SAE layer 16, d_sae 65536)

| Result | Value |
|---|---|
| toy superposition: represented (dense → sparse) | 5/20 → 17/20 in 5 dims |
| SAE reconstruction FVU / per-token L0 | 0.374 / 113.5 |
| features silent on corpus | 65.7% |
| peak-vs-frequency ranking overlap (top 15) | 0 |
| atlas verdicts | 1 survived, 3 narrowed, 6 polysemantic, 16 killed* |
| transcoder FVU / mean splice-in KL | 0.457 / 0.012 |
| Lab 4 truth-direction best cosine | 0.067 (a finding, not a bug) |
| feature clamp (id 1265 "law") | 0 → 5 hits at 1× peak; random 0 → CAUSAL |

\*`n_killed` counts killed + token-feature + polysemantic + silent in the
metrics rollup; the atlas table breaks them out separately.

**A live teaching case:** feature 1854 ("code") has top contexts that are 100%
code (label purity 1.0) yet held-out AUC only **0.57** → *killed*. Labeling
from top contexts alone, without negative validation prompts, would have
declared a clean win. That is the lab's central lesson, caught by its own
battery.

Tier A (gpt2) runs end-to-end on CPU in ~75s: toy collapse reproduces, SAE FVU
0.0019, transcoder FVU 0.46, atlas spans the verdict range (incl. a
token-feature and killed labels). The clamp is honestly `causal=False` on gpt2
— a weak base model whose narrowed features do not steer cleanly — and the lab
emits no CAUSAL claim in that case.

## Files

- `labs/lab08_sae_transcoders.py`, `labs/lab08_sae_transcoders.md`
- `data/sae_feature_corpus.csv`, `data/make_sae_corpus.py`
- registry entry `lab8` in `interp_bench.py`
- canonical runs: `runs/lab08_tiera`, `runs/lab08_tierb_full`
