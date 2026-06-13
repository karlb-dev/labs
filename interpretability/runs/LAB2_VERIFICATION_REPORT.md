# Lab 2 verification report — direct logit attribution

Date: 2026-06-10 · Machine: Colab A100-SXM4-80GB · Branch: `lab1_colab`

## What was built

Lab 2 (Direct Logit Attribution and Component Accounting) per COURSE.md §Lab 2 and
how_to_design_labs.md, implemented on the Lab 1 bench:

- **Bench extensions** (`interp_bench.py`): verified component capture
  (`resolve_component_anatomy` probes both raw-submodule and post-norm hook points and
  keeps the pair that reconstructs every block's residual delta — no name heuristics),
  `run_with_component_cache`, decomposition self-check (components must sum to the final
  pre-norm stream), direct-path component ablation, `lab2` profile, `--ablate-top` /
  `--dla-tolerance` flags.
- **Lab module** (`labs/lab02_direct_logit_attribution.py`): 16 verified single-token
  answer-pair prompts in 4 families (fact / relation / grammar / conflict; one example
  deliberately fails the tokenization gate as a teaching artifact), frozen-norm DLA
  linearization written out and commented in the lab (course convention §1.11 — handles
  both RMSNorm and LayerNorm incl. mean-subtraction and bias/constant rows), cumulative
  ledger curves, attribution-vs-ablation comparison with Spearman rho, DLA-vs-lens
  showcase overlay, category summaries, ATTR/CAUSAL claims with falsifiers.
- **Handout** (`labs/lab02_direct_logit_attribution.md`): ledger-not-causal-map framing,
  artifact reading path, writeup questions, symptom-first debugging table.

### Deliberate deviation from COURSE.md

COURSE.md describes Lab 2 via TransformerLens/TransformerBridge. The README's documented
design decision (raw HF + explicit hooks for Labs 1–2) is kept instead; the frozen-norm
handling that TransformerBridge makes implicit is taught explicitly in lab code.

## Validation evidence

| Check | Tier A (gpt2, fp32) | Tier B (Olmo-3-7B, bf16, A100) |
|---|---|---|
| Hook parity | OK — max diff 0, 12/12 | OK — max diff 0, 32/32 |
| Component anatomy probe | selected attn=module, mlp=module; recon err 1.6e-7 | selected attn=post_norm, mlp=post_norm; recon err 9.7e-3 |
| Lens self-check | OK, top-1 exact | OK, top-1 exact, max diff 0.0000 |
| Decomposition check | OK, rel err 7.9e-8 | OK, rel err 5.0e-3 |
| Ledger balance (worst) | 2.0e-7 logits | 0.025 logits (bf16 accumulation, reported in summary §5) |
| Examples | 4/4 kept | 15/16 kept; `plural_mouse` dropped at gate **by design** |
| Spearman rho (attribution vs ablation) | 0.47 (n=19) | 0.85 (n=74) |

The anatomy probe result is itself the strongest validation: it selected **different hook
points per architecture** (GPT-2 pre-norm adds raw submodule outputs; Olmo-3 post-norm
adds normed outputs) from evidence, not configuration — guessing wrong silently would
have produced plausible-looking nonsense scores.

## Headline science (Tier B, Olmo-3-7B)

| category | n | mean logit diff | attn total | mlp total | modal top component |
|---|---:|---:|---:|---:|---|
| fact | 5 | +6.34 | 3.59 | 2.73 | attn @ ~26 |
| relation | 4 | +7.29 | 6.48 | 0.83 | attn @ ~23 |
| grammar | 3 | +3.26 | 0.53 | 2.75 | attn @ 30 |
| conflict | 3 | −2.66 | −1.69 | −0.94 | attn @ 25 |

Late attention (layers 23–26) dominates fact/relation answers; grammar leans on MLPs;
conflict prompts show the same late-attention components actively pushing the stored
fact against the in-context answer. Rank inversions between attribution and ablation
exist in both runs (the lab's intended teaching payload) — e.g. Tier A's low-attribution
control `mlp@2` (score 0.01) had 4× the causal effect of a top-3 component.

## Runs included

- `lab02_direct_logit_attribution-20260610_234822-05163b/` — Tier A smoke (gpt2)
- `lab02_tierb_full/` — Tier B science run (Olmo-3-7B bf16, full set, top-10, ablations)

Audit path per run: `run_summary.md` → `diagnostics/component_anatomy.json` →
`plots/` → `tables/ablation_results.csv` → `logs/console.log`.
