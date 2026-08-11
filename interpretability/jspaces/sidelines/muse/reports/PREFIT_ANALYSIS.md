# Muse Glimmer — pre-fit analysis (development)

Generated while the 120-prompt J-lens fit is running on GPU.

## Instrument geometry: GO

| hard gate | result |
|---|---|
| shape 52×6656 | PASS |
| CUDA load (~59.6 GB BF16) | PASS |
| finite residual hooks | PASS |
| identity-J readout parity | PASS (max abs diff 0.0) |
| smoke top-1 sensible | PASS (`Paris` on capital-of-France) |

**Unlike Gemma**, residual-stack readout geometry is instrument-valid after the
Muse-specific `output_multiplier` patch. This is a green light for fitting a
J-lens and running the battery.

## Muse-specific unembed

```
logits = softcap * tanh( (W_U @ RMSNorm(h) * output_multiplier) / softcap )
output_multiplier ≈ 0.19611613513818404
softcap = 20.0
```

Upstream jlens only applied softcap. Without the multiplier, pre-softcap
logits were ~5× too large, softcap saturated, and top-k collapsed to
punctuation. Patched in `jspace_muse.adapters._patch_muse_unembed`;
intervention unembedding rows also fold the multiplier.

## Soft canary: local linearity

Finite-difference odd-symmetry at mid-layer (L25) fails at both ε=1e-2 and
ε=1e-3 (`odd_symmetry_rel_err ≈ 1.88`). Responses are large and asymmetric.
Interpretation:

- **Not** a readout-parity failure (identity J works).
- Likely gated / local-global attention + large residual norms make naive
  secants poor (canary, not Gemma-class transport block).
- Expect in-band Jacobians to be only *locally* linear; H7-style
  linearity ceilings may be tight. Record and revisit after fit.

## Boot sentinel (logit lens only, pre-fit)

Prompt: `Fact: The currency used in the country shaped like a boot is`

| layer | top-1 | top-5 |
|---|---|---|
| 0 | /are | '/are', ' bagaimana', ' þ', 'claimer', ' how' |
| 4 | てなブックマーク | 'てなブックマーク', ' ਦ', ' Fridays', '/w', 'ಾಡ' |
| 8 |  ISBN | ' ISBN', 'faktor', ' 모', ',.', ' Philippe' |
| 12 |  God's | " God's", 'icu', ' Moines', '/are', '348' |
| 16 |  nations | ' nations', ' membutuhkan', ' многое', ' not', ' nô' |
| 20 |  KG | ' KG', ' autop', ' only', 'ute', 'อร์' |
| 24 |  celle | ' celle', 'ント', ' отопления', ' svolgere', ' Orlando' |
| 28 |  actually | ' actually', '/are', ' moneda', ' called', 'currency' |
| 32 |  called | ' called', ' actually', ' named', '/are', '…' |
| 36 |  called | ' called', ' currency', ' Italy', 'called', ' named' |
| 40 |  Italian | ' Italian', ' euro', ' lira', ' currency', ' euros' |
| 44 |  euro | ' euro', ' euros', ' lira', ' Euros', ' Euro' |
| 48 |  called | ' called', ' euros', ' Euros', ' euro', ' Euro' |
| 51 |  the | ' the', ' called', ' not', ' lira', ' a' |

**Reading:** content about Italy/euro emerges around L36–L44 under pure logit
lens, then final-layer next-token mass shifts toward function words (`the`,
`called`). This is the classic depth profile where a J-lens *might* recover
earlier band content if transport is faithful — exactly what the post-fit
depth_profile cell tests.

## Fit in progress

- Recipe: n=120 WikiText, 21 source layers → L51, max_seq=128, skip=16
- dim_batch: timing ladder starting at 16 (d_model=6656 is heavier than OLMo 5120)
- Outputs: `lens/muse_glimmer_lens.pt`, `metrics/fit.json`

## Next

1. Finish fit → post-fit parity + g-fold
2. Battery: depth profile, selectivity, modulation, dual-task, capacity,
   ignition, verbal-report, protected ablation
3. Decide: is there a workspace-like signal worth a larger prereg?

Tier: development/methods only.
