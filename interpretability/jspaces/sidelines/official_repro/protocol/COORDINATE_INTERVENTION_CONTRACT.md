# Coordinate intervention contract (frozen pre-data)

Addendum §2.2 folded in. Implements the paper's §2.5 two-coordinate
pseudoinverse patch verbatim; hook-point identity per addendum §3.2.

## 1. Three distinct objects (addendum §2.2)

**(a) Readout parity — the hard stop.** An independent recomputation of
`W_U · final_norm(J_ℓ h)` (plus `final_logit_softcapping` if present;
expected `None` on both lanes, recorded as fact) on actual activations
must match `lens.apply()` within fp tolerance (`rtol=1e-4, atol=1e-4`
fp32 on the same device path; exact top-20 order agreement additionally
required). This tests plumbing and is exactly achievable. Failure = stop;
no transposition chosen by whichever gives sensible words.

**(b) Intervention vectors.** `v_t` := row *t* of `W_U J_ℓ` — fixed,
norm-free, paper §2.1. These feed the swap and steering algebra. Computed
as `J_ℓᵀ (W_U[t] ⊙ g-fold-policy)` — see §5 for the g-folding audit; the
paper-literal default is **unfolded** `u_tᵀ J_ℓ`.

**(c) Probe form.** Fixed-row inner products `⟨v_t, h⟩` match the full
readout only up to a data-dependent per-position normalization (paper
§2.5, because `final_norm` rescales per position). Probe-vs-readout
conformance therefore uses exact top-k **order** agreement and rank
correlation, never raw-logit tolerance.

## 2. Coordinate swap (paper §2.5; plan §6.5)

For source token s, target token t at layer ℓ, position p, residual h:

```
V = [v_s, v_t]              # [d_model, 2] source-space columns
c = pinv(V) @ h             # 2 coordinates
c_swap = [c[1], c[0]]
h_patched = h + alpha * V @ (c_swap - c)
```

Primary `alpha = 1`. `alpha = 2` runs for **every eligible
flexible-generalization off-diagonal cell** as the frozen sensitivity
(paper-sourced: 76/192 → 101/192 on Sonnet 4.5), never only for failed
α=1 items. Orthogonal complement unchanged by construction; verified
numerically per trial.

## 3. Per-trial logs

layer, position, source/target token IDs + strings, vector norms, cosine,
singular values, condition number, original + swapped coordinates, patch
norm, residual norm, coordinate reconstruction error,
orthogonal-complement preservation error, output logits/ranks before and
after.

## 4. Conformance tests (all must pass before science)

- `s == t` → numerical no-op; `alpha == 0` → numerical no-op.
- Double swap recovers original coordinates.
- Non-orthogonal pairs swap correctly (verified against a dense
  least-squares reference).
- Near-singular pairs (condition number > 1e6 or cosine > 0.999) are
  `GEOMETRY_GATED`, never padded or regularized.
- Batch/item order invariance.
- Hook cleanup after success and exceptions.
- Directional check (addendum §3.2): a swap at ℓ changes the recorded
  activation at ℓ and downstream, and provably not at ℓ−1.
- Hook fires exactly at the frozen (layer, position) plan (§15 stop
  rule); fire counters prove batching never smears interventions.

## 5. RMSNorm gain folding audit (free CPU audit, addendum §2.2)

Before any intervention: cosine between `u_tᵀ J_ℓ` and
`(g ⊙ u_t)ᵀ J_ℓ` over every battery target token × band layer. If
min ≥ 0.99, register g-folding immaterial and proceed with paper-literal
unfolded `v_t`; else open a SPEC_DIVERGENCE_LOG decision before any
intervention runs.

## 6. Steering (verbal-introspection; plan §6.6)

Per released README: unit-normalized `v_t` (transpose row for the
surface token), scaled by the layer's mean residual norm × strength
scalar, added at every primary-band layer and every token of the user
question turn; strength 0 = paired control. Exact paper strengths are not
recoverable from the release → symmetric reconstructed ladder
`{0, 1, 2, 4, 8, 16}` frozen here (reported as reconstructed, not
official). Layer mean residual norms come from the model's own forward on
the rendered prompt (positions in the steered turn), recorded per layer.

## 7. Hook-point identity (addendum §3.2)

Interventions patch the same residual object `ActivationRecorder` reads:
the recorded output hidden state of block ℓ (post-block residual), edited
in place via a forward hook that returns the patched output. One
implementation serves readout, swap, and steering.

## 8. Intervention span

"Every prompt position" includes position 0 (and BOS where present) —
the released instruction is unqualified. Near-singular geometry at
individual positions gates the *position*, recorded per trial, only when
the pair is globally valid otherwise.
