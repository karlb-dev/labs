# QWEN_INSTRUMENT_SYNTHESIS.md — A3 output

Inputs: `tables/qwen_ladder_progression.csv` (50 verified rows) and
`tables/recon_qwen_ladder.csv` (22/22 reconstructed, 0 failed; prompt-323
row recomputed from the released 6.6 GB fp32 tensor with bit-identical
layer norms; all seven canonical routing-input hashes re-verified,
including the 3.3 GB A1000 lens). Figure:
`figures/qwen_multilevel_convergence.{png,pdf}` (regenerates from
`tables/qwen_multilevel_convergence.csv`). No convergence rate is fitted
— three registered comparisons are three points, not a curve.

## SQ5 answer: convergence splits cleanly by object level

| Level | Verdict across A120→A250→A500→A1000 |
|---|---|
| Averaged operator (task rows) | **Converges, monotonically**: q50 0.99522 → 0.99771 → 0.99870; q05 0.99308 → 0.99684 → 0.99812; every gate passes. Raw/minus-identity operator medians ≥ 0.9955 throughout. |
| Aggregate scientific endpoints | **Fit-stable at every boundary** (15/15 gates): max |Δ|: occupancy 0, centered excess 0.0711 pp, span-safe specific 0.0960 nat (gate 0.15), tail rate 0.0333 (gate 0.05), G4 flip 0. Bridge *preference* also stable (−0.055 / −0.002 / +0.016). |
| Sparse selection identity | **Never passes**: selected-ID Jaccard median is exactly 7/13 = 0.538462 at *all three* boundaries (a quantized median, not a trend) against a 0.75 floor; normalized projector overlap creeps 0.6748 → 0.7030 → 0.7098 against a 0.85 floor — direction favorable, deficit large. |
| Causal mechanism endpoint | **Oscillates**: bridge-rescue difference −0.2147 (PASS) → +0.5589 (FAIL, sign flip) → −0.2940 (FAIL). Per-lens equal-family rescue means run +0.251 (A120), +0.466 (A250), −0.129/−0.093 (A500), +0.165 (A1000). |

## The falsifiers do not rescue the instrument

- **Ties/margins**: the selection-margin audit retains 17,381/17,381
  positions (15,536 near-tie, 1,845 stable-core, 0 rank-deficient) —
  reconstructed label-by-label at 100% agreement. Near-ties are
  pervasive, but the frozen analysis shows exact ties do not explain the
  identity failures, and the stable-core stratum fails too.
- **Single-prompt influence**: prompt-323's materiality metrics are all
  negligible — the closest is 3,856.8× below its threshold (A500 task
  median 5.19e-6 vs 0.02); primary max norm 181.776618 recomputed from
  the raw tensor. Scope: **current runtime only** (C7): the fit-era
  backward semantics were not reproducible (181.83 vs fit-log 173.35;
  the prompt-112 control's 55.54/55.59 vs registered 160.07), and A500's
  rescue value itself differs across the two events that measured it
  (−0.093363 vs −0.129414) — within-event comparisons only.

## Mechanical route

Q-L4 reproduces from the frozen truth table; notably it is **forced by
the bridge-rescue gate row before the sparse-geometry rows are
consulted** — the mechanism endpoint, not the selection geometry, is the
binding failure. No canonical sparse Qwen lens is nominated; no Phase 4
confirmatory primary opened.

## Nested vs published comparator (required distinction)

Structural convergence rows compare *nested same-corpus draw-A fits*
(n→2n). Functional gate rows compare *each draw-A fit against the frozen
published lens* (the partially specified comparator). The two families
are never mixed in one panel or one claim; the figure's panels carry the
provenance in their source events.

## Registry note

The Drive mirror of the Phase 4 registry is a stale exact byte-prefix
(69/82 lines) of the repository registry; the repository file — whose
hash matches `FREEZE_HANDOFF.md` — is authoritative and the audit used
it. The A120–A250 deficit re-verified: 16/17 outputs resolve; only the
operational `state.json` (registered SHA `361bda08e9ffbe1d333fd3cf…`)
is absent, exactly as accepted.

## What Paper B may conclude (C6 wording, verified)

Averaged-operator convergence — even monotone, gate-passing convergence
with fit-stable aggregate causal endpoints — does not license sparse
instrument identity or mechanism-endpoint invariance. A preflight that
checks only operator/task-row similarity would have approved a lens
whose selected coordinates (Jaccard 0.538) and rescue endpoint (sign
oscillation) are not the same instrument across adjacent nested fits.
