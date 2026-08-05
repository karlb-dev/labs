# TRANSPORT_APPLICABILITY_SYNTHESIS.md — A5 output

Inputs: `tables/gemma_stage1_layer_table.csv` (byte-identical
reconstruction), the registered H6 joint row table
(`ol2-transport-validation-joint-v1`, 672 rows, verified), and
`tables/transport_applicability.csv` (61 tidy rows). Figure:
`figures/transport_applicability_map.{png,pdf}`. Companion:
`A5_H6_PHASE3_RECONCILIATION.md` (the estimand boundary). Every
statement names map (frozen source→final-residual), prompts, layers,
direction families, dose, and gate; "linear/nonlinear" never appears
unqualified.

## The map in three sentences

1. **Gemma-4-31B** (L22/L30/L37/L44/L52, ε=0.10 declared dose, frozen
   prompts/directions): 0/12 rows pass at every layer; bootstrap median
   tangent relative error runs 1.357 → 1.573 → 3.525 → 5.365 → 4.386
   with depth — 17.2–68.2× the frozen calibrated backend ceiling
   (0.0787), 51.7–204.5× the pooled backend q99 (0.0262), and
   12.2–48.3× the OLMo control's late-anchor 0.111. The mismatch is **directional, not a scale
   error** (declared-dose median tangent cosine 0.0389 at L22, −0.0622
   at L37, gains ≈ 1–5.5). Closed finite-scale methods result; cause
   not localized; no nondifferentiability/workspace claim.
2. **OLMo-3 Base** (L24/32/40/56 × ε 0.001–0.10): zero licensed cells.
   Most cells have no decision-eligible rows (below measurability —
   a data state, not a fail); where eligibility exists the floor fails;
   L56@ε=0.10 reaches 9/12 = 0.75 < 0.90.
3. **OLMo-3.1 Think**: exactly **one licensed cell in the entire
   campaign grid** — L56 at ε=0.10, 12/12. No in-band (L24/L32/L40)
   cell is licensed at any tested dose on either checkpoint. All 672
   backend rows sit under the imported ceiling (max 0.0222/0.0253), so
   the failures are model-behavior, not backend noise.

## Key comparisons (plan §8.5)

- **Late anchors vs in-band:** the only licensed transport surface is a
  late-layer, largest-dose anchor on one checkpoint. The causal assay
  band (L24–L40) licenses nothing on the tested ladder.
- **Base vs Think:** the late anchor is checkpoint-specific (12/12 vs
  9/12) — transport validity itself is training-state-dependent.
- **Gemma vs OLMo control:** same estimator, same target map — Gemma's
  mismatch is an order of magnitude above the OLMo control's late
  anchor; the five-layer classifier is a model property at tested
  scope, not an estimator artifact.
- **Backend disagreement vs scientific mismatch:** the historical
  blocker-era all-slot disagreement (0.00246) is 32× below the
  calibrated ceiling; every scientific mismatch is 17–68× above it.
  The calibration rescued the *measurement*, not the *premise*.
- **Actual-dose coverage:** the mapping from causal-assay doses onto
  this ε ladder is `archive_unavailable` (no exact site-dose records);
  neither "the causal doses were in-band" nor "out-of-band" may be
  asserted (Phase 5D is the prospective fix).

## Consequences for the papers

- Paper B gets the applicability map as a core figure: transport
  licensing is model-, checkpoint-, layer-, and dose-specific, and a
  preflight that skips any of the four axes is unlicensed.
- Paper A cites the map only through the reconciliation: the paired
  ablation results are not bounded by these gates; using the lens as a
  finite-dose *predictor* on OLMo in-band is.
- The Qwen row of this map is **empty by design** — the transport gate
  was never run on Qwen. That absence is a stated limitation, not a
  pass.
