# Paper B outline (P8)

Title family: *Operator Convergence Is Not Instrument Invariance:
Validity Boundaries for Jacobian-Lens Causal Analysis*.
Thesis: an averaged Jacobian operator can converge strongly and stably
in its aggregates while its sparse selections, mechanism endpoints,
finite-dose tangent predictions, and backward semantics under
version-pinning all fail invariance — so J-lens causal work needs a
per-checkpoint, per-dose, per-backend preflight.

1. **Introduction.** An averaged operator is not automatically a causal
   instrument; the four-level hierarchy (Figure M1); relationship to
   the empirical companion (cites A; re-argues nothing).
2. **Formal decomposition of validity levels.** Operator similarity →
   task-row geometry → sparse selection identity → causal/mechanism
   endpoints; what license each level grants. [C6]
3. **The Qwen nested-fit design.** Draw-A n=120/250/500/1000; frozen
   gates; the published-lens comparator kept distinct from nested
   comparisons.
4. **Structural convergence results.** q50/q05 monotone and
   gate-passing; operator medians ≥0.9955; aggregates fit-stable
   (15/15). Figure M2 (top panels). [TB1]
5. **Sparse-selection and causal-endpoint instability.** Jaccard exactly
   7/13 at every boundary; projector under floor; bridge-rescue sign
   oscillation; Q-L4 forced by the rescue row. Figure M2 (bottom).
6. **Falsifiers.** Selection-margin strata (near-tie vs stable-core;
   stable-core fails too); prompt-323 influence (closest materiality
   ratio 3,856.8× under threshold — current runtime). Figure M3.
7. **Runtime identity (named section).** The incident, the prospective
   catch, the diagnostic boundary, the amendment; consequence for any
   fit-then-audit pipeline. [C7] Where the two registered prompt-323
   maxima co-occur, distinguish them explicitly: 181.826310 is the
   blocked runtime-control era value; 181.776618 is the amended
   influence recompute.
8. **Mixed-precision exact-backend calibration.** G2.1 target-blind
   design; ceiling = max(3·q99, ten-quantum q99) = 0.0787…; route
   benign_scheduling_floor; the relicense logic (Study 1's failed 1e-5
   gate preserved, never retro-passed). [TB3]
9. **Gemma finite-scale mismatch.** Five layers, 0/12 passes, 17–68×
   above the frozen calibrated backend ceiling, directional; closed at
   tested scope; cause not localized. Figure M4.
10. **The OLMo H6 boundary.** One licensed cell in the campaign grid;
    eligibility as a data state; the dose-archive schema gap; the
    estimand reconciliation (why none of this bounds the paired
    ablations). Figure M5. [C5; A5 reconciliation]
11. **The preflight and reporting standard.** Figure M6 + TB5: backend
    calibration → delivery/SNR → local tangent gate → fit invariance →
    sparse-selection invariance → endpoint invariance → intervention
    license; each gate names the campaign failure it would have caught.
12. **Limitations and implications for published J-lens work.** Scope
    of the fit ladder (one model, nested draws, no rate); Qwen
    transport untested; A8 reproducibility statement + registers.
