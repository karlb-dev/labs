# Gemma handout — update plan (P8, §11.5)

Target: `interpretability/jspace_paper/gemma4_nonlinear_jacobian_handout.tex`
(graduate course handout; remains educational; Paper B cites it).

Edits applied at P8:

1. **§ terminal classification (line ≈222):** replace the standing
   "methods blocker" terminal sentence with the two-study sequence:
   Study 1 correctly failed its precommitted 1e-5 all-slot gate
   (0.002458111383020878) and published a blocker; Study 2 froze a
   target-blind pooled mixed-precision ceiling (0.07870368901355948 =
   max(3·q99, ten-quantum q99), route `benign_scheduling_floor`),
   under which the preserved discrepancy lies and the selected slot is
   bit-identical; the five-layer `local_tangent_mismatch` classifier
   (L22/L30/L37/L44/L52) is a **closed exact-JVP finite-scale methods
   result** at tested scope.
2. **§ present-tense statement (line ≈788):** keep the dated 2026-08-02
   sentence as chronology; append the dated Study-2 resolution
   sentence.
3. Preserve throughout: the blocker chronology (never retro-passed);
   the no-nondifferentiability and no-workspace cautions; the
   mechanism-not-identified boundary; scoped-empirical vs
   general-teaching separation.
4. Add the OLMo H6 result as a checkpoint-specific comparison (one
   licensed late-anchor cell; in-band fail) with evidence ids.
5. Update version/date block and evidence-id list
   (`gm2-backend-parity-calibration-v1`, `gm2-stage1-relicense-v1`).
6. Rebuild the PDF deterministically; register byte identity in the
   analysis mirror.
