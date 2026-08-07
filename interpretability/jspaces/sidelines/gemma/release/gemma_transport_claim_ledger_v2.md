# Gemma transport claim ledger V2

All entries are development or methods tier. V2 preserves the immutable
Study-1 blocker event while recording the separately preregistered Study-2
calibration and relicense decision.

| ID | Status | Scoped statement | Governing evidence |
|---|---|---|---|
| GM2-C01 | licensed methods | The G2.1 threshold derivation was target-isolated: it did not read or join the Study-1 outcome, and the ceiling was frozen before the registry was opened. | `gm2-foundation-v1`, `gm2-backend-parity-calibration-v1` |
| GM2-C02 | licensed methods | All 216 full exact-backend pairs are finite with primal parity; 232 pair reconstructions, 1,008 dtype-quantum reconstructions, deterministic replay, and row-order audit pass. | `gm2-backend-parity-calibration-v1` |
| GM2-C03 | licensed methods | The target-blind route is `benign_scheduling_floor`; no path-ambiguity, batch-nuisance, or architecture-dependent route is active under the frozen routers. | `gm2-backend-parity-calibration-v1` |
| GM2-C04 | licensed methods | The pooled all-frozen-batches exact-backend relative-error ceiling is `0.07870368901355948` for Gemma and the exact OLMo control. | `gm2-backend-parity-calibration-v1` |
| GM2-C05 | licensed methods | The preserved Study-1 all-slot discrepancy `0.0024581113830208778` lies within the calibrated ceiling while the selected slot remains bit-identical. | `gm2-stage1-relicense-v1` |
| GM2-C06 | licensed methods | The unchanged L22/L30/L37/L44/L52 `local_tangent_mismatch` classifier is a closed exact-JVP finite-scale result over the tested prompts, directions, target map, and doses. | `gm-jvp-gemma-stage1-v1`, `gm2-stage1-relicense-v1` |
| GM2-C07 | licensed methods | G2.2 selected `branch_1_relicense_without_recompute`; no G2.2 model response was computed. | `gm2-stage1-relicense-v1` |
| GM2-C08 | licensed methods | The OLMo H6 lane may import the exact pooled ceiling and its threshold hash as a methods calibration dependency. | `gm2-stage1-relicense-v1` |
| GM2-C09 | licensed historical | Study 1 correctly stopped under its own frozen `1e-5` gate; that failed event remains immutable and is not retroactively passed. | `gm-jvp-gemma-backend-parity-v1`, `gm-state-of-record-v1` |
| GM2-C10 | blocked | Gemma is nondifferentiable or has no Jacobian at the tested layers. | A finite-scale predictor mismatch is not failure of mathematical differentiation. |
| GM2-C11 | blocked | The mismatch identifies curvature, routing, normalization, attention, gated-MLP, or context heterogeneity as its cause. | Study 2 calibrates and licenses the estimator result; it does not run mechanism discriminators. |
| GM2-C12 | blocked | Information or a global workspace is absent, or a late layer is invalid for all intervention. | Transport accuracy, representation existence, and workspace claims are distinct. |
| GM2-C13 | blocked | The two exact backends are bit-identical over every calibration row. | They agree within the calibrated envelope; only the selected historical scientific slot is asserted bit-identical. |
| GM2-C14 | blocked | The result upgrades a Phase-4 confirmatory or replication conclusion. | Native tier remains development/methods. |
| GM2-C15 | blocked | This track supplies independent review or PI sign-off. | Neither role is exercised. |

## Maximum licensed wording

> Under the prospectively calibrated pooled all-frozen-batches backend
> envelope, the registered study-1 all-slot disagreement
> (0.0024581113830208778; predeclared rounded value 0.002458) lies within the
> frozen ceiling (0.07870368901355948), while the selected scientific slot
> remains bit-identical. The unchanged five-layer local_tangent_mismatch
> classifier is therefore licensed as a closed exact-JVP finite-scale methods
> result on the tested prompts, layers, directions, target map, and doses; it
> is not a claim of nondifferentiability, missing information, or workspace
> absence.

## Universal ceiling

> At the tested prompts, layers, directions, and intervention-relevant finite
> scales, the prompt-specific first-order tangent of the chosen
> source-to-target residual map predicts Gemma's finite response substantially
> less accurately than the same estimator predicts the OLMo control; the
> mismatch changes character with depth.
