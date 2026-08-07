# Gemma transport claim ledger

All entries are development or methods tier. “Blocked” means the sentence is
not licensed by this track.

| ID | Status | Scoped statement | Governing evidence |
|---|---|---|---|
| GM-C01 | licensed | Both exact PyTorch JVP implementations pass analytic and deterministic nonlinear tiny-transformer goldens. | `gm-jvp-goldens-v1` |
| GM-C02 | licensed | The OLMo control shows the expected shallow-to-late increase in tangent faithfulness and passes all 14 frozen target-readiness criteria. | `gm-jvp-olmo-calibration-v1`, `gm-jvp-olmo-positive-control-v1` |
| GM-C03 | licensed descriptive only | On four prompts, non-lens directions, final-residual target, and the frozen delivery/SNR gates, the forward-mode Gemma Stage-1 classifier returns `local_tangent_mismatch` at L22/L30/L37/L44/L52. | `gm-jvp-gemma-stage1-v1` |
| GM-C04 | licensed methods blocker | The frozen L52 backend replay succeeds at both exact backends and is bit-identical at the selected slot, but the full eight-slot relative error is 0.002458, above the precommitted 1e-5 ceiling. | `gm-jvp-gemma-backend-parity-v1` |
| GM-C05 | licensed | G2/G3 and all downstream mechanism interpretation are stopped; the backend threshold is not relaxed after observing Gemma. | `gm-jvp-gemma-backend-parity-v1`, state of record |
| GM-C06 | blocked | Gemma is nondifferentiable at the tested layers. | Backend agreement is not established over the frozen full-batch gate. |
| GM-C07 | blocked | Stage-1 mismatch is caused by finite curvature, routing, normalization, MLP gating, or context heterogeneity. | Required G2--G5 discriminators were not licensed to run. |
| GM-C08 | blocked | Gemma lacks relevant information or a global workspace. | Transport validity and representation existence are distinct. |
| GM-C09 | blocked | The late L44--L52 band is licensed or rejected for causal J-space intervention. | G6 was not licensed after the G1 stop. |
| GM-C10 | blocked | This side track upgrades a Phase 4 confirmatory or replication conclusion. | The study is methods-only and opens no Phase 4 model cell. |
| GM-C11 | blocked | This work provides independent review or PI sign-off. | No such role is exercised by the side track. |

## Paper-facing sentence

The maximum licensed wording is:

> On Gemma 4 31B, an exact-JVP transport audit stopped at a precommitted
> cross-backend consistency gate: the selected replay agreed exactly, but the
> full matched batch exceeded the relative-error tolerance, so downstream
> mechanism and workspace interpretation were not licensed.

Phase 5 may quote or further narrow this methods sentence after verifying the
import bundle. It may not strengthen it without new, prospectively registered
evidence.
