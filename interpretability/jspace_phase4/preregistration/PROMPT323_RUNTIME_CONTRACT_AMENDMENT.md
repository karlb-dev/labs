# Prompt-323 runtime-contract amendment

**Prospective amendment recorded 2026-08-03T15:51:51Z, after the runtime-identity diagnostic and before any prompt-323 contribution, layer-influence metric, materiality verdict, or canonical-decision output exists.**

## Decision criticality

The registered functional event already fixes the provisional branch candidate at Q-L4. The frozen canonical-decision producer recomputes Q-L1--Q-L5 solely from the registered structural and functional gates, checks the selection-margin audit, and only then verifies that the prompt-323 influence label is one of the three allowed labels. The influence label cannot alter the Q-L branch or its action. Prompt 323 is retained under every label.

Consequently, reproducing the historical Jacobian norm is not critical to the Phase 4 branch decision. It is critical only to a stronger claim that the influence estimate reproduces the unavailable historical runtime. This block does not make that stronger claim.

## Precommitted current-runtime shape audit

The first prompt-323 Jacobian contribution computed by the amended producer is the sole estimator used for the paired A500/A1000 leave-one-out audit. A second sequential computation with the same loaded model is a repeatability diagnostic only and is discarded. The first contribution may be durably banked only when all of the following hold:

1. both computations have the frozen prompt identity, sequence length, valid-token count, source layers, target layer, and tensor shapes;
2. all contribution tensors in both computations are finite;
3. the maximum, over source layers, of the absolute difference between the two layer Frobenius norms divided by `sqrt(d_model)` is at most `0.5`; and
4. the exact current distribution-content inventories below match before the model is loaded.

The numeric value `0.5` is the already frozen runtime-control tolerance. Here it is a technical repeatability gate, not an effect-size threshold or SESOI. It is not widened. There is no averaging, selection between repeats, trimming, refitting, or outcome-contingent rerun. If the gate fails, the current-runtime contribution remains null and the historical runtime must be restored before another evidential attempt.

The historical prompt-323 value (`173.345`) and its original absolute tolerance (`0.5`) remain recorded as a non-gating reference. The current runtime's discrepancy from it must be reported prominently in the result and manuscript limitation. Passing this amendment supports only a **current-runtime sensitivity shape** claim, not historical-runtime reproducibility.

## Exact current distribution-content lock

Inventory algorithm: sort the files reported by `importlib.metadata` by relative path; exclude `.pyc` files and paths containing `__pycache__`; hash every remaining regular file; then hash records of `relative_path NUL size NUL file_sha256 newline` in sorted order.

| distribution | version | files | bytes | content inventory SHA-256 | METADATA SHA-256 | RECORD SHA-256 |
|---|---:|---:|---:|---|---|---|
| `fla-core` | `0.5.2` | 321 | 3,276,300 | `6f813dc26c1a3633b70817928cb3621a8e59cbd8db07e5a2eb8aece49268a95e` | `8e641db5f26a58c254dd7c6977984d380aa9e36b4b46a82882f75954ca50c6c4` | `60d4796dc76dfddc8519b5a94266d79726f02458cb23d3944508c9fa6a26cbee` |
| `flash-linear-attention` | `0.5.2` | 156 | 1,475,375 | `1e89767e6bc19ce9c19f462320244bf98e335949a6cc21f0f5f5c42b2800db52` | `5ba189db45ddd61415bba9e71469bf396cec9b739192619b2fb95ee9bd1d892e` | `794321c0ae78c65e6e5c30d549bfe3682ebd06677942c06afad00fa2970855a6` |
| `transformers` | `5.13.1` | 2,543 | 46,649,387 | `62efb166f345418d70c0a1f65ddb0d8be27bc6318132d56a823156daf162a467` | `c3c8d11e913b70d1c2b591fa2962978680ee93e9cd21e3ee4f529dabfae43c0b` | `01605125de6e7af54a6380494860dfd2971271392fca44438df50f8104b0839f` |
| `triton` | `3.6.0` | 359 | 669,051,279 | `d29f29567d0e04bdb494a68cacf29ea4360251463558b3e624fc72a440f3edcd` | `5c9071db9b057bec89cd6d7f76f61e5f28a5e8d3717ff6c85420cf8015ee0321` | `6096e92fb41f4ab973d3092b757bb83dccf256486b3af75ee8220cf2eab935eb` |
| `torch` | `2.11.0+cu128` | 11,846 | 1,625,894,923 | `61b5a10c1be66fc769844c7bc7875a88bee31a0c4862fa2b22cffca519e66d4f` | `de881a744ff66b27fffc338c1624660cf7c87d3019574ad71e3dda886b2a172b` | `5b22d723c53bcefc515c6f30f0892ff632cc4df48d9275d71e4e6de3a698cdd6` |

The historical wheel/build hashes and compiled Triton/FLA cache do not survive, so the table locks the amended runtime but does not reconstruct the historical one.

## Unchanged scientific contract

- Evidence ID remains the first, not-yet-created `p4-qwen-lens-influence-prompt323-dev-v1` event.
- Banks, model revision, endpoint definitions, fixed token/probe samples, A500/A1000 lens hashes, equal-weight algebra, three materiality thresholds, decision wording, and retention rule are unchanged.
- No new bank, model, endpoint, SESOI, fit size, or lens is introduced. There is no A2000.
- The previously archived attempts remain null diagnostics: no contribution, layer part, materiality result, or registry event from them may be reused.
- Any of the three frozen influence labels is accepted without changing Q-L4. Tightening to the historical runtime may be considered later only to upgrade the runtime-reproducibility claim.
