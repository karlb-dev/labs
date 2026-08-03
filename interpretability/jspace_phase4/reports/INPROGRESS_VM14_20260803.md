# IN PROGRESS — Phase 4.4 decision block, VM14

Updated: 2026-08-03 05:01 UTC. Phase 4 remains development-only and is not frozen.

## Recoverable boundary

- Source parent: `901fb4fc7578a913088c7947a2e6240f7fc45aeb` from clean `interp_jspace_part2`.
- Working branch: `interp_jspace_phase4_4`; every scientific registry commit through `0ce3519b3abfc67ccfb3335e6e371b8a689a0fb5` is pushed.
- Terminal-B pre-commitment: pushed at `6f23a29896e61cc367ed884a2d840f0e08857f40`, before functional execution.
- M0 tests: 279/279 passed under the frozen package path before execution; after the tied-top-k instrumentation repair the expanded full suite passed 282/282.
- M0 A1000 tensor audit: pass; lens `6e48c773...f6bd6`, checkpoint `fd5a4ae...bf20`, header `b0cf4c8d...6d2a`, 63 exact finite layers.
- M0 Qwen snapshot/runtime: pass; exact revision `6a9e13bd...`, 23 files, 48 fused bindings. The model was downloaded directly from Hugging Face rather than copied from Drive.
- External published lens: exact hash `1718c8c...11e1`.
- M0 fresh-VM durability: 230/232 verified, with exactly two known historical deficits and zero unexpected failure.
- Fresh local registered-output backups cover the A1000 fit (3 outputs), A500–A1000 structural event (6), functional event (18), and selection-margin event (34).

## Completed sealed stages

Functional evidence `p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1` is registered, locally backed up, committed at `603bbcec271c8a5a0b7ce6f519a4845988e959b3`, and pushed. Its 18-output backup is 303,287,915 bytes with manifest SHA-256 `988844d3...35e7`. A500–A1000 selected-ID Jaccard is 0.538462 and normalized projector overlap is 0.709818, so both selection gates fail. Bridge-rescue difference is -0.294028 nat and also fails; occupancy, centered excess, span-safe specificity, tail rate, G4, and bridge preference pass. The event emits provisional branch candidate Q-L4 and explicitly nominates no canonical lens.

Selection-margin evidence `p4-qwen-selection-margin-a500-a1000-dev-v1` is registered, locally backed up, committed at `0ce3519b3abfc67ccfb3335e6e371b8a689a0fb5`, and pushed. Its 34-output backup is 14,411,038 bytes with manifest SHA-256 `12e76809...f451`. All 17,381 positions and all strata remain in the functional decision: 15,536 are `near_tie`, 1,845 `stable_core`, and zero rank-deficient. The audit exactly reconstructs the registered selection geometry and retains Q-L4 as the functional branch candidate. Its 52,261 behavior-blind lexical rows marked for manual review were not manually adjudicated because that sheet is nondecisional.

The registered figures `p4f26` and `p4f27` and their PDFs are durable on Drive and copied byte-for-byte into `reports/figures/`; both were visually checked. The living Markdown synthesis is updated. The compiled handout TeX/PDF remains at its governed pre-canonical boundary and must not be regenerated while the canonical event is absent.

## Prompt-323 runtime-identity hard stop

The next queue stage stopped before writing any contribution or layer result. The frozen prompt-323 norm is 173.345 with absolute tolerance 0.5; two clean current processes returned 181.826310 and 181.785516. Replaying prompt 322 first returned 59.545969 versus historical 52.150 and left prompt 323 at 181.854247.

The registered prompt-112 control proves this is a broader backward-semantics mismatch. Its fit log is 159.952 and its registered clean recompute is 160.070954, whereas two clean current processes returned 55.544060 and 55.587600. The current repeats agree within 0.044, all results are finite, and every maximum is at layer 0. Nominal GPU, driver, CUDA, Torch, Transformers, Triton, FLA, model, corpus, binding, and `jlens` identities match the historical manifests; historical distribution-content and compiled-kernel/cache identities were not preserved.

No influence event exists, no prompt contribution exists, no canonical event exists, and Q-L4 is not canonical. The two identical null states have `contribution: null`, zero completed layers, 1,146 bytes, and SHA-256 `4eef3124...239eb3`. They are preserved under distinct blocked Drive and local backup directories. Exact values, content inventories, hashes, paths, and the inference boundary are in `PHASE4_PART4_PROMPT323_RUNTIME_BLOCK.md` and `.json`; all scripts and logs are mirrored under Drive `diagnostics/prompt323_runtime_identity_20260803/`.

## No authorized next command

Do not rerun the queue blindly, widen the 0.5 tolerance, reuse either null state, synthesize a contribution, or open the canonical producer. Continuation requires independent scientific/governance review and one of two prospective resolutions:

1. reconstruct a historically content-pinned backward runtime, then rerun the unchanged influence stage from a fresh canonical output directory; or
2. approve a prospective runtime-contract amendment before a fresh attempt.

If and only if a fresh prompt-323 attempt passes its reviewed contract, bank, back up, commit, and push that influence event before opening the unchanged mechanical canonical producer. The queue lock is free, the canonical output directory is absent, no model process remains, and the GPU is released.

## Hard boundaries

- Queue order remains functional → selection margin → prompt-323 influence → canonical Q-L decision. The first two are complete; the third is blocked; the fourth has never opened.
- M2 side admission requires the registered canonical event and is not authorized. M3/M4 remain closed.
- P4-P1 stays estimation-only; P4-P3 stays blocked at 16/20.
- P4-P2 is the sole conditional primary and cannot run without a Q-L1/Q-L2 canonical event and actual independent producer review.
- Do not self-sign review or PI fields, do not open confirmatory/replication outcomes, do not fit A2000, and do not add a bank, model, endpoint, or SESOI change.
