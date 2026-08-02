# LIVE — Gemma transport workstream

Last updated: 2026-08-02 04:12 UTC. This is the Git-tracked mirror of the
canonical Drive handoff at
`/content/drive/MyDrive/interpret/gemma_transport_inprogress.md`.

## Current boundary

- Worktree: `/content/labs`.
- Branch: `interp_jspace_gemma_transport`.
- Exact fork: `3b041735d8b842de46a9c0a474fccd0c44e0841a`.
- Tested scaffold commit: `b49af6d1c285a300be9007461b695d22b90d5930`.
- Restart-boundary commit: `11501b804690559fabe571e941a95018eefbe19b`.
- Diagnostic/repair commit: `5b486fed0e93a6b87139430b3ac65f225470bac8`.
- Foundation/import commit: `2bb7428c931323d5d377ad04bd7e19a17958c491`.
- Exact-JVP golden commit: `91ecba1df24f6726faa94034c78f41806d7f38bb`.
- OLMo control-runner commit: `42b34b13192a56f941ce9ad92a757384e153f05c`.
- Pre-model calibration repair commit:
  `3a599f75de8bcb67bfb696be0d6d7c07010bac79`.
- Band-convention registry/report commit:
  `8fbdb88bb4ae17ac96e2e71a63807c78cb6a3187`.
- Control-smoke handoff/compute commit:
  `06b2a3d2fbe42fd5f70abb121573b1e7a62b45ec`.
- Incident-audit producer commit:
  `a196c4fdf267944c1b5d9daa467aadcbd65b93ce`.
- Incident registry/report commit:
  `173d6c26b802eed3757e485aca238751479050a5`.
- Pure finalizer commit: `374f511ef1fffa265631b59865980e184466444e`.
- Finalized calibration registry/report commit:
  `8300ca1df2b50a2c838d79232eb05501b20943c6`.
- Frozen-threshold definition/producer commit:
  `7f6a36e19417b618cbdf3c81a6da6d9171a3739a`.
- Positive-control registry/firewall/report commit:
  `36a3e61e2b17bb196662b6e8bf1c2d9635ef262f`.
- Guarded Gemma target-staging commit:
  `4f00d43e0810b98dfe1c281d260548ac791a14ab`.
- Frozen Gemma Stage-1 execution/runner commit:
  `036e55233babcabacae061ab41d1410a35715aea`.
- Stage-1 registry/report commit:
  `a017e632c9889970e6fe2f0353f91fc2e52878bc`.
- Frozen actual-Gemma backend-parity producer/compute commit:
  `af21c2068508a28871f541c82b8dd1ff0f59916b`.
- Backend-parity blocker registry/report commit:
  `0e8acf3234ffa22ca79b99e3a3ea9de148951faf`.
- Frozen terminal release producer/compute commit:
  `b80004843a5bbe57536e4da18297f7c52cf201a3`.
- The terminal release is registered and independently verified; the remote
  branch is synchronized through its producer, and this registry/report
  boundary is the next publication.
- Dedicated Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_20260802`.
- GPU hard gate: PASS on NVIDIA RTX PRO 6000 Blackwell Server Edition,
  driver 580.82.07, PyTorch 2.11.0+cu128, CUDA build 12.8, capability 12.0.
- Pinned `jacobian-lens` revision `581d398613e5602a5af361e1c34d3a92ea82ba8e`
  is installed editable and byte-identical to the campaign Drive copy.
- Hugging Face authentication: PASS (`kburtram`).
- Shared Part-2 conformance bootstrap: PASS.
- No Gemma or OLMo model producer is running. OLMo calibration/control and the
  Gemma Stage-1 result are registered; the target model is no longer loaded.
- The isolated package scaffold is committed. Its 48-test conformance suite
  passes, including analytic, nonlinear tiny-transformer, and explicit Hugging
  Face Gemma/OLMo suffix/JVP tests plus a batch-shape/slot adversarial test.
- OLMo local staging completed from clean pushed commit `42b34b1`. The
  dedicated local snapshot verifies all 26 remote files, all 14 weight shards,
  64,476,964,249 bytes, and zero failures. Its Drive manifest file SHA-256 is
  `fa620d543b8cc80d3545aee177343036b04a5a0de84b8e65c8dcfa15dec1776c`.
  The historical Drive cache remains only 11/14 complete and must never be
  treated as load-ready by itself. The scoped local staging copy was removed
  after the complete calibration and positive-control outputs hash-verified.
- First foundation attempt at clean remote commit `11501b8` stopped before
  producing a foundation result: the governing TeX exists as a Git object at
  later shared commit `4ea7a9b`, not as a worktree file in the exact fork.
  `gm-foundation-diagnostic-v1` is registered with `scientific_result=false`.
  The producer now pins the source by commit/blob/SHA-256; the repair and
  diagnostic registry row are committed and pushed.
- `gm-foundation-v1` now passes from clean commit `5b486fe`. Its primary
  manifest is `manifests/gm_foundation_v1.json` in the dedicated Drive root,
  SHA-256 `df9e85cc2f3afcb05a4881db18c5a4e17cc83323906577a99b7fae6fdcc06304`.
  Nine historical imports are registered; 22 tests and all 15 live output
  checks pass. No model target result was produced.
- `gm-jvp-goldens-v1` passes from clean commit `2bb7428`. Both exact backends
  hit the analytic derivative at zero recorded error; forward/fallback/reverse
  tiny-transformer directional derivatives agree within `8.33e-17`, and
  central-secant error falls about 4x per epsilon halving. Drive artifact
  SHA-256: `ca02c95c022741970468ea91bd80327e4c2d95c46fda1be2f225d9a83f4f8234`.
- `gm-band-convention-v1` is registered from clean commit `3a599f7`, with no
  model opened. The primary paper's 38--92% reindexed band maps approximately
  to Gemma L23--L55; the artifact SHA-256 is
  `04e56c9bffc02d9a9d2580a1982f520c1069e87e0898fe52a80b911b210e5c8f`.
- The OLMo run completed all 56/56 atomic cells (1,568 rows and 28 bit-exact
  clean-parity checks) from clean pushed commit `06b2a3d`. All state-listed
  metric/raw hashes verify. Final summary serialization then stopped on a
  pandas `numpy.int64` source-layer value. No summary, Parquet, inventory, or
  calibration registry event exists. State SHA-256:
  `f696f28cecc44d3a3d925308dd10226f1f7fa84e09e6e63ff37913ea3960278c`;
  full-run log SHA-256:
  `28a6aecdff750821603e5355bf0776ff38bc069181ad83d6edf4677249225dfe`.
- `gm-olmo-calibration-finalize-diagnostic-v1` is registered from clean commit
  `a196c4f`. It reread finite raw tensors, provenance, and all 112 metric/raw
  files; manifest SHA-256:
  `78d53fca50b2a8ac2e114f71a7900a3581214e5367b0892dadf624ec736e8e25`.
- A pre-finalizer probe caught the valid mixed int/string `source_position`
  column before derived Parquet creation. The prepared finalizer retains the
  original per-cell JSON and stores canonical strings plus original runtime
  types; a 1,568-row Parquet write/read probe passes.
- `gm-jvp-olmo-calibration-v1` is registered with compute commit `06b2a3d`,
  finalizer `374f511`, and no recomputed cell or model load during finalization.
  It contains 56 cells/1,568 rows (645 faithfully delivered), 28 bit-exact
  suffix parity checks, and zero exact-JVP primal parity error. Single-position
  median tangent cosine is 0.693 at L4, 0.991 at L56, and 0.996 at L60.
  Summary SHA-256:
  `b0088651fa953d58939e4c509bae779ad2fbaeba92f1bde7d8d4722030ca98ef`;
  finalization manifest SHA-256:
  `b68aba140db58e7f5caa02821dc89b395be64e4bfeaa76b094fa7d53152d608d`.
- `gm-jvp-olmo-positive-control-v1` is registered from clean commit `7f6a36e`
  with all 14 frozen criteria passing and `target_model_opened: false`.
  Artifact SHA-256:
  `fc957c9a6f6f397cbaf3274193713ebec332bb81dbb99b61cf9f56d058cd1942`;
  byte-identical threshold-config SHA-256:
  `3cb1e68c548bce1dc350c8b60a52e5bc6594a4fadb7abec1c4e00f931d855630`;
  producer log SHA-256:
  `cf94f3308c6cf5239d9bcb24c2f2a27d584652439810b93517f7016894bb895e`.
  The target firewall is now open without a numeric change. Measurement SNR
  is 12; primary decision SNR is 20. The primary epsilon-0.10 gate is cosine
  >=0.98, forward relative error <=0.20, and central relative error <=0.10
  with >=90% row passage. All 32 L56/L60 anchors pass. The curvature intercept
  ceiling is 0.30 and positive slope floor is 0.15.
- The exact Gemma snapshot is fully staged from clean `4f00d43` at
  `/content/hf_gemma_target/.../snapshots/842da379...`. Producer and independent
  verification both pass all 12 files, both weight shards, LFS hashes, ordinary
  Git blob IDs, sizes, and the safetensor index. Manifest file SHA-256:
  `5b8d26a91b5cdc74e7fbc982d89bbf6d661233ee3da81705d165ef31cf6e308a`;
  payload SHA-256:
  `cfb98f55c3453319f19aedce51419445260b03355632d13c2402897ffbab4ec1`;
  staging log SHA-256:
  `f8fbe657d09c7d459b656c51335120428680d5300cb23fb7c96ea5355bf91072`.
  Staging itself loaded no model and created no target response.
- `gm-jvp-gemma-stage1-v1` is registered from clean pushed `036e552`. The
  frozen 40-cell/1,120-row non-lens core completed with 20 bit-exact clean
  suffix checks, zero exact-JVP primal error, 538 faithfully delivered rows,
  508 measurement-SNR rows, and 477 primary-SNR rows. All 80 metric/raw cell
  files pass an independent finite-tensor/provenance/hash audit, and aggregate
  recomputation from Parquet is exact. Primary smallest-secant pass counts are
  L22 0/12, L30 0/13, L37 0/12, L44 0/13, and L52 1/14; all five layers
  classify as `local_tangent_mismatch`. At epsilon 0.10, 12/16 primary rows
  are evaluable per layer and zero pass. This is an instrument-validity result,
  not evidence for nondifferentiability, missing information, or workspace
  absence. Actual-model forward/fallback JVP parity is the next adversarial
  check before mechanism interpretation. Summary SHA-256:
  `0f28372591bc1ece4472b103d74d645416b1ddba59a08ae0688c19fccb56e384`;
  row-table SHA-256:
  `3b74f1e983f47c1f917fd8c407a6ea1f8abf42854adc0b9d6c3d3cf18d921550`;
  state SHA-256:
  `5d902ae4b7b2dd6a5d2073ca1238041e321dcee537457a22ab6847d9c5d2df65`;
  full log SHA-256:
  `3b99f1991ec70345f7998755ecb744a0d61054181eac0940db33490cc3108c0c`.
- `gm-jvp-gemma-backend-parity-v1` ran from clean pushed `af21c20` and failed
  its frozen all-slot backend-relative-error gate. Both exact backends
  succeeded; their selected epsilon-0.05 tangent is bit-identical, and every
  source/target/secant/tangent/hash/metric replay check passes. Across all
  eight original batch slots, however, tangent cosine is 0.99999958 while
  relative error is 0.002458, above the frozen 1e-5 ceiling. The strict event
  is therefore a methods blocker even though the selected Stage-1 mismatch is
  reproduced exactly. Artifact SHA-256:
  `22c327764034f77496971f6c555af0ec6f8e99a0ceb80cea1677db24ca404b7c`;
  raw SHA-256:
  `ac5ba50dbba6d3ed149cf5b7b6951b80bee5502d11ac90e4f93dc45d515c9e89`;
  log SHA-256:
  `61c6704f6b34d60682123fbfa3936d770e307e48df6a3435c2f2b269c19350df`.
  The model is unloaded and G2/G3 mechanism interpretation is stopped.
- `gm-state-of-record-v1` is registered from clean pushed `b800048` with
  `COMPLETE_METHODS_BLOCKER`, `scientific_expansion=false`, and no target
  model load. It independently verifies the 18-event/35-output source prefix
  and registers eight release outputs. Import-bundle SHA-256:
  `005532754166644e42a369358565b9ce72235151e64559a9f12254d987ff7729`;
  canonical payload SHA-256:
  `694c62db534953accadb4f2223109fabf689f02a5117333497f716c17bed0320`;
  release-manifest SHA-256:
  `1f896c7029a2f4cee10378a95c71463d1346e2a500529101af80de7063c1e483`;
  release log SHA-256:
  `335f16b7a7e45e250806e42b7b060276fe15d40de300003175d8ffa8644be5fe`.
  The state of record, claim ledger, and protocol are byte-identical between
  Git and Drive.

## Completed this VM

1. Read the full shared static resume and current Phase 4.2 dynamic handoff.
2. Read all 1,522 lines of `jspace_lab_gemma_1.md` and its binding addendum.
3. Read the governing nonlinear-Jacobian handout TeX in full, the Phase-2
   report, prior Gemma/OLMo producers and result summaries, and Phase-4
   conclusion sentence 5.
4. Read the Phase-4.3 branch/artifact/import contract and both side-track
   addenda. Confirmed three-way isolation from mainline Qwen and OLMo lineage.
5. Created this branch from the exact required parent and created the dedicated
   Drive run root.
6. Verified 189 GB local disk free before model staging and 97.9 GB GPU VRAM.
7. Downloaded only the pinned Gemma metadata/config so far at revision
   `842da3794eaa0b77d5f08bae87a17459d91ff475`; no Gemma weight shards have
   been staged yet.
8. Implemented the isolated package/path/registry/provenance contracts,
   architecture manifest, explicit no-cache decoder suffix, exact autodiff
   backend ladder, fp32 delivery audit, transport metrics, robust curvature
   fit, frozen prompts/design, report/TeX skeleton, and foundation/golden
   producers. All 22 unit tests pass.
9. Audited remote model inventories. Gemma has two expected weight shards and
   neither is local; the OLMo Drive cache needs repair during local staging as
   described above.
10. Registered the failed foundation pre-result attempt and repaired the
    worktree-vs-Git-object assumption. The full 22-test suite and live-output
    verifier pass after the repair.
11. Registered `gm-foundation-v1` plus nine read-only historical imports,
    exact source blobs, remote model inventories, architecture/runtime locks,
    and the development/methods claim firewall.
12. Registered `gm-jvp-goldens-v1`; analytic and nonlinear tiny-transformer
    forward/fallback/reverse derivative parity passes, with no numerical
    secant admitted as an exact backend.
13. Implemented safe OLMo cache seeding/download plus full LFS/Git-blob/index
    verification, batched exact JVPs and finite responses, wrong-hook and
    clean-repeat baselines, atomic raw cells, resume headers, robust
    `a+b*epsilon` fits, and the OLMo-only calibration producer.
14. Corrected the pre-run OLMo layer grid to add L4 shallow control and L60
    identity anchor around matched L24/L32/L40/L47/L56. No current-study
    model number had been observed.
15. Fully staged and hash-verified the exact OLMo control snapshot locally.
16. Before model load, repaired the runner so finite clean baselines, repeats,
    perturbed passes, and exact JVPs use identical batch shapes/slots; every
    post-cast negative/double/sum vector receives its own direct JVP; nonlinear
    residual defects subtract first-order delivery mismatch; and SNR includes
    a target-dtype half-step floor. All 30 tests and the live-output verifier
    pass from the pushed repair commit.
17. Registered the primary-paper band convention as `gm-band-convention-v1`.
    The immutable source record resolves 38--92% depth to approximately Gemma
    L23--L55. No model was opened.
18. Ran the infrastructure smoke and full OLMo grid from `06b2a3d`; all 56
    cells are durable and hash-valid. The producer failed only at final JSON
    serialization, before aggregate outputs or evidence registration.
19. Implemented a strict incident auditor that binds state, input manifest,
    every metric/raw file, row/raw provenance, finite tensors, parity records,
    failure log, and the no-summary boundary before finalization repair.
20. Registered that incident audit, fixed the aggregate's pandas integer at
    the JSON boundary, and implemented a pure finalizer with lossless Parquet
    type normalization and separate compute/finalizer provenance.
21. Registered the complete OLMo calibration from the immutable cells. The
    expected shallow-to-late tangent-faithfulness gradient is present, but no
    target threshold or Gemma result has been opened.
22. Derived numeric gates only from the registered control/random baselines,
    added prompt-bootstrap and positive-control validation code, and committed
    the frozen definition before any Gemma number.
23. Reproduced all 14 control criteria from clean `7f6a36e`, registered the
    positive-control artifact, and enabled `gemma_execution_allowed` without
    changing a numeric threshold. Gemma remained unstaged and unopened at that
    boundary.
24. Published the guarded target stager, removed only the 61 GB ephemeral OLMo
    cache after durable-output verification, and staged/rehashed the exact
    62.58 GB Gemma snapshot. No target model or response was opened.
25. Froze a run-specific Stage-1 execution manifest for 40 cells/1,120 rows and
    a predeclared L52 smoke cell; implemented resumable raw checkpoints,
    immutable input headers, loaded-architecture audit, frozen decision rules,
    curvature partitioning, prompt bootstrap, and strict aggregate validation.
26. Ran/audited the predeclared L52 smoke, then resumed the same compatibility
    header and reused that cell by hash while completing the other 39 cells.
27. Registered the complete non-lens Stage-1 core and independently re-audited
    all 80 raw/metric files plus exact aggregate recomputation. Every frozen
    target layer returns local tangent mismatch; J-selected directions remain
    deferred until exact lens/token hashes are bound.
28. Froze an unrun one-cell backend-parity diagnostic on the registered
    `gm-p001-L52-single_position` / random-Rademacher / epsilon-0.05 row. It
    replays the original eight-request batch and selected offset, rehashes the
    target snapshot before load, and requires agreement between
    `torch.func.jvp` and `torch.autograd.functional.jvp` plus exact replay of
    the stored activation, target, secant, tangent, hashes, and metrics. No
    diagnostic output or registry event exists at this pre-run boundary.
29. Ran that diagnostic from clean pushed `af21c20`. Both exact backends and
    all selected-row replay checks pass, but the precommitted all-slot relative
    error is 0.002458 versus 1e-5. Registered the immutable failed-gate event,
    unloaded the model, and stopped G2/G3 rather than weakening the threshold
    after observing Gemma.
30. Prepared the model-free terminal release producer and frozen config. It
    verifies the 25,170-byte registry prefix (18 live events, 35 outputs),
    critical evidence hashes and blocker semantics, then emits the state of
    record, claim ledger, 20-minute gate protocol, inventory/environment lock,
    and Phase-4 methods-only import bundle. No release output or
    `gm-state-of-record-v1` event exists at this pre-run boundary.
31. Ran the release from clean pushed `b800048`, independently recomputed the
    bundle payload and registry-prefix hashes, rehashed every source and
    release output, and registered the terminal methods-only state of record.
    No model was opened and no scientific branch was resumed.

## Immutable scientific guardrails

- G1 exact autodiff JVP versus faithfully delivered secants is mandatory.
- The identical OLMo positive-control harness must pass in the same execution
  environment before any Gemma defect number is interpreted.
- G1 pass/fail thresholds must be written and committed after OLMo calibration
  but before the first Gemma target result.
- Finite differences are never called exact JVPs. If the autodiff backend
  ladder fails, register a methods blocker and stop that branch.
- Every result remains development/methods tier. No Phase-4 confirmatory or
  replication outcome is opened, and no independent-review/PI field is signed.
- Distinguish readout opacity, within-context finite curvature, and
  between-context tangent heterogeneity in every claim.
- Two 24-hour GPU blocks maximum; reserve approximately two hours of block 2
  for G9 release/state-of-record work.

## Immediate queue

1. Commit, fetch/integrate, test, and push the release registry/report
   boundary.
2. Reverify the final Gemma branch and merge it with ancestry preserved into
   the latest pulled `interp_jspace_part2`; keep G2/G3 stopped.

## Recovery checks and next commands

There is no live producer or GPU allocation. The backend diagnostic is
registered as a failed methods gate from `af21c20`; do not rerun it, weaken
its frozen criterion, delete it, or interpret Stage 1 mechanistically. The
terminal release is registered from `b800048`; do not rerun or overwrite its
eight outputs. The
56-cell calibration and positive control are finalized and registered; do not rerun, delete, or
rewrite them. Gemma Stage 1 is complete and registered from `036e552`; do not
rerun, delete, or rewrite its state/cells/aggregates. The target model is
unloaded, the fully verified local snapshot remains available, and the ephemeral OLMo
snapshot is removed. If this VM is reclaimed, Stage 1 needs no rerun; verify
the Drive outputs and continue from the latest pushed report boundary. Inspect:

```bash
git -C /content/labs status --short --branch
git -C /content/labs log -8 --oneline --decorate
nvidia-smi
find /content/drive/MyDrive/interpret/special-lab-1/gemma_transport_20260802 \
  -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %p\n' | sort
ps -eo pid,etimes,cmd | rg -i 'jspace_gemma|gm_exact|gemma-4|Olmo-3-32B'
find /content/hf_gemma_target -maxdepth 4 \( -type f -o -type l \) 2>/dev/null | sort
cat /content/gemma_transport_work/locks/gm_gemma_stage1.lock 2>/dev/null
cat /content/gemma_transport_work/locks/gm_gemma_backend_parity.lock 2>/dev/null
```

If a later handoff records a lock or checkpoint, that later section supersedes
the no-process statement above. Never launch a duplicate producer until the
process, lock inode holder, GPU process, log tail, and checkpoint header all
agree that the run is unowned.

## Push/merge rule

Every evidence boundary is: update this handoff and the detailed report,
commit, `git fetch`, integrate the remote Gemma branch, run tests, then push.
The final completed Gemma branch is merged with ancestry preserved into
`interp_jspace_part2`; no earlier partial merge.
