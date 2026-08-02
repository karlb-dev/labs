# LIVE — Gemma transport workstream

Last updated: 2026-08-02 02:52 UTC. This is the Git-tracked mirror of the
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
- Remote Gemma branch: synchronized through the frozen-threshold producer;
  the positive-control registry/firewall transition is not yet committed.
- Dedicated Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_20260802`.
- GPU hard gate: PASS on NVIDIA RTX PRO 6000 Blackwell Server Edition,
  driver 580.82.07, PyTorch 2.11.0+cu128, CUDA build 12.8, capability 12.0.
- Pinned `jacobian-lens` revision `581d398613e5602a5af361e1c34d3a92ea82ba8e`
  is installed editable and byte-identical to the campaign Drive copy.
- Hugging Face authentication: PASS (`kburtram`).
- Shared Part-2 conformance bootstrap: PASS.
- No Gemma or OLMo model producer is running. The OLMo calibration and passing
  positive control are registered; no Gemma target has been opened.
- The isolated package scaffold is committed. Its 35-test conformance suite
  passes, including analytic, nonlinear tiny-transformer, and explicit Hugging
  Face Gemma/OLMo suffix/JVP tests plus a batch-shape/slot adversarial test.
- OLMo local staging completed from clean pushed commit `42b34b1`. The
  dedicated local snapshot verifies all 26 remote files, all 14 weight shards,
  64,476,964,249 bytes, and zero failures. Its Drive manifest file SHA-256 is
  `fa620d543b8cc80d3545aee177343036b04a5a0de84b8e65c8dcfa15dec1776c`.
  The historical Drive cache remains only 11/14 complete and must never be
  treated as load-ready by itself.
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
    changing a numeric threshold. Gemma remains unstaged and unopened.

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

1. Commit/pull/test/push this positive-control registry/firewall/report
   boundary.
2. Confirm every OLMo artifact remains durable, remove only the scoped local
   cache `/content/hf_olmo_control/`, and stage/hash the pinned Gemma snapshot
   on local NVMe.
3. Commit/publish the target staging/runner boundary, then run one Stage-1
   infrastructure cell and the unchanged full Stage-1 grid from a clean
   pushed commit.
4. Route to G2/G3/G5/G4/G6 by the observed G1 branch, preserving G9 time.

## Recovery checks and next commands

There is no live producer or GPU allocation. The 56-cell calibration and
positive control are finalized and registered; do not rerun, delete, or
rewrite them. Gemma execution is licensed, but its weights are not staged and
no target number has been opened. The local OLMo snapshot remains at
`/content/hf_olmo_control/` until its deliberate scoped removal; all evidence
and reports are durable in Drive. Inspect:

```bash
git -C /content/labs status --short --branch
git -C /content/labs log -8 --oneline --decorate
nvidia-smi
find /content/drive/MyDrive/interpret/special-lab-1/gemma_transport_20260802 \
  -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %p\n' | sort
ps -eo pid,etimes,cmd | rg -i 'jspace_gemma|gm_exact|gemma-4|Olmo-3-32B'
find /content/hf_olmo_control -maxdepth 4 -type f -o -type l 2>/dev/null | sort
cat /content/gemma_transport_work/locks/gm_exact_transport_gate.lock 2>/dev/null
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
