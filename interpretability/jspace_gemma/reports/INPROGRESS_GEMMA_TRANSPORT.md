# LIVE — Gemma transport workstream

Last updated: 2026-08-02 02:08 UTC. This is the Git-tracked mirror of the
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
- Remote Gemma branch: synchronized through the calibration repair commit;
  the new band-convention registry/report boundary is not yet committed.
- Dedicated Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_20260802`.
- GPU hard gate: PASS on NVIDIA RTX PRO 6000 Blackwell Server Edition,
  driver 580.82.07, PyTorch 2.11.0+cu128, CUDA build 12.8, capability 12.0.
- Pinned `jacobian-lens` revision `581d398613e5602a5af361e1c34d3a92ea82ba8e`
  is installed editable and byte-identical to the campaign Drive copy.
- Hugging Face authentication: PASS (`kburtram`).
- Shared Part-2 conformance bootstrap: PASS.
- No Gemma or OLMo model producer is running. No current-study model response
  and no scientific Gemma target outcome has been opened.
- The isolated package scaffold is committed. Its 30-test conformance suite
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

1. Commit the band registry/report boundary; fetch, pull/reconcile, test, and
   push.
2. From that clean pushed commit run:

   ```bash
   python -u -m jspace_gemma.experiments.gm_exact_transport_gate --max-cells 1
   ```

   If runtime, VRAM, parity, delivery, and checkpoint integrity pass, resume
   the same producer without `--max-cells` from the same code commit. This
   emits OLMo-only calibration, not Gemma thresholds or a target result.
3. Freeze numeric Gemma thresholds in a clean, pre-target config and commit.
4. Remove the local OLMo staging copy, download the pinned Gemma 4 31B IT
   snapshot to local NVMe, then run Gemma Stage 1 with the unchanged harness.
5. Route to G2/G3/G5/G4/G6 by the observed G1 branch, preserving G9 time.

## Recovery checks and next commands

There is currently no live producer, lock owner, or partial calibration
checkpoint. The OLMo local snapshot is complete on this VM, but local NVMe is
ephemeral. On a new VM, run the bootstrap in `RESUME_GEMMA_TRANSPORT.md`, then
rerun the resumable staging producer if the local snapshot is absent. The
foundation, golden, band, and staging manifests are durable in Drive. Inspect:

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
