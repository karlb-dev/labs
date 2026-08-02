# LIVE — Gemma transport workstream

Last updated: 2026-08-02 00:43 UTC. This is the Git-tracked mirror of the
canonical Drive handoff at
`/content/drive/MyDrive/interpret/gemma_transport_inprogress.md`.

## Current boundary

- Worktree: `/content/labs`.
- Branch: `interp_jspace_gemma_transport`.
- Exact fork/HEAD: `3b041735d8b842de46a9c0a474fccd0c44e0841a`.
- Remote Gemma branch: not yet created as of this boundary.
- Dedicated Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_20260802`.
- GPU hard gate: PASS on NVIDIA RTX PRO 6000 Blackwell Server Edition,
  driver 580.82.07, PyTorch 2.11.0+cu128, CUDA build 12.8, capability 12.0.
- Pinned `jacobian-lens` revision `581d398613e5602a5af361e1c34d3a92ea82ba8e`
  is installed editable and byte-identical to the campaign Drive copy.
- Hugging Face authentication: PASS (`kburtram`).
- Shared Part-2 conformance bootstrap: PASS.
- No Gemma or OLMo model producer is running. No scientific Gemma target
  outcome has been opened.
- The isolated package scaffold is complete but not yet committed at this
  checkpoint. Its 22-test conformance suite passes, including analytic,
  nonlinear tiny-transformer, and explicit Hugging Face suffix/JVP tests.
- The historical Drive OLMo snapshot has 11/14 complete weight shards. Shards
  6, 9, and 14 are broken symlinks with stale `.incomplete` blobs; control
  staging must reuse verified complete blobs and download/verify the missing
  content. Never treat that Drive snapshot as load-ready.

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

1. Commit the tested package scaffold and publish the new Gemma branch.
2. From that clean commit, run/register `gm-foundation-v1`, commit, pull,
   test, and push the evidence boundary.
3. From the next clean commit, run analytic and tiny-transformer JVP goldens;
   register
   `gm-jvp-goldens-v1` only from a clean commit.
4. Stage the exact historical OLMo positive-control checkpoint
   `allenai/Olmo-3-32B-Think@ebd033e4f0b284d5973b82c0ccb62ad0dbe877d7`
   from Drive to local NVMe, rehash its snapshot inventory, and run the G1
   control.
5. Freeze numeric Gemma thresholds in a clean, pre-target config and commit.
6. Remove the local OLMo staging copy, download the pinned Gemma 4 31B IT
   snapshot to local NVMe, then run Gemma Stage 1 with the unchanged harness.
7. Route to G2/G3/G5/G4/G6 by the observed G1 branch, preserving G9 time.

## Recovery checks and next commands

There is currently no live producer, lock owner, or partial scientific
checkpoint. The package scaffold is an uncommitted worktree at this exact
handoff, so if reclamation occurs before the next boundary, recover it from
the Colab filesystem if available; otherwise resume from the last remote
commit and rebuild from this queue. On a new VM, run the bootstrap in
`RESUME_GEMMA_TRANSPORT.md`, then inspect:

```bash
git -C /content/labs status --short --branch
git -C /content/labs log -8 --oneline --decorate
nvidia-smi
find /content/drive/MyDrive/interpret/special-lab-1/gemma_transport_20260802 \
  -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %p\n' | sort
ps -eo pid,etimes,cmd | rg -i 'jspace_gemma|gm_exact|gemma-4|Olmo-3-32B'
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
