# Resume Phase 4.2 — restart-safe handoff

Generated 2026-08-01 23:26 UTC on VM12. Phase 4 is development-only. Never
open confirmatory or replication intervention outcomes before independent
review, PI sign-off, a freeze commit, and a freeze tag.

Canonical mirrors of this file must remain byte-identical:

```text
/content/resume-phase-4-2.md
/content/drive/MyDrive/interpret/resume-phase-4-2.md
```

Read the fuller live ledger at:

```text
/content/drive/MyDrive/interpret/inprogress.md
/content/labs/interpretability/jspace_phase4/reports/INPROGRESS_VM12_20260801.md
```

## Current state

- Repository: `/content/labs`
- Branch: `interp_jspace_part2`
- Last registered compute commit: `f73614d6288179e73a496283dcd10090f76f2815`
- Registry: 63 append-only events
- Tests after merge: 153 passed
- A1000 was intentionally paused at the clean n=554 checkpoint for a VM
  handoff. No fitter/router process, lock inode holder, or GPU compute process
  remained after the pause audit.

The CPU preparation branch is merged. The A500 event, four post-fit events,
and their commits are pushed. All registered post-fit outputs have verified
local backups under:

```text
/content/sl4_work/postfit_registered_backups/
```

## Completed A500 boundary

Evidence: `p4-qwen-lens-fit-drawA-n500-dev-v1`.

```text
lens SHA-256:
84404956fb71a84f5af7fa22c34a8f07761777d048b3db314b1330037e4168a8

checkpoint SHA-256:
f1f5c3eebca93bd0f8d00cbd8794848df569bc67be475d3ab4ef319d8aaa61b1

content-addressed lens:
/content/sl4_work/inputs/84404956fb71a84f5af7fa22c34a8f07761777d048b3db314b1330037e4168a8/qwen36-27b_jlens_drawA_n0500.pt
```

All prompts 251--500 logged finite diagnostics. Prompt 323 is a retained
heavy-tail row; no outcome-dependent trimming occurred.

## Completed post-fit decisions

1. `p4-qwen-lens-convergence-drawA-n250-n500-dev-v1`
   - structural median/q05: 0.997715 / 0.996838;
   - both frozen structural gates pass;
   - commit `bbbd6a7`.
2. `p4-qwen-multilens-functional-gate-a250-a500-published-dev-v1`
   - projector overlap 0.70302 < 0.85;
   - selected-ID Jaccard 0.53846 < 0.75;
   - bridge-rescue difference 0.55891 > 0.25;
   - frozen result **Branch B: continue draw A to n=1000**;
   - commit `d54688e`.
3. `p4-qwen-mode-gate-dev-v2`
   - all development gates pass;
   - on/off accuracy 0.90/0.85, zero parse failures and truncations,
     17/20 common-correct;
   - commit `08d5773`.
4. `p4-bank-w-capability-qwen36-27b-dev-v1`
   - Qwen independently eligible;
   - both-load accuracy 0.83333, difference 0.00000 with 90% interval
     [-0.02083, 0.02083], 20 capable families;
   - commit `f73614d`.

p4f19, p4f20, and p4f21 were visually inspected and copied into the report
figure directory with hashes matching their registered outputs.

## Paused A1000 continuation

The report/PDF integration is pushed at
`3b041735d8b842de46a9c0a474fccd0c44e0841a`. The frozen router verified the
registered functional-result hash, resolved Branch B, and launched:

```bash
bash interpretability/jspace_phase4/run_qwen_continuation_fit.sh draw_a_n1000
```

The producer recovered exact n=500 and banked atomic three-prompt
checkpoints through the intentional handoff boundary:

| field | value |
|---|---|
| prompts banked | 554 / 1000 |
| checkpoint SHA-256 | `bf992067d690123109198c182a21169379e5752d89e73e96514fab7127fba74d` |
| checkpoint-state SHA-256 | `cc5478c836023183ec974d30a2c6eae7e1f7856d50fedac96c24ef35cfa3590d` |
| checkpoint bytes | 6,606,047,399 |
| seconds/new prompt | 178.48 |
| peak allocated VRAM | 62.846 GB |
| process state | intentionally paused; no owner |
| next chunk | 554:557 |

Authoritative recovery files and prior router log:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_frozen_branch_followup_20260801.log
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_frozen_branch_followup.lock
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json
```

All continuation diagnostics through prompt 554 are finite. Prompts 541 and
542 are retained local-tail rows at 25.727 and 36.435 respectively, still
below the later-A500 maximum 79.579 and the retained prompt-323 maximum
173.345; no outcome-dependent trimming occurred.

At the measured rate, n=554--1000 alone requires about 22.1 compute hours,
before successor gates and report integration. n=554 is a recovery boundary,
not a registered evidence milestone and not a canonical-lens decision.

The wrapper was intentionally interrupted with exit code 130 only after the
n=554 checkpoint was mirrored and rehashed. Its next loop announced 554:557
and loaded n=554, but prompt 555 was never processed. The zero-byte lockfile
paths remain on Drive as normal `flock` targets; no process holds them.

### Exact next-VM restart

1. Mount the same Drive root and check out/pull clean branch
   `interp_jspace_part2` in `/content/labs`.
2. Read this file and `/content/drive/MyDrive/interpret/inprogress.md`.
3. Verify `checkpoint_state.json` reports `n_done=554`, then rehash
   `recovery/fit.ckpt` to the checkpoint SHA above. Check that no fitter,
   router, lock holder, or GPU process already owns the run.
4. Provide the frozen runtime and Qwen cache (default
   `HF_HUB_CACHE=/content/hf_local`) for exact model revision
   `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
5. From the clean repository run:

```bash
bash interpretability/jspace_phase4/run_qwen_frozen_branch_followup.sh
```

The router must reverify the registered functional-result hash, resolve
Branch B, and print `resuming from checkpoint: 554/...`. Do not launch the
lower-level fitter in parallel, delete the recovery pair, or register n=554
as a partial pseudo-milestone.

## Drive durability audit

The 23:25 UTC whole-registry pass checked 49 live events and 220 output
references. It found no wrong hashes, but three paths from the older
`p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1` event were
absent. The published reconstruction was restored byte-for-byte from the
hash-verified A250--A500 backup and now matches registered SHA-256
`ebe0c641115b40db089412fb969e8c37eef90a7992a40996c4277d9dac43d703`.

Thus 218/220 live outputs are hash-accounted. These two older registered paths
remain missing and keep whole-registry verification red:

```text
state.json
expected 361bda08e9ffbe1d333fd3cfaf3c7b9545e6a3504246a16dd8b0c07ad26f45e8

capacity_reconstructions_a120.pt
expected 6b0399df2c57158e7fdad24274e50f8c1058021d233412afdcc5177f6c651b6f
```

Both belong under the registered A120--A250 functional-gate result directory.
No preserved exact copy was found on this VM. Do not fabricate bytes, edit the
registry, or rerun model-scale work on CPU. Recover exact bytes from backup or
version history if available; otherwise use a reviewed append-only correction
plan, then rerun `jspace-phase4 verify`.

## Other remaining work

- Run the OLMo-3.1 Think and Instruct Bank-W capability gates and compute the
  frozen joint capable-family intersection. Do not delete Qwen cache while
  A1000 needs it.
- Review the P4-P2 GPU variance-pilot producer. The passing v2 baseline does
  not authorize intervention execution by itself; canonical lens, pilot,
  SESOI, exact power, and untouched split remain open.
- Make a substantive reviewed Bank-B replacement-design or estimation-only
  decision. The current 0.25-nat IUT is not repairable by reallocating 40
  families.
- Recover the two missing older A120--A250 functional-gate outputs above,
  confirm Drive cloud durability, and rerun whole-registry verification.
- Obtain independent protocol review and PI sign-off.

## Hard governance boundary

- No confirmatory/replication intervention outcomes.
- No self-signing reviewer or PI fields.
- Registered evidence is immutable; corrections append new events.
- Model-scale CPU fallback is forbidden.
- Phase 4 is not frozen until review, sign-off, freeze commit, and freeze tag
  all exist.
