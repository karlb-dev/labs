# Resume Phase 4.2 — restart-safe handoff

Generated 2026-08-01 20:15 UTC on VM12. Phase 4 is development-only. Never
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
- User-reported soft reclaim: approximately 22:50 UTC

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

## Immediate continuation

First finish, compile, visually inspect, commit, push, and mirror the current
report integration. The worktree must then be clean.

Launch the already-frozen mechanical router:

```bash
cd /content/labs
bash interpretability/jspace_phase4/run_qwen_frozen_branch_followup.sh
```

It must resolve Branch B and emit:

```json
{"branch":"B","continuation":"draw_a_n1000"}
```

Then it execs the unchanged continuation fitter. Monitor:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_frozen_branch_followup_20260801.log
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_frozen_branch_followup.lock
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json
```

A500--A1000 is approximately 24.4 hours at the observed 175.66 seconds per
prompt. This VM can only make partial, checkpointed progress. Preserve the
highest valid atomic checkpoint before reclaim and never register a partial
pseudo-milestone.

Do not launch a duplicate fitter. Check process, lock, GPU, log, and recovery
header first. If no process owns the run, rerun the same router/continuation
from a clean branch; it must recover the highest valid checkpoint.

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
- Restore/confirm Drive cloud durability, then run a fresh whole-registry
  output verification.
- Obtain independent protocol review and PI sign-off.

## Hard governance boundary

- No confirmatory/replication intervention outcomes.
- No self-signing reviewer or PI fields.
- Registered evidence is immutable; corrections append new events.
- Model-scale CPU fallback is forbidden.
- Phase 4 is not frozen until review, sign-off, freeze commit, and freeze tag
  all exist.
