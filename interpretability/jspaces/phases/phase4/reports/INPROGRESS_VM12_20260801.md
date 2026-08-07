# LIVE — Phase 4.2 block 2, VM12

Last updated: 2026-08-01 23:26 UTC. This is the canonical dynamic handoff.
Phase 4 remains **development-only**. Never open confirmatory or replication
intervention outcomes, and never self-sign independent-review or PI fields.

## Current boundary

- Main worktree: `/content/labs`, branch `interp_jspace_part2`.
- Last registered compute commit: `f73614d6288179e73a496283dcd10090f76f2815`;
  it is pushed to origin.
- The isolated preparation branch was merged without flattening its ancestry.
- The append-only registry has 63 events. The merge preserves the preparation
  branch's exact 58-event prefix and the exact registered A500 event once.
- The merged suite passes: **153 tests passed**.
- Phase 4 Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731`.
- A1000 was intentionally paused at the clean n=554 checkpoint for a VM
  handoff. No fitter/router process, lock inode holder, or GPU compute process
  remained after the pause audit.

No confirmatory or replication intervention outcome has been opened.

## Completed Qwen draw-A n=500 fit

Live evidence: `p4-qwen-lens-fit-drawA-n500-dev-v1`.

The exact n=250 estimator continued on the unchanged nested draw-A corpus and
fit contract to n=500. The n=499 recovery handoff terminated only the active
Python process, removed the exact temporary recovery bind, restored the
registered A250 bytes, and launched the unchanged normal producer for prompt
500. The final registered event was committed at `9eac945`.

| field | value |
|---|---|
| prompts | 500 |
| lens SHA-256 | `84404956fb71a84f5af7fa22c34a8f07761777d048b3db314b1330037e4168a8` |
| lens bytes | 3,303,034,078 |
| final checkpoint SHA-256 | `f1f5c3eebca93bd0f8d00cbd8794848df569bc67be475d3ab4ef319d8aaa61b1` |
| checkpoint bytes | 6,606,047,399 |
| fit contract | `bf4caff4ff7c389d29f235a91062ae86e3a37dfc526c42bbd9af7c5d7e1f3b00` |
| peak allocated VRAM | 62.846 GB |
| model | Qwen/Qwen3.6-27B at `6a9e13bd...` |

All 250 continuation prompt records 251--500 have finite diagnostics. Prompt
323 is the retained heavy-tail maximum
(`max||J||/sqrt(d)=173.345`, `max_d_mean=0.271`), comparable to the
registered prompt-112 sensitivity. Every later prompt is at or below 79.579;
prompt 500 is ordinary at 13.380. No row was trimmed or refit.

The A500 artifact is independently preserved and content-addressed:

```text
/content/phase4_recovery_local_20260801_w4Pv6g/qwen36-27b_jlens_drawA_n0500.preserved.pt
/content/sl4_work/inputs/84404956fb71a84f5af7fa22c34a8f07761777d048b3db314b1330037e4168a8/qwen36-27b_jlens_drawA_n0500.pt
/content/sl4_work/postfit_registered_backups/p4-qwen-lens-fit-drawA-n500-dev-v1/
```

The preserved file and both local links share one inode and rehash to the
registered lens SHA. The result and input manifest backups also match their
registered hashes.

## Completed post-A500 queue

The exact A500 hash was bound in only the two frozen successor configs and
pushed at `e6fea8e`. The four-stage queue completed at 20:10 UTC. Each event
was registered, committed, pushed, then copied into a hash-verified local
backup under `/content/sl4_work/postfit_registered_backups/<evidence-id>/`.

### 1. A250--A500 structural convergence

Evidence: `p4-qwen-lens-convergence-drawA-n250-n500-dev-v1`.
Registry commit: `bbbd6a7`.

- Frozen L20--L44 median raw/minus-identity/minus-scaled-identity matrix
  cosines: 0.99791 / 0.99936 / 0.99774.
- Conservative task-token median cosine: 0.997715 (floor 0.95).
- Conservative task-token q05: 0.996838 (floor 0.90).
- Both structural gates pass.
- p4f19 was visually checked and is legible.

### 2. A250--A500 functional gate

Evidence:
`p4-qwen-multilens-functional-gate-a250-a500-published-dev-v1`.
Registry commit: `d54688e`.

The frozen decision is **Branch B**.

| gate | result |
|---|---:|
| normalized projector overlap | 0.70302 < 0.85 — fail |
| selected-ID Jaccard | 0.53846 < 0.75 — fail |
| bridge rescue difference | 0.55891 > 0.25 — fail |
| occupancy | pass |
| centered excess | pass |
| span-safe-specific point estimate | pass |
| tail rate | pass |
| G4 positive control | pass |
| bridge preference | pass |
| structural dependency | pass |

The span-safe-specific mean difference is -0.02432 nat, but its 95% interval
[-0.25248, 0.16762] does not establish formal TOST equivalence. The
precommitted interpretation is functional instability; continue draw A to
n=1000. Neither A250 nor A500 is canonical. p4f20 was visually checked.

### 3. Official-Qwen mode-v2 baseline

Evidence: `p4-qwen-mode-gate-dev-v2`.
Registry commit: `08d5773`.

All model-backed development gates pass:

- thinking-on/off accuracy: 0.90 / 0.85;
- parse failure: 0 / 0;
- truncation: 0 / 0;
- parse-valid in both modes: 20/20;
- correct in both modes: 17/20, above floor 12;
- median generated tokens: 323.5 on / 4 off;
- thinking-on minus off accuracy: +0.05, 90% family-bootstrap interval
  [0.00, 0.15].

This is baseline parser/capability evidence only. No P4-P2 intervention ran.
p4f21 was visually checked.

### 4. Qwen Bank-W capability

Evidence: `p4-bank-w-capability-qwen36-27b-dev-v1`.
Registry commit: `f73614d`.

Qwen is independently capability-eligible:

- complete grid: 384 rows;
- accuracy: 0.83333 at both low and high load;
- high-minus-low accuracy: 0.00000;
- 90% family-bootstrap interval: [-0.02083, 0.02083], wholly inside
  [-0.08, 0.08];
- capable at both loads: 20/24 families, meeting the floor.

Both OLMo capability gates and the all-model joint-support decision remain
open. No Bank-W intervention outcome ran.

## Report and figure integration

The exact p4f19/p4f20/p4f21 PNG/PDF pairs were copied from their registered
local backups into `reports/figures/` and rehashed to the registry. All
three PNGs were visually inspected. The Markdown development report,
preregistration candidate 0.9, TeX handout, this handoff, and the restart
handoff were integrated, rebuilt, visually checked, and pushed at
`3b041735d8b842de46a9c0a474fccd0c44e0841a`.

## Paused Branch-B A1000 continuation

The frozen router verified registered result SHA-256
`c3290abc85c93b0fc8144432a2b674b886e3173609379598bd8c95a96ec471e4`,
resolved Branch B without reinterpretation, and launched:

```bash
bash interpretability/jspaces/phases/phase4/run_qwen_continuation_fit.sh draw_a_n1000
```

The run started from clean commit `3b04173`, recovered exact n=500, and
reverified the model, CUDA, and all 48 fused modules. It banked atomic
three-prompt checkpoints through the intentional handoff boundary:

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

1. Mount this same Drive root; check out/pull clean branch
   `interp_jspace_part2` in `/content/labs` and read both handoff files.
2. Verify `checkpoint_state.json` reports `n_done=554`; rehash
   `recovery/fit.ckpt` to the checkpoint SHA above; verify no fitter, router,
   lock holder, or GPU process already owns the run.
3. Provide the frozen runtime and Qwen cache (default
   `HF_HUB_CACHE=/content/hf_local`) for model revision
   `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
4. From the clean repository run:

```bash
bash interpretability/jspaces/phases/phase4/run_qwen_frozen_branch_followup.sh
```

The router must reverify the registered functional-result hash, resolve
Branch B, and print `resuming from checkpoint: 554/...`. Do not launch the
lower-level fitter in parallel, delete the recovery pair, or register n=554
as a partial pseudo-milestone.

Do not delete the Qwen cache while A1000 needs it. Run the two OLMo Bank-W
gates only after A1000 or after a safe, explicit cache/staging transition.

## Drive durability audit

The 23:25 UTC whole-registry pass checked 49 live events and 220 output
references. It found no wrong hashes, but three paths from the older
`p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1` event were
absent. The published reconstruction was restored byte-for-byte from the
hash-verified A250--A500 backup and now matches registered SHA-256
`ebe0c641115b40db089412fb969e8c37eef90a7992a40996c4277d9dac43d703`.

Thus 218/220 live outputs are hash-accounted. These two older registered paths
remain missing in that event directory:

```text
state.json
expected 361bda08e9ffbe1d333fd3cfaf3c7b9545e6a3504246a16dd8b0c07ad26f45e8

capacity_reconstructions_a120.pt
expected 6b0399df2c57158e7fdad24274e50f8c1058021d233412afdcc5177f6c651b6f
```

No preserved exact copy was found on this VM. Do not fabricate bytes, edit the
registry, or rerun model-scale work on CPU. Recover exact bytes from backup or
version history if available; otherwise use a reviewed append-only correction
plan, then rerun `jspace-phase4 verify`.

## Freeze blockers

Phase 4 is **not frozen**. Remaining blockers include:

1. completion of the Branch-B A1000 fit and its frozen successor decision;
2. substantive Bank-B replacement-design or estimation-only decision;
3. reviewed P4-P2 GPU pilot, variance estimate, SESOI, exact power, and
   untouched family split;
4. both OLMo Bank-W capability gates and joint common support;
5. recovery of the two missing older A120--A250 outputs and a clean
   whole-registry verification after DriveFS durability is confirmed;
6. independent protocol review;
7. PI sign-off, freeze commit, and freeze tag.

Confirmatory and replication jobs remain forbidden.
