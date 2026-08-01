# LIVE — Phase 4.2 block 2, VM12

Last updated: 2026-08-01 20:15 UTC. This is the canonical dynamic handoff.
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
- The user reported a soft VM reclaim near 22:50 UTC. The remaining session
  should prioritize a clean document commit and restartable A1000 checkpoints.

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
handoff are being updated in one integration commit. Rebuild and visually
inspect the PDF before banking that commit.

## Next executable path

The frozen router must mechanically read the registered functional result and
map Branch B to `draw_a_n1000`:

```bash
bash interpretability/jspace_phase4/run_qwen_frozen_branch_followup.sh
```

The router refuses a dirty worktree, branch drift, hash drift, or branch-
interpretation drift. Launch it only after the report/PDF integration commit
is clean and pushed. It writes:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_frozen_branch_followup_20260801.log
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_frozen_branch_followup.lock
```

A500--A1000 requires about 500 new prompts. At the observed 175.66 seconds per
prompt, a full fit needs roughly 24.4 hours, so this VM cannot finish it.
Use the remaining window to bank as many atomic three-prompt checkpoints as
possible, then preserve a restart-safe handoff. Do not invent a partial
milestone evidence event.

Do not delete the Qwen cache while A1000 is active. The two OLMo Bank-W gates
should run only on a later machine or after a safe, explicit cache/staging
transition.

## Freeze blockers

Phase 4 is **not frozen**. Remaining blockers include:

1. completion of the Branch-B A1000 fit and its frozen successor decision;
2. substantive Bank-B replacement-design or estimation-only decision;
3. reviewed P4-P2 GPU pilot, variance estimate, SESOI, exact power, and
   untouched family split;
4. both OLMo Bank-W capability gates and joint common support;
5. whole-registry/output verification after DriveFS durability is confirmed;
6. independent protocol review;
7. PI sign-off, freeze commit, and freeze tag.

Confirmatory and replication jobs remain forbidden.
