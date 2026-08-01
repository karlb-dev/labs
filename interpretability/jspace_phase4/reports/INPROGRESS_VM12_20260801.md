# LIVE — Phase 4.2 block 2, VM12

Last updated: 2026-08-01 08:41 UTC. This is the canonical dynamic handoff.
Phase 4 remains **development-only**: do not open confirmatory or replication
intervention outcomes, and do not self-sign independent-review or PI fields.

## Live repository and compute boundary

- Main worktree: `/content/labs`, branch `interp_jspace_part2`.
- Main launch commit: `11ae9db626e9776091840bf5b29b3217b9fd99c0`;
  main is clean and synced while the fitter runs.
- Isolated CPU/report worktree: `/content/labs_phase4_cpu`, branch
  `codex/phase4-cpu-20260801`; registered methods are pushed through
  `6661514` (P4-P2 design feasibility), with scoring/runner hardening through
  `258ae3c`. The successor queue has
  four prospectively frozen stages: A250--A500 structural, functional,
  mode-v2 baseline, and Qwen Bank W baseline capability.
- The isolated tree has only the intentionally regenerated handout PDF dirty.
- Phase 4 Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731`.
- Governing review:
  `interpretability/jspace_phase4/reviews/jspace_lab_nextsteps_4_2.md`
  and its accepted addendum.
- Candidate preregistration is not a freeze. Every evidence boundary must be
  clean-tree, registered, committed, pushed, and mirrored; registered outputs
  are immutable.

The user reported about 16 hours of VM time remaining near 06:50 UTC. Treat
roughly 22:50 UTC as a soft reclaim boundary and preserve a restartable state
well before it.

## Active Qwen Branch-B continuation

The frozen A120/A250/published functional rule emitted Branch B, so draw A is
continuing from the exact registered n=250 checkpoint to n=500. The active
host process was launched with:

```bash
bash interpretability/jspace_phase4/run_qwen_continuation_fit.sh draw_a_n500
```

It recovered fit contract
`bf4caff4ff7c389d29f235a91062ae86e3a37dfc526c42bbd9af7c5d7e1f3b00`
at `recovered_next_idx=250`, verified the pinned 59 GB model snapshot, bound
all 48 fused FLA delta-rule modules, and forbids CPU fallback.

Latest durable boundary at this update:

| field | value |
|---|---|
| prompts banked | 289 / 500 |
| recovery checkpoint SHA-256 | `fc0a2e8f65b1f75fd87f5d2ea432a3f37dcfcbe12cfb624a70084351413bbc70` |
| checkpoint bytes | 6,606,047,399 |
| observed seconds/new prompt | 176.38 in the restarted invocation; 179.1 long-run ruler |
| peak allocated VRAM | 62.846 GB |
| active chunk | 289:292 |

Durable monitoring paths:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_continuation_draw_a_n500_20260801.log
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/QWEN_CONTINUATION_WATCHDOG.log
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/QWEN_CONTINUATION.lock
```

At the observed rate, the remaining 211 prompts should need about 10.5 hours
plus finalization, placing the n=500 boundary near 19:10 UTC if throughput
holds. The expanded frozen post-fit queue should then need roughly 1.5--2
hours, leaving about 1.5 hours for result integration and a restartable branch
continuation before the soft reclaim boundary.

**Temporary recovery routing.** Repeated 6.6 GB atomic checkpoint versions
filled the DriveFS content cache to about 85 GB while the Drive API returned
`userRateLimitExceeded` 403s and marked the cache non-evictable. Obsolete,
already-registered local contract/influence scratch was removed, recovering
18.6 GB; its hash-pinned Drive outputs remain intact. Do not manually delete
DriveFS cache internals. To prevent a disk-full failure, only the recovery
subdirectory is currently bind-mounted to local NVMe:

```text
source: /content/phase4_recovery_local_20260801_w4Pv6g/recovery_bind
target: /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery
active local checkpoint: /content/sl4_work/qwen_nested_lens_fit/qwen36-27b/draw_a/bf4caff4ff7c389d29f235a91062ae86e3a37dfc526c42bbd9af7c5d7e1f3b00/fit.ckpt
```

Thus n=289 is machine-local durable, not yet independently cloud-durable.
The canonical corpus path and fit contract are unchanged. A 15-second watcher
will terminate only the Python fitter after the locally mirrored n=499 header
appears. Then unmount the exact target, confirm the underlying Drive recovery
is visible, and rerun the normal command: recovery selection must choose the
newer local n=499 checkpoint and execute only prompt 500 plus the final real
Drive sync/registration. Do not remove the bind or local mirror early.

At n=500 the runner registers
`p4-qwen-lens-fit-drawA-n500-dev-v1`, commits the registry append, and pushes.
Do not edit the main worktree before that runner exits.

If the process disappears before n=500, inspect the log/lock and GPU first.
Only when no fitter is alive, rerun the same command from a clean main tree;
it must recover the highest valid three-prompt Drive checkpoint. Never create
a pseudo-milestone evidence row for a partial fit.

## Completed development evidence in this block

- Registered draw-A n=250 lens:
  `p4-qwen-lens-fit-drawA-n250-dev-v1`; lens SHA-256
  `b78427c84ddd3b9f7f4361b952b5169cf49335e77fe364e584b6abd799f79006`.
- Registered A120–A250 structural convergence:
  `p4-qwen-lens-convergence-drawA-n120-n250-dev-v1`. Frozen assay-band
  task median/q05 gates pass.
- Registered retained-prompt sensitivity:
  `p4-qwen-lens-influence-prompt112-dev-v1`. Prompt 112 remains in A120 and
  is not structurally load-bearing on L20–L44.
- Registered functional gate:
  `p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1`. Structural
  and seven other functional criteria pass, but normalized projector overlap
  (0.67479 < 0.85) and selected-ID Jaccard (0.53846 < 0.75) fail; Branch B is
  `functional instability; continue draw A to n=500`.
- Registered model-mode baseline:
  `p4-qwen-mode-gate-dev-v1`. Thinking-on truncation and parse-failure rates
  are 0.35; only 11 families are correct in both modes versus floor 12. This
  is a baseline methods failure, not an intervention result.
- Prospective mode methods successor `p4-qwen-mode-parser-gate-dev-v2` is
  registered and passes every tokenizer/template/golden gate. It changes only
  the new-token cap from 512 to 2,048. A fresh model-backed baseline
  `p4-qwen-mode-gate-dev-v2`, with one shared at-most-128-reasoning-token
  instruction, was frozen before outcomes and added to the post-A500 queue.
- Registered outcome-blind mode design feasibility:
  `p4-qwen-mode-design-feasibility-dev-v1`. At conservative Holm alpha
  0.05/3 and 80% Gaussian planning power, a 0.20 accuracy-point interaction
  requires 56 families at family SD 0.5, 221 at SD 1.0, and 3,528 at the
  support-based SD bound. No SESOI, bank size, or split was selected; a
  consumed-development intervention variance pilot is the next methods step
  after the v2 baseline and canonical-lens decision.
- Bank B candidate v2 passes independent source reverification (160/160,
  zero mismatches/unresolved flags), but registered power evidence shows the
  0.25-nat joint SESOI is unusably underpowered. Registered feasibility
  successor `p4-bank-b-design-feasibility-dev-v1` proves no allocation of the
  existing 40 families repairs it: the optimistic 0.25-nat lower bound needs
  3,562 families, the consumed 1.342-nat effect needs 124, and the all-40
  heavy-tail IUT MDE is 3 nats. Split-only repair and SESOI inflation are not
  licensed; a substantive estimand/status/bank decision needs review.
- Bank W candidate v2 and shared-family max-T power are registered and
  outcome-blind. The 24-family side clears the 0.80 simulation target.
  Registered methods evidence `p4-bank-w-capability-protocol-dev-v1` pins
  full-sequence eight-answer scoring, the 0.70 endpoint floor, the 90% CI
  equivalence requirement inside [-0.08, +0.08], and a non-circular 20-family
  joint-support rule. The Qwen baseline is queued; both OLMo baselines and
  independent review remain open. No Bank W intervention outcome was opened.
- The OLMo four-checkpoint common-support development closure is registered.

Main evidence integration is pushed through merge commit `11ae9db`. Registry
verification after that merge found 39 live events and 176 registered outputs
with zero failures. The isolated successor branch now passes all 145 Phase 4
tests. Preregistration candidate 0.7 and the report/handout sources integrate
the registered P4-P2 envelope; the rebuilt PDF remains the sole intentional
dirty file until this documentation boundary is compiled and checked.

## Pre-outcome A250–A500 decision path

Before any A500 output existed, isolated commit `b9cb1b1` froze and tested the
successor structural/functional path. The prospective mode-v2 model baseline
was frozen at `76c22fd`. Bank W capability protocol
`p4-bank-w-capability-protocol-dev-v1` was registered at `38c761c` without
model outcomes, and its Qwen baseline was queued at `6f7cce3`. The current
combined isolated branch passes all 145 Phase 4 tests.

Successor configs:

```text
interpretability/jspace_phase4/configs/p4_qwen_lens_convergence_drawA_n250_n500_dev.yaml
interpretability/jspace_phase4/configs/p4_qwen_multilens_functional_gate_a250_a500_dev.yaml
interpretability/jspace_phase4/configs/p4_qwen_mode_model_gate_v2_dev.yaml
interpretability/jspace_phase4/configs/p4_bank_w_capability_protocol_dev.yaml
```

The first two configs deliberately contain only the sentinel
`PENDING_REGISTERED_A500_SHA256`. After the n=500 evidence event exists,
replace exactly those two sentinels with the registered lens hash, commit that
binding before execution, and do not change any threshold, subset, order,
seed, or branch interpretation.

The frozen successor order is published -> A250 -> A500. The primary pair is
A250–A500 with the same consumed Phase 3 subset and the same structural,
selection, capacity, causal, bridge, and G4 thresholds. The result maps to:

- A: same-corpus structure/function stable; fit independent draw B before a
  smaller-lens nomination.
- B: functional instability persists; continue draw A to n=1000.
- C: structure remains sensitive but endpoints are stable; retain A500 as the
  larger same-corpus candidate and fit independent draw B.

Expected successor evidence/figures are:

```text
p4-qwen-lens-convergence-drawA-n250-n500-dev-v1
p4f19_qwen_lens_convergence_a250_a500.{png,pdf}
p4-qwen-multilens-functional-gate-a250-a500-published-dev-v1
p4f20_qwen_multilens_functional_gate_a250_a500.{png,pdf}
p4-qwen-mode-gate-dev-v2
p4f21_qwen_mode_model_gate_v2.{png,pdf}
p4-bank-w-capability-qwen36-27b-dev-v1
```

## Immediate execution order after n=500

1. Confirm the n=500 runner registered, committed, pushed, and left main
   clean. Independently rehash the registered lens.
2. Merge `codex/phase4-cpu-20260801` into main. Preserve the registry and all
   original commit ancestry; resolve no evidence by overwriting.
3. Bind the registered A500 lens hash into only the two successor YAMLs,
   commit, and push.
4. Launch the durable four-stage queue. It runs structural convergence,
   the functional gate, the prospectively amended mode-v2 baseline, and the
   Qwen Bank W baseline capability gate, banking/pushing each event in turn:

   ```bash
   bash interpretability/jspace_phase4/run_qwen_a500_postfit_queue.sh
   ```

   It writes a five-minute heartbeat and durable stdout at:

   ```text
   /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_a500_postfit_queue_20260801.log
   /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/QWEN_A500_POSTFIT_QUEUE_WATCHDOG.log
   ```

5. Integrate exact A500 fit and gate results into Markdown/TeX/PDF, visually
   inspect new figures and affected handout pages, mirror byte-identical
   report artifacts to Drive, and refresh this ledger plus the resume file.
6. Execute the frozen branch if enough VM time remains to preserve a
   valid three-prompt checkpoint. A/C selects draw-B n=120; B selects draw-A
   n=1000. The continuation entrypoint supports both `draw_b_n120` and
   `draw_a_n1000`. Neither continuation can finish inside the estimated
   remaining VM window, so bank a restartable three-prompt checkpoint and do
   not delay final registry/report/handoff closure near reclaim.

## Final verification and freeze boundary

Before handoff or freeze, require:

```bash
git status --short --branch
git rev-parse HEAD
git ls-remote origin refs/heads/interp_jspace_part2
python -m jspace_phase4 verify
bash interpretability/jspace_phase4/repro.sh
```

Candidate preregistration 0.7 still cannot freeze until the canonical Qwen
lens path is truthfully resolved or bounded; Bank B receives a substantive
reviewed replacement design or estimation-only reclassification; mode v2
passes a fresh baseline, then receives a consumed-development intervention
variance pilot, reviewed SESOI, exact power ruler, untouched split, and review;
all three Bank W baseline capability gates and their joint support decision
pass; and independent review plus PI sign-off are obtained. The agent may
prepare evidence and amendments but may not self-approve those governance
gates.
