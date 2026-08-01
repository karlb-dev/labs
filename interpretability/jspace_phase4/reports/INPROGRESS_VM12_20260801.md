# LIVE — Phase 4.2 block 2, VM12

Last updated: 2026-08-01 06:52 UTC. This is the canonical dynamic handoff.
Phase 4 remains **development-only**: do not open confirmatory or replication
intervention outcomes, and do not self-sign independent-review or PI fields.

## Live repository and compute boundary

- Main worktree: `/content/labs`, branch `interp_jspace_part2`.
- Main launch commit: `11ae9db626e9776091840bf5b29b3217b9fd99c0`;
  main is clean and synced while the fitter runs.
- Isolated CPU/report worktree: `/content/labs_phase4_cpu`, branch
  `codex/phase4-cpu-20260801`; the durable successor-queue code boundary is
  `dc75ca2692e80183aa20142c428d0dd436a6c444` (inspect `git rev-parse HEAD`
  for later handoff-only commits).
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
| prompts banked | 256 / 500 |
| recovery checkpoint SHA-256 | `24ce0f37d99423d4eb1ccd253563f99e1d8e14fb8dcee8ee21bee62e14a277c3` |
| checkpoint bytes | 6,606,047,399 |
| observed seconds/new prompt | 178.22 |
| peak allocated VRAM | 62.846 GB |
| next chunk | 256:259 |

Durable monitoring paths:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_continuation_draw_a_n500_20260801.log
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/QWEN_CONTINUATION_WATCHDOG.log
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/QWEN_CONTINUATION.lock
```

At the observed rate, the fit should need about 12.1 hours plus finalization.
The runner atomically mirrors every three prompts and, at n=500, registers
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
- Bank B candidate v2 passes independent source reverification (160/160,
  zero mismatches/unresolved flags), but registered power evidence shows the
  0.25-nat joint SESOI is unusably underpowered. A prospective design/estimand
  revision remains mandatory.
- Bank W candidate v2 and shared-family max-T power are registered and
  outcome-blind. The 24-family side clears the 0.80 simulation target;
  model-specific capability and independent review remain open.
- The OLMo four-checkpoint common-support development closure is registered.

Main evidence integration is pushed through merge commit `11ae9db`. Registry
verification after that merge found 39 live events and 176 registered outputs
with zero failures; the combined Phase 4 suite passed 130 tests.

## Pre-outcome A250–A500 decision path

Before any A500 output existed, isolated commit `b9cb1b1` froze and tested the
successor structural/functional path. All 132 Phase 4 tests pass there.

Successor configs:

```text
interpretability/jspace_phase4/configs/p4_qwen_lens_convergence_drawA_n250_n500_dev.yaml
interpretability/jspace_phase4/configs/p4_qwen_multilens_functional_gate_a250_a500_dev.yaml
```

Both deliberately contain only the sentinel
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
```

## Immediate execution order after n=500

1. Confirm the n=500 runner registered, committed, pushed, and left main
   clean. Independently rehash the registered lens.
2. Merge `codex/phase4-cpu-20260801` into main. Preserve the registry and all
   original commit ancestry; resolve no evidence by overwriting.
3. Bind the registered A500 lens hash into only the two successor YAMLs,
   commit, and push.
4. Launch the durable two-stage queue, which runs structural convergence,
   banks/pushes its registry event, then runs and banks/pushes the functional
   gate:

   ```bash
   bash interpretability/jspace_phase4/run_qwen_a500_postfit_queue.sh
   ```

   It writes a five-minute heartbeat and durable stdout at:

   ```text
   /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_a500_postfit_queue_20260801.log
   /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/QWEN_A500_POSTFIT_QUEUE_WATCHDOG.log
   ```

5. Execute the frozen branch only if enough VM time remains to preserve a
   valid three-prompt checkpoint. A/C selects draw-B n=120; B selects draw-A
   n=1000. The continuation entrypoint supports both `draw_b_n120` and
   `draw_a_n1000`. Do not delay report/registry/handoff closure near reclaim.
6. Integrate exact A500 fit and gate results into Markdown/TeX/PDF, visually
   inspect new figures and affected handout pages, mirror byte-identical
   report artifacts to Drive, and refresh this ledger plus the resume file.

## Final verification and freeze boundary

Before handoff or freeze, require:

```bash
git status --short --branch
git rev-parse HEAD
git ls-remote origin refs/heads/interp_jspace_part2
python -m jspace_phase4 verify
bash interpretability/jspace_phase4/repro.sh
```

The preregistration still cannot freeze until the canonical Qwen lens path is
truthfully resolved or bounded, Bank B and mode protocols are prospectively
revised and re-audited, Bank W capability language is fixed, and independent
review plus PI sign-off are obtained. The agent may prepare evidence and
amendments but may not self-approve those governance gates.
