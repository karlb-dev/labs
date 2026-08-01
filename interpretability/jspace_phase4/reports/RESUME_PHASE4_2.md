# Resume Phase 4.2 — restart-safe handoff

Generated 2026-08-01 06:52 UTC on VM12. Phase 4 is development-only. Never
open confirmatory or replication intervention outcomes before independent
review, PI sign-off, a freeze commit, and a freeze tag.

Canonical mirrors of this file must remain byte-identical at:

```text
/content/resume-phase-4-2.md
/content/drive/MyDrive/interpret/resume-phase-4-2.md
```

Read the fuller live ledger first:

```text
/content/drive/MyDrive/interpret/inprogress.md
interpretability/jspace_phase4/reports/INPROGRESS_VM12_20260801.md
```

## Current live boundary

- Main repository/worktree: `/content/labs`
- Branch: `interp_jspace_part2`
- Clean launch commit: `11ae9db626e9776091840bf5b29b3217b9fd99c0`
- Isolated preparation worktree: `/content/labs_phase4_cpu`
- Isolated branch: `codex/phase4-cpu-20260801`; durable successor-queue code
  boundary `dc75ca2692e80183aa20142c428d0dd436a6c444` (inspect
  `git rev-parse HEAD` for later handoff-only commits)
- Phase 4 Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731`
- User-reported VM window: about 16 hours remaining near 06:50 UTC; treat
  roughly 22:50 UTC as a soft reclaim boundary.

The frozen A120/A250/published functional gate emitted Branch B because
A120–A250 selection geometry failed both load-bearing thresholds. Draw A is
therefore actively continuing from exact n=250 to n=500 via:

```bash
bash interpretability/jspace_phase4/run_qwen_continuation_fit.sh draw_a_n500
```

The runner verified the model snapshot, exact fit contract, CUDA, and all 48
fused FLA bindings. CPU fallback is forbidden. It recovered
`recovered_next_idx=250`; at this handoff the latest atomically mirrored
boundary is n=256 with checkpoint SHA-256
`24ce0f37d99423d4eb1ccd253563f99e1d8e14fb8dcee8ee21bee62e14a277c3`.
Observed throughput is 178.22 seconds per new prompt and peak allocated VRAM
is 62.846 GB.

Monitoring/recovery paths:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_continuation_draw_a_n500_20260801.log
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/QWEN_CONTINUATION_WATCHDOG.log
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/QWEN_CONTINUATION.lock
```

Do not launch a duplicate. Check the lock, process, GPU, heartbeat, and log.
If the fitter has genuinely stopped before n=500, rerun the same command from
a clean main tree; it must recover the highest valid three-prompt checkpoint.
Never register a partial pseudo-milestone.

At n=500 the runner should register
`p4-qwen-lens-fit-drawA-n500-dev-v1`, commit the evidence registry append, and
push. It intentionally refuses unexpected main-worktree changes, so keep main
untouched until it exits.

## Evidence already completed

- A250 lens: `p4-qwen-lens-fit-drawA-n250-dev-v1`, lens SHA-256
  `b78427c84ddd3b9f7f4361b952b5169cf49335e77fe364e584b6abd799f79006`.
- A120–A250 convergence:
  `p4-qwen-lens-convergence-drawA-n120-n250-dev-v1`; structural assay gates
  pass.
- Prompt-112 retained influence:
  `p4-qwen-lens-influence-prompt112-dev-v1`; not structurally load-bearing on
  the frozen assay band.
- Functional Branch B:
  `p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1`; projector
  overlap 0.67479 < 0.85 and selected-ID Jaccard 0.53846 < 0.75.
- Qwen model-mode baseline: `p4-qwen-mode-gate-dev-v1`; thinking-on parse and
  truncation rates 0.35, common-correct families 11 < 12. No intervention was
  run.
- Bank B v2: independent verification passes all 160 facts, but registered
  power makes the 0.25-nat joint SESOI unusable; prospective redesign needed.
- Bank W v2/max-T power and OLMo common-support closure are registered.
- Main integration merge `11ae9db` verified 39 live evidence events and 176
  outputs; the then-current combined suite passed 130 tests.

## Pre-frozen A250–A500 successor gate

Isolated commit `b9cb1b1` was created before any A500 output existed. It
generalizes the audited producers, freezes the successor configs, and passes
all 132 Phase 4 tests.

```text
interpretability/jspace_phase4/configs/p4_qwen_lens_convergence_drawA_n250_n500_dev.yaml
interpretability/jspace_phase4/configs/p4_qwen_multilens_functional_gate_a250_a500_dev.yaml
```

Both contain `PENDING_REGISTERED_A500_SHA256`. Once and only once the n=500
event is live, replace exactly those two sentinels with its registered lens
SHA-256 and commit that binding before running either producer. Do not change
the frozen subset, order, thresholds, seeds, or branch interpretations.

The successor evidence IDs and figure stems are:

```text
p4-qwen-lens-convergence-drawA-n250-n500-dev-v1
p4f19_qwen_lens_convergence_a250_a500
p4-qwen-multilens-functional-gate-a250-a500-published-dev-v1
p4f20_qwen_multilens_functional_gate_a250_a500
```

Frozen successor decisions:

- A: A250–A500 structural/functional stability; fit independent B120 before
  any smaller-lens nomination.
- B: functional instability persists; continue draw A to n=1000.
- C: structural sensitivity with functional equivalence; retain A500 as the
  larger same-corpus candidate and fit independent B120.

## Exact continuation after n=500

1. Verify the fitter registered/committed/pushed n=500 and main is clean.
2. Merge `codex/phase4-cpu-20260801` into `interp_jspace_part2` without
   flattening evidence ancestry.
3. Bind and commit the registered A500 lens hash in the two successor YAMLs.
4. Launch the durable successor queue. It runs structural convergence,
   commits/pushes that registry event, then runs and commits/pushes the fixed
   functional gate:

   ```bash
   bash interpretability/jspace_phase4/run_qwen_a500_postfit_queue.sh
   ```

   Durable logs:

   ```text
   /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_a500_postfit_queue_20260801.log
   /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/QWEN_A500_POSTFIT_QUEUE_WATCHDOG.log
   ```

5. Follow the branch without reinterpretation. The continuation entrypoint
   supports A/C as `draw_b_n120` and B as `draw_a_n1000`. Launch only if the
   remaining VM window can preserve a valid checkpoint; prioritize
   report/registry/handoff closure near reclaim.
6. Integrate exact fit/gate results and p4f19/p4f20 into the Markdown, TeX,
   final PDF, in-progress ledger, and this resume file. Visually inspect all
   new figures and affected pages. Mirror byte-identical artifacts to Drive.

## Remaining freeze blockers

The Phase 4 preregistration is still a candidate. A truthful freeze requires:

1. a resolved or explicitly bounded canonical Qwen lens path;
2. a prospective Bank B design/estimand revision with a passing power audit;
3. a prospective Qwen mode amendment, fresh baseline gate, canonical-family
   split, intervention SESOI/power ruler, and no use of the failed baseline as
   intervention evidence;
4. fixed Bank W model-capability rules;
5. independent protocol review and PI sign-off.

The agent may prepare and test these materials, but cannot self-approve the
last governance gates or open untouched intervention outcomes.

## Boundary verification

Before VM reclaim or freeze, run:

```bash
git status --short --branch
git rev-parse HEAD
git ls-remote origin refs/heads/interp_jspace_part2
python -m jspace_phase4 verify
bash interpretability/jspace_phase4/repro.sh
```

Commit and push every result-bearing registry boundary. Keep registered
evidence immutable, keep long GPU work on Drive-backed three-prompt recovery,
and refresh all three handoff mirrors after each material milestone.
