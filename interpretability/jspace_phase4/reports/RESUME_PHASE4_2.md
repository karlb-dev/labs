# Resume Phase 4.2 — restart-safe handoff

Generated 2026-08-01 08:04 UTC on VM12. Phase 4 is development-only. Never
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
- Isolated branch: `codex/phase4-cpu-20260801`; current pushed boundary
  `1d849a2ada83252ecb542cefc3bed93f14db390d`; the successor queue has four
  prospectively frozen stages: structural, functional, mode-v2 baseline, and
  Qwen Bank W baseline capability
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
boundary is n=280 with checkpoint SHA-256
`af1a7b777beeb1443031be03bcaf69d8c49286450f0a88143b8149dbb3925b9d`.
Observed throughput is 179.21 seconds per new prompt and peak allocated VRAM
is 62.846 GB. The remaining fit is about 11.0 hours plus finalization, so n=500
is projected near 19:05 UTC if throughput holds. The expanded post-fit queue
should consume another 1.5--2 hours.

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
- Prospective mode methods successor `p4-qwen-mode-parser-gate-dev-v2` passes
  all tokenizer/template/golden gates and changes only the 512-token cap to
  2,048. Fresh model baseline `p4-qwen-mode-gate-dev-v2`, with one shared
  at-most-128-reasoning-token instruction, is frozen and queued but unopened.
- Bank B v2: independent verification passes all 160 facts, but registered
  power makes the 0.25-nat joint SESOI unusable. Registered feasibility
  evidence `p4-bank-b-design-feasibility-dev-v1` shows that no allocation of
  the existing 40 families can repair it: optimistic required family counts
  are 3,562 at 0.25 nat and 124 at the consumed 1.342-nat effect; the all-40
  heavy-tail IUT MDE is 3 nats. Substantive review is required.
- Bank W v2/max-T power and OLMo common-support closure are registered.
  Bank W capability methods evidence
  `p4-bank-w-capability-protocol-dev-v1` now pins full-sequence eight-answer
  scoring, the 0.70 endpoint floor, the 90% load-equivalence CI inside
  [-0.08, +0.08], and the non-circular 20-family joint-support rule. Qwen is
  queued; both OLMo baselines and independent review remain pending. No Bank W
  intervention outcome was opened.
- Main integration merge `11ae9db` verified 39 live evidence events and 176
  outputs. The current isolated branch passes 139 tests. Preregistration
  candidate 0.6 and report source are pushed at `1d849a2`; the rebuilt and
  visually checked 18-page PDF is the isolated tree's sole intentional dirty
  file pending A500 integration.

## Pre-frozen A250–A500 successor gate

Isolated commit `b9cb1b1` was created before any A500 output existed. It
generalizes the audited structural/functional producers and freezes the
successor configs. Mode-v2 was prospectively frozen at `76c22fd` and appended
to the queue at `19f5fc4`. Bank W capability protocol was registered at
`38c761c` without model outcomes and the Qwen gate queued at `6f7cce3`; the
combined branch passes all 139 Phase 4 tests.

```text
interpretability/jspace_phase4/configs/p4_qwen_lens_convergence_drawA_n250_n500_dev.yaml
interpretability/jspace_phase4/configs/p4_qwen_multilens_functional_gate_a250_a500_dev.yaml
interpretability/jspace_phase4/configs/p4_qwen_mode_model_gate_v2_dev.yaml
interpretability/jspace_phase4/configs/p4_bank_w_capability_protocol_dev.yaml
```

The first two contain `PENDING_REGISTERED_A500_SHA256`. Once and only once the
n=500 event is live, replace exactly those two sentinels with its registered
lens SHA-256 and commit that binding before running either producer. Do not
change the frozen subset, order, thresholds, seeds, or branch interpretations.

The successor evidence IDs and figure stems are:

```text
p4-qwen-lens-convergence-drawA-n250-n500-dev-v1
p4f19_qwen_lens_convergence_a250_a500
p4-qwen-multilens-functional-gate-a250-a500-published-dev-v1
p4f20_qwen_multilens_functional_gate_a250_a500
p4-qwen-mode-gate-dev-v2
p4f21_qwen_mode_model_gate_v2
p4-bank-w-capability-qwen36-27b-dev-v1
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
4. Launch the durable successor queue. It runs and banks/pushes structural
   convergence, the fixed functional gate, the prospectively amended mode-v2
   baseline, and the Qwen Bank W baseline capability gate in that order:

   ```bash
   bash interpretability/jspace_phase4/run_qwen_a500_postfit_queue.sh
   ```

   Durable logs:

   ```text
   /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_a500_postfit_queue_20260801.log
   /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/QWEN_A500_POSTFIT_QUEUE_WATCHDOG.log
   ```

5. Integrate exact fit/gate results and p4f19/p4f20/p4f21 into the Markdown,
   TeX, final PDF, in-progress ledger, and this resume file. Visually inspect
   all new figures and affected pages. Mirror byte-identical artifacts to
   Drive.
6. Follow the branch without reinterpretation. The continuation entrypoint
   supports A/C as `draw_b_n120` and B as `draw_a_n1000`. Launch only if the
   remaining VM window can preserve a valid checkpoint. Neither continuation
   is expected to finish before reclaim, so bank a valid three-prompt recovery
   boundary and prioritize final report/registry/handoff closure.

## Remaining freeze blockers

The Phase 4 preregistration is still a candidate. A truthful freeze requires:

1. a resolved or explicitly bounded canonical Qwen lens path;
2. a reviewed substantive Bank B replacement design with passing power, or an
   explicit estimation-only reclassification outside the Phase 4 primary
   family;
3. a passing fresh Qwen mode-v2 baseline gate, canonical-family split,
   intervention SESOI/power ruler, and no use of the failed v1 baseline as
   intervention evidence;
4. fixed Bank W model-capability rules plus passing results from all three
   model-specific baseline gates and their joint support decision;
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
