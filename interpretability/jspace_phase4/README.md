# J-space Phase 4

Phase 4 asks what computation occupies the verbalizable channel, how
post-training reroutes it, when controlled load engages it, and when the
J-lens is a valid transport instrument.

Phase 2 and Phase 3 are closed inputs. Their artifacts enter this package
only through immutable import events. The Phase 4 preregistration remains a
candidate: **no Phase 4 confirmatory or replication model cell is authorized
until PI sign-off and a freeze tag exist.**

## GPU invariant

Every model-scale entrypoint must call
`jspace_phase4.gpu.require_cuda_gpu()` in the same process before loading
weights and must call `assert_model_on_cuda(model)` after load. A sandboxed
CUDA failure is a hard stop and a request to relaunch the exact command with
host GPU access; it is never permission to fall back to CPU.

CPU is reserved for conformance tests, hashing, state/manifest handling,
statistics, plotting, and document compilation.

## Quick start

```bash
export JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731
bash interpretability/jspace_phase4/repro.sh
jspace-phase4 registry-list
jspace-phase4 verify
```

The run root, model cache, Phase 3 imports, and repository files are resolved
through logical URIs. Scientific modules must not embed machine paths.

## Phase 4.3 resumable work block

The registered post-A500 decision is Branch B. Its Qwen Draw A continuation
resumed from the durable n=554 VM handoff under the frozen wrapper. The live
checkpoint is recorded in the Drive handoff; n=554 remains the immutable
restart provenance boundary:

```text
checkpoint SHA-256: bf992067d690123109198c182a21169379e5752d89e73e96514fab7127fba74d
checkpoint bytes:   6,606,047,399
next chunk:         554:557
```

n=554 is a restart boundary, not a registered evidence milestone. The
continuation resumes from the highest contract-matched local/Drive checkpoint
and refuses CPU or slow-kernel fallback. Read both handoffs before recovery:

```bash
export JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731
export JSPACE4_LOCAL_WORK=/content/sl4_work
export HF_HUB_CACHE=/content/hf_local
cat /content/resume-phase-4-2.md
cat /content/drive/MyDrive/interpret/inprogress.md
```

The n=500 fit and four-stage post-fit queue are complete and registered. The
exact A500 hash is bound in the two successor YAMLs, and the structural,
functional, official mode-v2, and Qwen Bank-W capability events are pushed.
Do not rerun that queue. Apply the registered A/B/C branch without changing
thresholds:

```bash
bash interpretability/jspace_phase4/run_qwen_frozen_branch_followup.sh
```

The router verifies the registered result envelope and exact frozen wording;
the registered result is Branch B, so it continues with `draw_a_n1000` and
must resume from the highest contract-matched checkpoint recorded by the live
handoff (never below n=554). A/C would continue with `draw_b_n120`. Do not
launch a duplicate while either lock is held.

The lower-level producer remains available for exact recovery diagnostics:

```bash
python -m jspace_phase4.experiments.p4_qwen_nested_lens_fit \
  --config interpretability/jspace_phase4/configs/p4_qwen_nested_lens_fit_dev.yaml \
  --draw draw_a --stop-at 1000
```

The handoff-time whole-registry audit accounts for 218/220 live outputs with
no hash mismatch. Two files from the older A120--A250 functional-gate event
are absent: `state.json` (expected SHA-256 `361bda08...f45e8`) and
`capacity_reconstructions_a120.pt` (expected `6b0399df...51b6f`). The missing
published reconstruction was restored from an exact hash-verified backup.
See the handoffs for full paths and hashes; do not fabricate the remaining
bytes or edit the append-only registry. Candidate 0.11 adds a hash-gated A120
capacity recovery which may run only after A1000 releases the GPU and only
installs a bit-exact match. `state.json` remains an external-review/PI
governance decision if no exact backup can be found.

The n=250 convergence, retained prompt-112 influence, functional Branch-B
gate, A500 fit and successor gates, passing mode-v2 baseline, Qwen Bank-W
capability, mode-parser v2 contract, Bank B feasibility, Bank-W protocol, and
conditional P4-P2 variance-pilot protocol are registered. Do not rerun or
overwrite them; registry verification checks their immutable hashes.

Candidate 0.11 keeps Bank B/P4-P1 estimation-only while preparing the mandated
single consumed-development orthogonal feasibility shot, fixes the pre-pilot
0.20-point P4-P2 SESOI, and now includes the executable GPU producer plus its
mean-masked exact/Monte-Carlo power ruler. The OLMo capability service result
fails at 16/20 common families, so P4-P3 is blocked. The A500--A1000 successor
configs, selection-margin contract, prompt-323 influence contract, and
conditional Q-L2 estimand were committed before A1000 existed; their A1000
hash stays unbound until registration.

Gemma and OLMo execute on isolated branches and registries. Their results can
enter mainline only through `protocol/SIDE_TRACK_IMPORT_BUNDLE_CONTRACT.md`.
Whole-registry release requires the two-pass procedure in
`protocol/DRIVEFS_DURABILITY_PLAN.md`; known deficits remain failures.

## Current development synthesis

- Living Markdown: `reports/PHASE4_DEVELOPMENT_REPORT.md`
- Candidate freeze-gate ledger:
  `preregistration/FREEZE_GATE_LEDGER_PHASE4.md`
- Methods decision record: `paper/PHASE4_METHODS_DECISION_RECORD.md`
- Falsifiable conclusion skeleton: `paper/PAPER_CONCLUSION_SKELETON.md`
- Compiled handout:
  `reports/handout/jspace_phase4_development.{tex,pdf}`
- Durable restart ledger:
  `reports/INPROGRESS_VM12_20260801.md`
- Compact restart handoff:
  `reports/RESUME_PHASE4_2.md`
- Governing development block: `jspace_lab_nextsteps_4_3.md` plus its
  accepted addendum in the campaign Drive folder

These documents now cover the base, 3.0 Think, and sibling 3.1
Think/Instruct capability and intervention points, seed-paired
own/common-frame audits, and the registered four-checkpoint trajectory
synthesis. Phase 4.2 adds the Qwen same-corpus convergence and functional
invariance gates, CPU-first common-cohort closure, Bank B/W authoring and
power, Bank W capability rules, official-mode gates, and P4-P2 design/pilot
methods. Phase 4.3 closes the A1000 instrument decision, side-track import,
and remaining freeze blockers. They remain development summaries, not frozen
claims.

## Tiers

- `phase2-confirmatory-import`
- `phase3-confirmatory-import`
- `phase3-replication-import`
- `side-development-import`
- `phase4-development`
- `phase4-confirmatory`
- `phase4-replication`
- `methods`

Native Phase 2/3 evidence creation is rejected. Imported events name and
hash their source registry, source commit/tag, evidence ID, and outputs.
