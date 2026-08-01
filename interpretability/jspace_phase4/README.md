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

## Phase 4.2 resumable work block

The Qwen Draw A continuation resumes from the highest contract-matched
local/Drive checkpoint and refuses CPU or slow-kernel fallback. The durable
wrapper also locks the run, writes a heartbeat/log, and banks the registry
append on completion:

```bash
export JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731
export JSPACE4_LOCAL_WORK=/content/sl4_work
export HF_HUB_CACHE=/content/hf_local
bash interpretability/jspace_phase4/run_qwen_continuation_fit.sh draw_a_n500
```

Do not launch a duplicate while `QWEN_CONTINUATION.lock` is held. Read the
dynamic handoff before recovery because it records any temporary storage
routing needed by the current VM:

```bash
cat /content/resume-phase-4-2.md
cat /content/drive/MyDrive/interpret/inprogress.md
```

After `p4-qwen-lens-fit-drawA-n500-dev-v1` is registered, merge the isolated
preparation branch, replace only `PENDING_REGISTERED_A500_SHA256` in the two
successor YAMLs with the event's exact lens hash, commit that binding, and run:

```bash
bash interpretability/jspace_phase4/run_qwen_a500_postfit_queue.sh
```

That queue prospectively runs and banks A250--A500 structural convergence,
the fixed functional gate, the official mode-v2 model baseline, and Qwen Bank
W baseline capability. Follow its registered A/B/C branch without changing
thresholds: A/C continues with `draw_b_n120`; B continues with `draw_a_n1000`.

The lower-level producer remains available for exact recovery diagnostics:

```bash
python -m jspace_phase4.experiments.p4_qwen_nested_lens_fit \
  --config interpretability/jspace_phase4/configs/p4_qwen_nested_lens_fit_dev.yaml \
  --draw draw_a --stop-at 500
```

The n=250 convergence, retained prompt-112 influence, functional Branch-B
gate, mode-parser v2 contract, Bank B feasibility, Bank W capability protocol,
and conditional P4-P2 variance-pilot protocol are already registered. Do not
rerun or overwrite them; registry verification checks their immutable hashes.

## Current development synthesis

- Living Markdown: `reports/PHASE4_DEVELOPMENT_REPORT.md`
- Compiled handout:
  `reports/handout/jspace_phase4_development.{tex,pdf}`
- Durable restart ledger:
  `reports/INPROGRESS_VM12_20260801.md`
- Compact restart handoff:
  `reports/RESUME_PHASE4_2.md`
- Governing development block:
  `reviews/jspace_lab_nextsteps_4_2.md` plus its addendum

These documents now cover the base, 3.0 Think, and sibling 3.1
Think/Instruct capability and intervention points, seed-paired
own/common-frame audits, and the registered four-checkpoint trajectory
synthesis. Phase 4.2 adds the Qwen same-corpus convergence and functional
invariance gates, CPU-first common-cohort closure, Bank B/W authoring and
power, Bank W capability rules, official-mode gates, and P4-P2 design/pilot
methods. They remain development summaries, not frozen claims.

## Tiers

- `phase2-confirmatory-import`
- `phase3-confirmatory-import`
- `phase3-replication-import`
- `phase4-development`
- `phase4-confirmatory`
- `phase4-replication`
- `methods`

Native Phase 2/3 evidence creation is rejected. Imported events name and
hash their source registry, source commit/tag, evidence ID, and outputs.
