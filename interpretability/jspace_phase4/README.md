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

The Qwen Draw A fit resumes from the highest contract-matched local/Drive
checkpoint and refuses CPU or slow-kernel fallback:

```bash
export JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731
export JSPACE4_LOCAL_WORK=/content/sl4_work
export HF_HUB_CACHE=/content/hf_local
python -m jspace_phase4.experiments.p4_qwen_nested_lens_fit \
  --config interpretability/jspace_phase4/configs/p4_qwen_nested_lens_fit_dev.yaml \
  --draw draw_a --stop-at 250
```

In a second host-visible process, arm the non-mutating Drive-backed watchdog:

```bash
bash interpretability/jspace_phase4/watchdog_phase4.sh \
  /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json \
  250
```

The CPU-first common-cohort sensitivity can run while the GPU fit is active:

```bash
MPLCONFIGDIR=/tmp/matplotlib-phase4 \
python -m jspace_phase4.experiments.p4_lineage_common_cohort_analysis \
  --config interpretability/jspace_phase4/configs/p4_lineage_common_cohort_olmo_dev.yaml
```

## Current development synthesis

- Living Markdown: `reports/PHASE4_DEVELOPMENT_REPORT.md`
- Compiled handout:
  `reports/handout/jspace_phase4_development.{tex,pdf}`
- Durable restart ledger:
  `../jspace_phase3/reports/INPROGRESS_VM11_20260731.md`
- Governing development block:
  `reviews/jspace_lab_nextsteps_4_2.md` plus its addendum

These documents now cover the base, 3.0 Think, and sibling 3.1
Think/Instruct capability and intervention points, seed-paired
own/common-frame audits, and the registered four-checkpoint trajectory
synthesis. Phase 4.2 adds the Qwen same-corpus convergence and functional
invariance gates, CPU-first common-cohort closure, Bank B/W authoring, and
candidate-preregistration repairs. They remain development summaries, not
frozen claims.

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
