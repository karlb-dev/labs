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

## Current development synthesis

- Living Markdown: `reports/PHASE4_DEVELOPMENT_REPORT.md`
- Compiled handout:
  `reports/handout/jspace_phase4_development.{tex,pdf}`
- Durable restart ledger:
  `../jspace_phase3/reports/INPROGRESS_VM11_20260731.md`

These documents now cover the base, 3.0 Think, and sibling 3.1
Think/Instruct capability and intervention points, seed-paired
own/common-frame audits, and the registered four-checkpoint trajectory
synthesis. They remain development summaries, not frozen claims.

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
