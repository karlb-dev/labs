# OLMo lineage live in-progress state

Last updated: 2026-08-02T01:21:00Z

This is the volatile restart pointer for the OLMo-only parallel workstream.
The stable recovery procedure is `OLMO_LINEAGE_RESUME.md`; the scientific
narrative is `OLMO_LINEAGE_DEVELOPMENT_REPORT.md`. Canonical heavy outputs
live in Drive under `olmo_lineage_20260801`.

## Current durable state

- Branch: `interp_jspace_olmo_lineage`.
- Remote tracking branch: `origin/interp_jspace_olmo_lineage`.
- Branch parent before this package: `4ea7a9ba7a534daa61e0d8c9960763b921a1b80b`.
- Immutable scientific import boundary:
  `3b041735d8b842de46a9c0a474fccd0c44e0841a`.
- Foundation source commit:
  `dc5f62336ac83481f09dafebaad666a93157f812`.
- Durable Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801`.
- Registry:
  `interpretability/jspace_olmo_lineage/reports/evidence_events.jsonl`.
- Active model job: none; both OLMo baseline jobs finished and GPU memory is
  free.
- Last registered native evidence: `ol-bank-w-capability-joint-dev-v1`
  (development), created at 2026-08-02T01:20:37Z from clean commit `241be17`.
  Its new registry line is awaiting the immediate Git checkpoint below.
- Both OLMo Bank-W baseline capability outcomes and the joint analysis are
  open. No Bank-W intervention, confirmatory, or replication outcome has been
  opened.

## Work currently in progress

The isolated foundation is complete. Four immutable Drive manifests are
registered and verify cleanly: environment lock, import manifest, foundation
conformance, and foundation manifest. The import manifest validates 33 named
source events across Part 2, Phase 3, and Phase 4; 13 direct artifacts; three
code dependencies; and six governance documents. Fourteen conformance tests
pass. No model job is active.

Both OLMo models completed 384/384 unique rows with all 4,608 checked numeric
values finite per model and eight candidate-sequence scores per row. Think
low/high accuracy is 0.7135417/0.7187500 with high-low 0.0052083 and 90% CI
[-0.03125, 0.0416667]. Instruct low/high is 0.7395833/0.7187500 with high-low
-0.0208333 and 90% CI [-0.0572917, 0.0208333]. Both pass independently and
each has 17/24 capable families.

The exact Think/Instruct/Qwen common intersection is 16 families, below the
prospective minimum of 20. `olmo_phase4_service_ready=false`; the source Phase
4 aggregation and stricter side service decision both say BLOCKED. The next
action is to publish this joint event and immediately emit the required early
hash-pinned Phase 4 import bundle. No Bank-W intervention is authorized on the
failed service set. O2/O3 can proceed while any Bank-W redesign is handled
prospectively.

## Exact next actions

From `/content/labs`:

```bash
git switch interp_jspace_olmo_lineage
git status --short --branch
python -m pip install -q -e interpretability/jspace_part2
python -m pip install -q -e interpretability/jspace_phase3
python -m pip install -q -e interpretability/jspace_phase4
python -m pip install -q -e interpretability/jspace_olmo_lineage
python -m pytest interpretability/jspace_olmo_lineage/tests -q
```

Commit and publish the joint registry/report checkpoint, mirror it, then emit
the early import bundle from the resulting clean tree:

```bash
git add interpretability/jspace_olmo_lineage
git commit -m 'olmo: register joint Bank-W capability decision'
git pull --rebase origin interp_jspace_olmo_lineage
bash interpretability/jspace_olmo_lineage/repro.sh
git push origin interp_jspace_olmo_lineage
python -m jspace_olmo_lineage.recovery
python -m jspace_olmo_lineage.experiments.bank_w_capability \
  --config interpretability/jspace_olmo_lineage/configs/ol_bank_w_capability_v1.yaml \
  --emit-early-bundle
```

After registering and publishing the early bundle, update/mirror the reports
and move to O2 symmetric capacity plus the O3 lens-provenance audit. Do not
open an O4 Bank-W intervention under the failed 20-family protocol.

## Hardware and weight state at last update

- GPU observed: NVIDIA RTX PRO 6000 Blackwell Server Edition, approximately
  96 GiB VRAM, CUDA working.
- OLMo-3.1 Think and Instruct snapshots are complete in the Drive HF cache.
- The exact Think snapshot is also complete on local NVMe. Its result is
  durable in Drive, the side registry, and GitHub; GPU memory is free.
- The exact Instruct snapshot is complete on local NVMe after a direct pinned
  Hub download; all 14 shard sizes match the Drive copy. Its baseline event is
  published in registry commit `241be172ee75f453eca7de23244a5c531bd9e1b0`.
- OLMo-3 Think is also present in Drive.
- Base is not complete and must be downloaded later for O2.
- Never load model weights through DriveFS. Copy one pinned snapshot at a time
  to local NVMe and load from there.

## Safety boundary

Do not write Phase 3, Phase 4, Gemma, or main-paper registries/run roots. Do
not open untouched Phase 4 confirmatory or replication intervention outcomes.
The OLMo O1 outputs are baseline capability only and exist to service the
parallel Phase 4 Bank-W design.
