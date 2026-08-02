# OLMo lineage live in-progress state

Last updated: 2026-08-02T01:25:00Z

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
- Last registered native evidence: `ol-phase4-early-import-bundle-v1`
  (methods), created at 2026-08-02T01:22:48Z from clean commit `dc20c90`.
  Its new registry line and these report updates await the immediate Git
  checkpoint.
- Both OLMo Bank-W baseline capability outcomes and the joint analysis are
  open. No Bank-W intervention, confirmatory, or replication outcome has been
  opened.

## Work currently in progress

The isolated foundation and O1 service obligation are complete. Four immutable
foundation manifests and six live evidence events verify cleanly; 19 package
tests pass. No model job is active.

Both OLMo models completed 384/384 unique rows with all 4,608 checked numeric
values finite per model and eight candidate-sequence scores per row. Think
low/high accuracy is 0.7135417/0.7187500 with high-low 0.0052083 and 90% CI
[-0.03125, 0.0416667]. Instruct low/high is 0.7395833/0.7187500 with high-low
-0.0208333 and 90% CI [-0.0572917, 0.0208333]. Both pass independently and
each has 17/24 capable families.

The exact Think/Instruct/Qwen common intersection is 16 families, below the
prospective minimum of 20. `olmo_phase4_service_ready=false`; the source Phase
4 aggregation and stricter side service decision both say BLOCKED. The
required early import bundle is complete at Drive
`release/IMPORT_BUNDLE_PHASE4_EARLY.json` (SHA-256
`debb29ef67ffa8741a4971ec2b0b21340bd5b48dc5729ac12f74f78839bf4f2b`)
with its readable companion (SHA-256
`a7e6faf9ad412bd965cfbc9f7b1e9e98c9194cc5019e669d0695b5852a55159d`).
Its embedded registry prefix covers the first 8,651 bytes through the joint
event and hashes to
`dcaca5a819a070f006a8534b820bfd476e0ebe63cd0b583412bfbfd050a79f10`.
Envelope and live-prefix verification pass. No Bank-W intervention is
authorized on this failed service set. O2 symmetric capacity and O3 lens
provenance are now the active work; any Bank-W redesign must be a separate,
prospectively frozen protocol.

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

If this event/report checkpoint is still dirty after a reclaim, publish and
mirror it first:

```bash
git add interpretability/jspace_olmo_lineage
git commit -m 'olmo: publish early Phase 4 capability bundle'
git pull --rebase origin interp_jspace_olmo_lineage
bash interpretability/jspace_olmo_lineage/repro.sh
git push origin interp_jspace_olmo_lineage
python -m jspace_olmo_lineage.recovery
```

Then implement and run the O3 four-lens provenance audit and the O2 symmetric
capacity estimator/corpus freeze. O3 is audit-first: do not refit a lens unless
the audit establishes that the existing artifacts cannot support the planned
comparisons. O2 includes Base even though Base capability is not required.
Do not open an O4 Bank-W intervention under the failed 20-family protocol.

## Hardware and weight state at last update

- GPU observed: NVIDIA RTX PRO 6000 Blackwell Server Edition, approximately
  96 GiB VRAM, CUDA working.
- OLMo-3.1 Think and Instruct snapshots are complete in the Drive HF cache.
- The exact Think snapshot is also complete on local NVMe. Its result is
  durable in Drive, the side registry, and GitHub; GPU memory is free.
- The exact Instruct snapshot is complete on local NVMe after a direct pinned
  Hub download; all 14 shard sizes match the Drive copy. Its baseline event is
  published in registry commit `241be172ee75f453eca7de23244a5c531bd9e1b0`.
- Direct pinned Hub staging was substantially faster than DriveFS for Instruct
  and is the preferred staging path while authentication/network remain healthy;
  the complete Drive snapshots remain recovery fallbacks.
- OLMo-3 Think is also present in Drive.
- Base is not complete and must be downloaded later for O2.
- Never load model weights through DriveFS. Copy one pinned snapshot at a time
  to local NVMe and load from there.

## Safety boundary

Do not write Phase 3, Phase 4, Gemma, or main-paper registries/run roots. Do
not open untouched Phase 4 confirmatory or replication intervention outcomes.
The OLMo O1 outputs are baseline capability only and exist to service the
parallel Phase 4 Bank-W design.
