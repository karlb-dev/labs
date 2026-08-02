# OLMo lineage live in-progress state

Last updated: 2026-08-02T00:30:51Z

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
- Active model job: none.
- Last registered native evidence: `ol-foundation-v1` (methods), created at
  2026-08-02T00:30:35Z from clean commit `dc5f623`.
- No OLMo Bank-W capability or intervention outcome has been opened on this
  side track yet.

## Work currently in progress

The isolated foundation is complete. Four immutable Drive manifests are
registered and verify cleanly: environment lock, import manifest, foundation
conformance, and foundation manifest. The import manifest validates 33 named
source events across Part 2, Phase 3, and Phase 4; 13 direct artifacts; three
code dependencies; and six governance documents. Fourteen conformance tests
pass. No model job is active.

The current work is the O1 producer/config/conformance implementation. The
next model-backed checkpoint is OLMo-3.1 Think Bank-W baseline capability,
then OLMo-3.1 Instruct, then joint support and the early Phase 4 import bundle.

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

First commit and publish the newly appended foundation registry line, then
mirror the recovery documents:

```bash
git add interpretability/jspace_olmo_lineage
git commit -m 'olmo: register isolated study foundation'
git pull --rebase origin interp_jspace_olmo_lineage
bash interpretability/jspace_olmo_lineage/repro.sh
git push origin interp_jspace_olmo_lineage
python -m jspace_olmo_lineage.recovery
```

Then implement and freeze the O1 compatibility tests and protocol config from
the exact pinned Phase 4 scoring contract. Do not start a 32B run from a dirty
tree. The first experiment is OLMo-3.1 Think Bank-W baseline capability,
followed by Instruct. Each model has 384 rows and no intervention condition.

## Hardware and weight state at last update

- GPU observed: NVIDIA RTX PRO 6000 Blackwell Server Edition, approximately
  96 GiB VRAM, CUDA working.
- OLMo-3.1 Think and Instruct snapshots are complete in the Drive HF cache.
- OLMo-3 Think is also present in Drive.
- Base is not complete and must be downloaded later for O2.
- Never load model weights through DriveFS. Copy one pinned snapshot at a time
  to local NVMe and load from there.

## Safety boundary

Do not write Phase 3, Phase 4, Gemma, or main-paper registries/run roots. Do
not open untouched Phase 4 confirmatory or replication intervention outcomes.
The OLMo O1 outputs are baseline capability only and exist to service the
parallel Phase 4 Bank-W design.
