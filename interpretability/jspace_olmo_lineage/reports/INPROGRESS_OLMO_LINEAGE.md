# OLMo lineage live in-progress state

Last updated: 2026-08-02T00:27:29Z

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
- Current remote tip: `e01959e83db04a06b296d8bc383366ad1b33e736`.
- Durable Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801`.
- Registry:
  `interpretability/jspace_olmo_lineage/reports/evidence_events.jsonl`.
- Active model job: none.
- Last registered native evidence: none; foundation is being prepared.
- No OLMo Bank-W capability or intervention outcome has been opened on this
  side track yet.

## Work currently in progress

The isolated package, prospective preregistration, import manifest config,
registry implementation, path guards, 13 conformance tests, development
report, and recovery documents are committed and pushed. The first foundation
attempt correctly refused a preregistration hash mismatch introduced by final
whitespace normalization; it created no output file or registry event. The
hash pin is now corrected. The next durable checkpoint is
`ol-foundation-v1`.

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

After committing the corrected hash and pulling/rebasing before pushing, from
a clean tree create the foundation:

```bash
python -m jspace_olmo_lineage.experiments.foundation \
  --config interpretability/jspace_olmo_lineage/configs/ol_foundation_v1.yaml
```

Commit the new append-only foundation registry event, pull/rebase, test, push,
and mirror these reports. The next experiment is O1, OLMo-3.1 Think Bank-W
baseline capability, followed by Instruct. Each model has 384 rows and no
intervention condition.

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
