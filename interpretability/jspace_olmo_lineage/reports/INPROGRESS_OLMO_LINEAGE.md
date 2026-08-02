# OLMo lineage live in-progress state

Last updated: 2026-08-02T00:46:00Z

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

The O1 producer, isolated config, NVMe staging runner, and compatibility tests
are implemented but not yet committed. Nineteen tests pass. The O1 config is
exactly equal to the registered Phase 4 source for selection, answer/scoring
contract, capability guard, model/tokenizer revisions, answer token IDs, and
model order. The wrapper imports the hash-pinned Phase 4 scorer, analyzer, and
state implementation directly, checks their installed module paths, adds a
384-row/no-drop/all-finite/eight-sequence gate, and writes only OLMo outputs.
No model baseline or intervention outcome has been opened.

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

Commit and publish the O1 implementation, then freeze its outcome-blind side
protocol from a clean tree:

```bash
git add interpretability/jspace_olmo_lineage
git commit -m 'olmo: add exact Bank-W capability service'
git pull --rebase origin interp_jspace_olmo_lineage
bash interpretability/jspace_olmo_lineage/repro.sh
git push origin interp_jspace_olmo_lineage
python -m jspace_olmo_lineage.experiments.bank_w_capability \
  --config interpretability/jspace_olmo_lineage/configs/ol_bank_w_capability_v1.yaml \
  --freeze-protocol
```

Commit/pull/rebase/test/push the new protocol event. Update and mirror this
file before starting the first long job. Then run OLMo-3.1 Think with
`run_bank_w_capability_model.sh olmo31-think`. The script stages the exact
Drive snapshot onto local NVMe, checkpoints every eight rows to Drive, and
pulls/rebases before automatically publishing the registry event. Do not start
the 32B run from a dirty tree.

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
