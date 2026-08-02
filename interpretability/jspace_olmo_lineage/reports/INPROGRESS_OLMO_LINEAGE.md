# OLMo lineage live in-progress state

Last updated: 2026-08-02T01:11:37Z

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
- Active model job: none; Think finished, verified, committed, and pushed.
- Last registered native evidence:
  `ol-bank-w-capability-olmo31-think-dev-v1` (development), created at
  2026-08-02T01:10:17Z from clean commit `663e49f` and published in registry
  commit `b35adae138492c9224d6f82fa0ae2b3b3e125680`.
- One OLMo Bank-W baseline capability outcome is open. No Bank-W intervention,
  confirmatory, or replication outcome has been opened.

## Work currently in progress

The isolated foundation is complete. Four immutable Drive manifests are
registered and verify cleanly: environment lock, import manifest, foundation
conformance, and foundation manifest. The import manifest validates 33 named
source events across Part 2, Phase 3, and Phase 4; 13 direct artifacts; three
code dependencies; and six governance documents. Fourteen conformance tests
pass. No model job is active.

Think O1 completed 384/384 unique rows, with all 4,608 numeric endpoints finite
and eight candidate-sequence scores per row. Low/high accuracy is
0.7135417/0.7187500. The paired high-minus-low estimate is 0.0052083 with
family-bootstrap 90% CI [-0.03125, 0.0416667]. Both aggregate accuracy floors
and load equivalence pass, so Think is independently capability-eligible.
Seventeen of 24 families meet the per-family 0.70 floor at both loads.

Because Think has only 17 capable families, the prospectively required
three-way Think/Instruct/Qwen intersection cannot reach 20. This makes the
eventual joint service decision mathematically blocked regardless of the
Instruct outcome. Instruct must still run and be reported; no intervention is
authorized on the failed service set. The next action is the Instruct baseline.

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

Commit and publish this report checkpoint, mirror it, then stage Instruct.
The prior Drive copy took about 15 minutes; a direct exact-revision Hugging
Face download may be used if faster. In either case, the scientific run command
is:

```bash
git add interpretability/jspace_olmo_lineage
git commit -m 'docs: record OLMo Think capability result'
git pull --rebase origin interp_jspace_olmo_lineage
bash interpretability/jspace_olmo_lineage/repro.sh
git push origin interp_jspace_olmo_lineage
python -m jspace_olmo_lineage.recovery
bash interpretability/jspace_olmo_lineage/run_bank_w_capability_model.sh \
  olmo31-instruct
```

The Instruct log is
`logs/bank_w_capability_olmo31-instruct_20260802.log`, its watchdog is in the
same directory, and its resumable state is
`metrics/olmo31-instruct/bank_w_capability/ol-bank-w-capability-olmo31-instruct-dev-v1/state.json`
under the OLMo Drive root. If state exists, the same shell command verifies the
input header and resumes missing rows. Do not start the 32B run from a dirty
tree.

## Hardware and weight state at last update

- GPU observed: NVIDIA RTX PRO 6000 Blackwell Server Edition, approximately
  96 GiB VRAM, CUDA working.
- OLMo-3.1 Think and Instruct snapshots are complete in the Drive HF cache.
- The exact Think snapshot is also complete on local NVMe. Its result is
  durable in Drive, the side registry, and GitHub; GPU memory is free.
- OLMo-3 Think is also present in Drive.
- Base is not complete and must be downloaded later for O2.
- Never load model weights through DriveFS. Copy one pinned snapshot at a time
  to local NVMe and load from there.

## Safety boundary

Do not write Phase 3, Phase 4, Gemma, or main-paper registries/run roots. Do
not open untouched Phase 4 confirmatory or replication intervention outcomes.
The OLMo O1 outputs are baseline capability only and exist to service the
parallel Phase 4 Bank-W design.
