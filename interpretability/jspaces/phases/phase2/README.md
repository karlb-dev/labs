# jspace_part2 — the repair-era package (in git, repro-pack native)

The confirmatory J-space Part-2 harness. Everything scientific produced
from here follows [`protocol/REPRO_CONTRACT.md`](protocol/REPRO_CONTRACT.md):
one registered evidence id per claim-bearing artifact, provenance block in
every result file, and single-command reproduction:

```bash
git clone git@github.com:karlb-dev/labs.git && cd labs
bash interpretability/jspaces/phases/phase2/repro.sh                    # install + conformance tests + env audit
bash interpretability/jspaces/phases/phase2/repro.sh <evidence-id>      # verify/re-run one evidence item
jspace-part2 registry-list                                     # what evidence exists, tiers, commits
```

Governing science docs (mirrored at `../jspace/part2/code/`):
`jspace_part2_plan1_addendum.md` (forensic review) →
`REPAIR_PREREGISTRATION.md` (Workstream R + gates) → `PLAN_PART2.md`
(REVISION 1). Part-1 corrections: `../jspace/REPORT_v2_ERRATA.md`.

## Layout

```
jspace_part2/         the package
  lib.py              conformance utilities (rank-safe SVD bases, output-
                      protected selection, phase-controlled ablator,
                      mergeable moments, full-sequence scoring, paired
                      cluster bootstrap, equivalence, nested bases, hashing)
                      — 27 CPU self-tests: `jspace-part2 selftest`
  provenance.py       git/model/lens pinning, provenance blocks, evidence registry
  inventory.py        R0: SHA-256 inventory of all run-dir artifacts (resumable)
  audit_env.py        environment audit + pip-freeze lock
  __main__.py         CLI: audit-env · inventory · selftest · registry-list · repro
tests/                test_lib.py (the conformance suite)
protocol/             REPRO_CONTRACT.md (+ specs/crosswalk as R1-R5 land)
configs/              typed YAML per model/experiment (grows with R1+)
reports/              evidence_registry.jsonl (append-only, in git)
repro.sh              the single-command entry
```

## Status

- Era: **Workstream R (assay repair)** — no confirmatory model-matrix cells
  until gates G0–G6 pass (`REPAIR_PREREGISTRATION.md`).
- Done: conformance utilities + tests (R3/R4 cores), provenance/registry
  (R0 core), environment audit; artifact inventory running.
- Next: R1 output-protected dynamic ablation module + adapter interface,
  R2 paper-occupancy estimator + solver validation, experiment runner with
  manifest enforcement, tiny-model golden tests (R6).
- Exploratory era (`../special_lab2/`, gitignored; mirrored at
  `../jspace/part2/`) is frozen for new science; its two banked results are
  registered as exploratory evidence. The 120-prompt Olmo-3.1-32B-Instruct
  recipient lens fit currently running from there remains a valid input
  (first nested point of the fit-size curve; hash-pinned on completion).

## Ops invariants

Preemptible 24 h GPU blocks: every GPU phase checkpoints at ≤10-min
granularity and resumes with the same command; the live queue
(`MyDrive/interpret/inprogress.md`) is ordered so any prefix fits a block.
