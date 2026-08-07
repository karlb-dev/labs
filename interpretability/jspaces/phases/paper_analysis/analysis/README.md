# jspace_paper/analysis — offline paper-analysis workspace

Post-Phase-4 frozen-evidence synthesis. Branch
`interp_jspace_paper_analysis` from tag `jspace-phase4-frozen-v1`.
CPU-only; every campaign registry and registered output is a read-only
input. Governing plan: `paper_analysis.md` + `paper_analysis_addendum.md`
(Drive: `MyDrive/interpret/`); entry points
`jspace_phase4/FREEZE_HANDOFF.md` and
`jspace_phase4/reports/PHASE4_TO_PAPER_ANALYSIS_HANDOFF.md`.

```text
ANALYSIS_FOUNDATION.json   source tag/commit/registry hashes (verified)
protocol/                  P0 frozen protocol (5 files)
manifests/                 campaign artifact + paper-source inventories
data/                      canonical parquet tables (schemas in protocol/)
scripts/                   deterministic builders; python scripts/<name>.py
reports/                   analysis reports incl. analysis_events.jsonl
paper_routes/              route outlines, claim tables, figure plans
figures/  tables/          regenerated only from data/ via scripts/
tests/                     invariant checks for the analysis tables
```

Rules that bind everything here: analysis-first (no paper sentence before
its reconstructed number); exactly one primary tier per row; missing ≠
gated ≠ not-applicable ≠ zero; retrospective outputs labeled and
event-logged; no forward/backward pass.
