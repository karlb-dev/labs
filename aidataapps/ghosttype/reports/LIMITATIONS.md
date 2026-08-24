# GhostType LIMITATIONS (Tier 1)

run: `ghosttype-cpu-dev-20260823T213801Z` · generated 2026-08-24T16:09:28.978Z · lab: aidataapps/ghosttype (Mac local campaign)

> Regenerated from GhostTypeControl by `npm run reports`. The database is the source of record.

- **Mac campaign, not the spec's GPU serving plane.** Single-resident MLX serving, sequential replay; no multi-user serving results exist yet (Colab lift-and-shift is the second campaign).
- **SQL timings are DEV-tier** (SQL Server 2025 under Rosetta emulation, outside Microsoft's support boundary).
- **Simulated acceptance is not observed user acceptance** (claim ceiling, addendum §7).
- **Power labels are provisional** until the E-1 power simulation runs; paired-test labels use a discordant-count rule of thumb.
- **8 package defects** are adjudicated findings (3 invalid golds, 5 cursor placements); their rows still score, so exact-match ceilings are slightly depressed for affected families until dataset v2.1.
- **The M-arm matrix is reduced**: the mac campaign replays the package's frozen prompt (arm M1-packaged). M0/M2/R0 prompt constructions and retrieval arms are not yet run.
- **B4's identifier-safe adaptation** is reduced to same-catalog exemplar reuse; cross-catalog adaptation is unimplemented.
- **Execution/compile oracles have not run**; parse and static gates only. compile_eligible/execution_eligible flags are stored and waiting.
