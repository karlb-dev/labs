# NEXTSTEPS_2_2 ACCEPTED — 2026-07-28

The principal investigator has approved the continuation plan in
`jspace_lab_nextsteps_2_2.md` (forensic branch review of
`interp_jspace_part2` @ `53532f8`/`d4f2c69` vs `main` @ `4097c44`) as
amended by `jspace_lab_nextsteps_2_2_addendum.md` (same directory).

Governing order where they conflict: **addendum §3 (PI decisions D1–D7)
governs; everywhere else nextsteps_2_2 is the operative repair spec.**

What this acceptance means:

- The end-of-VM7 pilot state is frozen as a historical checkpoint:
  tag `jspace-part2-pilot-vm7`, snapshot
  `../reports/PILOT_SNAPSHOT_VM7.json` (+ `_lenses.jsonl`).
- `SCIENTIFIC_PREREGISTRATION_DRAFT.md` is closed as
  `DRAFT_V2_PRE_REVIEW`; its six open decisions are resolved by the
  addendum (D1 split endpoint + protected-answer tail stratification;
  D2 run context-J study, mean-J stays replication primary, 3-estimator
  decomposition; D3 Gemma = methods boundary case; D4 freeze −1.0 nat;
  D5 expand to ≥60 families then family-split; D6 run G5, no waiver;
  D7 one paper, methods sections extractable).
- Execution proceeds through nextsteps stages N0–N8 with addendum §4
  amendments. Everything produced before the freeze is pilot/dev/methods
  tier.
- HARD STOP at stage N5: `SCIENTIFIC_PREREGISTRATION_CANDIDATE.md` is
  prepared and the campaign stops for explicit PI sign-off before the
  dedicated freeze commit. No confirmatory item generation, partition
  unveiling, or confirmatory cells before that commit.

Standing rules carried forward: withdrawal-and-supersession discipline;
the contradiction heuristic (a new measurement contradicting an
established result indicts the instrument first — now lab law, see
package README); evidence-tier vocabulary on everything.
