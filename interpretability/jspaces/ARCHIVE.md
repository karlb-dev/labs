# Historical trees and freeze provenance

## Historical material inside the phase trees

| Path | Origin | Status |
| --- | --- | --- |
| `phases/phase1/part2_exploratory/` | `interpretability/jspace/part2/` | Early Drive-mirrored Part 2 results/handout/scripts from the Part-1 era. **Not** the confirmatory package. Active Part 2 is `phases/phase2/`. |

Ops queue scripts for Phase 4 remain under `phases/phase4/run_*.sh` (still useful for resume books). They are not science API.

## Freeze tags as provenance anchors

Git tags below mark **immutable scientific boundaries**. They were created
**before** this monorepo path reorg. Checking out a tag still shows the old
`interpretability/jspace_*` layout. The current tree is a living mirror of
that code with updated paths.

| Tag | Role |
| --- | --- |
| `jspace-part2-confirmatory-freeze-v1` | Part 2 confirmatory freeze |
| `jspace-part2-complete-v1` | Part 2 complete |
| `jspace-phase3-freeze-v1` | Phase 3 design/partition freeze |
| `jspace-phase3-pre-release-audit-v1` | Phase 3 pre-release audit |
| `jspace-phase3-complete-v1` | Phase 3 complete / Phase 4 import parent |
| `jspace-phase4-frozen-v1` | Phase 4 campaign freeze; paper-analysis parent |

**Do not rewrite** append-only registry rows (`evidence_events.jsonl`) to
change historical command strings or absolute Drive output paths. Artifact
**hashes** are the scientific identity. Path resolvers accept old
`repo://interpretability/jspace_*` prefixes via an alias table so tools can
still locate files after the move.

## Pre-reorg boundary tag (required)

`pre-jspaces-reorg-v1` is the last commit with the old layout: use it for
trivial checkout of frozen-era paths. It is **load-bearing**, not optional —
the study-2 release verifiers compare their renderer/config against it for
pre-reorg bundles, and the reorg validation restored frozen artifacts from
it. Keep it pushed wherever the branch goes.
