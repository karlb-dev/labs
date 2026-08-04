# Phase 4 freeze verification (post-merge)

Generated: `2026-08-04T19:16:47Z` at `9a4d43eb60eb0af30d15cade538c95e18af3b744` on `interp_jspace_part2`.

| Check | Result |
|---|---|
| phase4 suite | `302 passed in 9.54s` |
| gemma suite | `71 passed, 1 warning in 19.90s` |
| olmo_lineage suite | `90 passed in 2.71s` |
| gemma live evidence | ok (23 events / 71 outputs) |
| olmo_lineage live evidence | ok (37 events / 163 outputs) |
| phase4 live evidence | 520/521; only failure is the known permanent deficit |
| durability | 521/522 verified; 1 known deficit; 0 unexpected |
| git diff --check | clean |
| large untracked artifacts | none |

Per the accepted Phase 4.5 addendum, the governed pre-canonical handout boundary lifts at the tag; handout regeneration belongs to the paper-analysis phase. Registered TeX/PDF/figure bytes are hash-verified by the durability pass rather than rebuilt here.
