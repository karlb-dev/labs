# Phase 4 freeze verification

Generated: `2026-08-04T19:01:41Z` at `826036fd37992c199237e53acc68877ee9663e37` on `interp_jspace_phase4_5`.

| Check | Result |
|---|---|
| phase4 suite | `302 passed in 22.42s` |
| gemma suite | `71 passed, 1 warning in 27.64s` |
| olmo_lineage suite | `90 passed in 17.83s` |
| gemma live evidence | ok (23 events / 71 outputs) |
| olmo_lineage live evidence | ok (37 events / 163 outputs) |
| phase4 live evidence | 520/521; only failure is the known permanent deficit |
| durability | 521/522 verified; 1 known deficit; 0 unexpected |
| git diff --check | clean |
| large untracked artifacts | none |

Per the accepted Phase 4.5 addendum, the governed pre-canonical handout boundary lifts at the tag; handout regeneration belongs to the paper-analysis phase. Registered TeX/PDF/figure bytes are hash-verified by the durability pass rather than rebuilt here.
