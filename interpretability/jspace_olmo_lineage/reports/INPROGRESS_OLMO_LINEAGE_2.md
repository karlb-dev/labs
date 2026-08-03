# OLMo lineage Study 2 — terminal handoff

Status: all mandatory scientific stages, CPU pair-power closure, reports,
deterministic paper, exact import bundle, Drive publication, and terminal
registration are complete. Only parent-branch integration remains.

- Parent at fork: `901fb4fc7578a913088c7947a2e6240f7fc45aeb`
- Branch: `interp_jspace_olmo_lineage_2`
- Worktree: `/content/labs_olmo2`
- Drive root: `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_2_20260803`
- Terminal event: `ol2-sidelines2-import-bundle-v1`
- Bundle source commit: `80213290125a56ad75bd9a23a638211a0dc1c618`
- Release-event producer commit: `c37e19de51a44ba3c646f9f3d0e40300e90bbdf6`
- Registry-event commit: `a1099589e7b5b1dadcc9e90b05ad7ea663d260e8`

## Scientific disposition

Think-SFT and Think-DPO each produced 972 exact-generation capability rows.
Capability rates are 0.00617 and 0.00309. Neither checkpoint has a Bank-S
fact capable on both direct and composed variants, so both prospective
intervention cohorts are empty. Across paired items, 965 are incapable at both
stages, two capable at both, four lose capability at DPO, and one Bank-F item
appears at DPO. The registered route is `null_or_unresolved`; effects are
missing, not zero, and Tier 2 is forbidden.

Base and OLMo-3.1 Think each produced 336 H6 rows. All 672 exact-backend checks
pass the imported Gemma G2.1 ceiling `0.07870368901355948`. No L24/L32/L40
cell passes. Base reaches 9/12 only at the L56, epsilon-0.10 late anchor; Think
passes 12/12 only there. The joint route is
`h6_fail_in_band_with_checkpoint_specific_late_anchor`.

The registered-dose audit finds no exact frozen site-dose joins among six
relevant tables, so coverage is unavailable/null, not zero. The outcome-blind
OLMo-pair power closure finds 16 shared capable families, power 0.7788 versus
the frozen 0.80 target, and a minimum of 18. Its route is
`not-powered-at-current-support`. No Bank-W intervention outcome was opened.

## Exact release

- Bundle ID: `jspace-olmo-lineage-sidelines2-v1`
- JSON SHA-256: `c213dc74aa78dcd6613c8bd1562dd07d2e2a0345409ee6da585001693d8e6b1c`
- Markdown SHA-256: `9e1d756c7b167e0cef356083e99f667182de3fa50cc5b81c9fd98a8845a1e989`
- Payload SHA-256: `b5cfdf92ffc309cb15d2f9fadfc87a0b86fc52f646a3fe891d56b4cedba05be3`
- Pre-release registry prefix: 86,627 bytes / 40 lines / 37 origins / 36
  live events / 145 live outputs; SHA-256
  `0a8973e01d562a82fa88da650ab8597c140050f6caf46c8bbd72e2b58acffb58`.
- Release artifacts: 12 pinned sources plus bundle JSON/Markdown/prefix; 18
  repo/Drive outputs registered and verified.
- Paper TeX SHA-256:
  `5783560a9938061efce9f47df85cde7bb021584a1bcef406598fda0763a8cc51`.
- Paper PDF SHA-256:
  `7809b76ab4cec90a5878c3542167eadb83399c6a9f3567fe28bc93db47e2f809`.
- Pair-power PNG SHA-256:
  `476ea88627f1d710b03e1bb8a310f54c46683af337980b90890b0ab57128364d`.

The 16-page paper was compiled deterministically and visually inspected. All
release files are copied under the Study-2 Drive root's `release/` directory
and verify byte-identically against the registered Git sources.

## Integration

Fetch the current `interp_jspace_part2`, require a clean source worktree, and
merge `interp_jspace_gemma_transport_2` first. Merge this branch second with
ancestry preserved. Then run the union test suites and paper checks, write the
joint sidelines integration record, and push the refreshed parent.
