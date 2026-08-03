# Gemma transport Study 2 — live handoff

Status: G2.1 and G2.2 complete and registered; V2 documentation/handout
complete; exact Study-2 import bundle not yet rendered. No further model
compute is required.

- Parent: `901fb4fc7578a913088c7947a2e6240f7fc45aeb`
- Branch: `interp_jspace_gemma_transport_2`
- Worktree: `/content/labs_gemma2`
- Drive root: `/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_2_20260803`
- Latest scientific registry event: `gm2-stage1-relicense-v1`
- Latest registered evidence commit: `029ba95`
- Required next event: `gm2-sidelines2-import-bundle-v1`

## Complete evidence

G2.1 contains 232 pair summaries and 1,008 rows: 216 frozen full exact-backend
pairs / 936 full rows plus 72 nested-operation rows. Every value is finite;
all primals, deterministic replays, pair reconstructions, dtype-quantum
reconstructions, row-order audit, and the required fresh-process replay pass.
The target firewall held.

The target-blind route is `benign_scheduling_floor`. The pooled disagreement
q99 is `0.026234563004519824`; the frozen formula yields ceiling
`0.07870368901355948`, with prompt-bootstrap 90% interval
`[0.07489779624371865, 0.10392247147209241]`. The architecture, batch, and
path-ambiguity routes are inactive. Threshold SHA-256:
`a6dc1e2a963c21a16f477f23af7260359b2337ebab47f2d5b1ff35112e0c9515`.

G2.2 opened the historical outcome only after G2.1 registered. The exact
all-slot error `0.0024581113830208778` lies below the ceiling and the selected
scientific slot remains bit-identical. It selected
`branch_1_relicense_without_recompute`; no model compute occurred. The
unchanged L22/L30/L37/L44/L52 `local_tangent_mismatch` classifier is a closed
exact-JVP finite-scale methods result over the tested scope. Decision SHA-256:
`22b090e02909ad4dfbfb44707463bf021c0a02e7629024f015f053a72849d58c`.

The exact OLMo-control ceiling export is licensed and has already been used by
the downstream OLMo H6 lane.

## Documentation state

The Study-2 report, V2 state of record, V2 claim ledger, V2 gate protocol, and
three-page TeX/PDF handout are complete. The registered calibration PNG is
included mechanically. The PDF was compiled twice with
`SOURCE_DATE_EPOCH=1785715200`; both builds have SHA-256
`518b1fac1997469e364ebe641fcc40b99b513c0a57144a47668793a3882fd5c9` and were
visually inspected page by page.

## Recovery and next actions

Require a clean, pushed branch and set:

```bash
export JSPACE_GEMMA_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_2_20260803
```

Then run the full package tests, freeze an exact pre-release registry prefix,
render and commit the partial-safe import bundle, publish all release artifacts
to the Drive root, register `gm2-sidelines2-import-bundle-v1`, verify every
repo/Drive hash, and push. Preserve all Study-1 files and events unchanged.

Claim boundary: development/methods only. No nondifferentiability, Jacobian
absence, mechanism, missing-information, workspace, confirmatory, replication,
independent-review, or PI-sign-off claim is licensed.
