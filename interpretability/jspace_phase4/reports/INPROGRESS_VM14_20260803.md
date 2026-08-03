# IN PROGRESS — Phase 4.4 closeout, VM14

Updated: 2026-08-03 16:55 UTC. Phase 4 remains development-only and is not
frozen.

## Recoverable boundary

- Working branch: `interp_jspace_phase4_4`, pushed through durability commit
  `411b3cbf10dc3ed6b97a9baed61ede65d9cd1fe1` before paper closeout.
- Parent: clean `interp_jspace_part2` commit `901fb4fc7578a913088c7947a2e6240f7fc45aeb`.
- Terminal-B precommit: `6f23a298...`, before functional execution.
- Full Phase 4 suite: 284/284 after the prospective runtime amendment.
- GPU work is complete; no model process is active.

## Completed scientific queue

- Functional A500--A1000: registered `603bbcec...`; provisional Q-L4.
- Selection margin: registered `0ce3519b...`; 17,381/17,381 positions retained.
- Prospective prompt-323 runtime amendment: `92831d3...`; historical mismatch
  retained as a non-gating reproducibility limitation.
- Prompt-323 influence: registered `01236a3...`; current-runtime repeat gate
  passes and all A500/A1000 materiality metrics are negligible.
- Canonical decision: registered `28e25fe...`; **Q-L4, no canonical sparse
  lens**. P4-P2 and the Bank-B orthogonal shot are not applicable; M3/M4 did
  not open.
- M2 imports/replay: OLMo early `bcf537e...`, Bank W joint replay
  `fddab68...`, Gemma `6ac62d2...`, OLMo final `a8e218c...`. P4-P3 remains
  blocked at 16/20 common-capable families.
- Optional M5 did not run because no retained-extremes producer/configuration
  was prospectively frozen before Q-L4.

## Durability boundary

- Exact A120 capacity artifact recovered at registered SHA-256
  `6b0399df...c651b6f` and backed up.
- Missing historical A120--A250 state classified append-only at
  `e282879...` as a permanent known deficit; external signatures remain
  required.
- Two same-mounted passes agree: 78 rows, 61 live events, 419 references, 418
  verified, one known deficit, zero unexpected deficits, zero pin conflicts.
- Pre-freeze inventory payload `0fbd2d4d...` is mechanically
  `NOT_REVIEW_READY` solely because the independent fresh-remount proof cannot
  be supplied by this mounted VM.

## Closeout completed

The terminal paper/TeX and review packet were committed and pushed on
`interp_jspace_phase4_4` at `0f3380d580ba5f78c87d4b00adb7906f3c2ad747`.
Nineteen closeout files plus a complete verified Git bundle are mirrored under
Drive `closeout/phase4_4_20260803_0f3380d/`. The branch was merged with
`--no-ff` into the then-current `interp_jspace_part2` (including the already
landed sidelines-2 block) at
`a63d49c1879f888893e9d005bdace1e46bcdc603`.

Post-merge Phase 4 passes 285/285 tests and OLMo passes 89/89. Gemma's union
suite exposed one integration-test boundary error: the prefix test used the
later registry containing the release event itself rather than the immutable
pre-release prefix snapshot. The narrow test repair anchors it to the exact
registered snapshot; the source producer and release remain byte-unchanged.
Targeted and full Gemma suites pass. No scientific result, registry row, or
source release artifact is changed by that repair.

The only remaining actions are repository push verification and external
fresh-remount/reviewer/PI work. Do not freeze/tag or self-sign.

No confirmatory or replication intervention outcome exists. No A2000, new
bank, model, endpoint, or SESOI change exists.
