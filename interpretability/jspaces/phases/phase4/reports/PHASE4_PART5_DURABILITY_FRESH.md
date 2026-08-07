# Phase 4.5 fresh-materialization durability proof

**INDEPENDENT FRESH-MOUNT PASS — ONE KNOWN PERMANENT DEFICIT — ZERO
UNEXPECTED DISCREPANCIES**

Pass label: `phase4-part5-fresh-materialization`, 2026-08-04.
Payload: `reports/PHASE4_PART5_DURABILITY_FRESH.json`.
Registry: `20124c4c95c3c179472db86f87144bdeee4aaf252a4302408e96f01266c2a00c`
(84 rows, 64 live events, 521 live output references).

## Independence of this materialization

The Phase 4.4 record honestly reported that its two agreeing durability
passes shared one mounted Drive materialization (VM14) and could not supply
independent proof. This pass runs on a different, freshly booted VM
(2026-08-04) with a fresh Drive mount and a fresh clone of the repository
from origin; no Phase 4 producer ever ran on this machine. Mount identity
and clone provenance are recorded in
`reports/PHASE4_PART5_FOUNDATION.json` (event
`p4-phase4-part5-foundation-v1`). Every Drive read in this pass is this
mount's first materialization of the referenced bytes, independent of the
VM14 cache.

## Result

```text
raw_all_live_outputs_verified   = false   (one known permanent deficit)
policy_aware_release_set_verified = true  (pending the external
                                           reviewer/PI signatures below)
```

| Quantity | Value |
|---|---:|
| Live output references | 521 |
| Verified | 520 |
| — via literal registered path | 515 |
| — via identical tracked repository file | 3 |
| — via exact append-only registry prefix at the registering commit | 2 |
| Failures | 1 |
| Known deficits among failures | 1 |
| Unexpected deficits | 0 |
| Wrong hashes | 0 |
| Path pin conflicts | 0 |
| Temporary/recovery-path violations | 0 |
| Native side IDs in Phase 4 registry | 0 |

The single failure is the declared permanent historical deficit: the
A120–A250 operational `state.json`
(SHA-256 `361bda08e9ffbe1d333fd3cfaf3c7b9545e6a3504246a16dd8b0c07ad26f45e8`),
absent exactly as recorded by the append-only event
`p4-qwen-a120-a250-state-permanent-deficit-v1`. The exact recovered A120
capacity tensor `capacity_reconstructions_a120.pt`
(`6b0399df2c57158e7fdad24274e50f8c1058021d233412afdcc5177f6c651b6f`)
verifies on this fresh mount, so the earlier second historical deficit is
resolved and appears in `resolved_known_deficits`.

## Non-literal resolutions (strict, hash-exact)

Two reference classes required the strict resolver added at commit
`0fc9a18` (recorded as a launch deviation in the Part-5 foundation):

1. Three `p4-bank-w-capability-joint-imported-dev-v1` outputs were
   registered under the producing VM's worktree prefix
   (`/content/labs_phase4_4/...`). The identically named tracked repository
   files hash byte-exactly to the registered pins.
2. The two Study-1 side-import events pinned whole-file hashes of the side
   registries, which legitimately grew (append-only) at the sidelines-2
   merge. The exact pinned bytes are recovered from the immutable tracked
   snapshots at each event's registering commit, and each live registry is
   a byte-prefix extension of them.

No hash was relaxed and no reference was skipped.

## Policy-aware release condition

`policy_aware_release_set_verified` becomes final only when, per
`jspace_lab_nextsteps_4_5.md` §5.2:

- [x] the fresh materialization reproduces the same one known deficit;
- [x] every other reference verifies;
- [x] the append-only deficit event is exact and unedited;
- [ ] an independent reviewer accepts the permanent-deficit disposition;
- [ ] the PI accepts the release with that explicit limitation.

The two open boxes are external-signature gates carried by
`reviews/PHASE4_PERMANENT_DEFICIT_REVIEW_PACKET.md`.

The companion mechanical inventory
(`manifests/phase4_pre_freeze_inventory_v4_5.{json,md}`, payload
`71ae60318128d4c279323bf9cfd06df7442426620dd536c55b63cbb1bc1bad61`) is
honestly `NOT_REVIEW_READY` solely because
`all_live_outputs_verified=false` under the known deficit; all other gates
pass.
