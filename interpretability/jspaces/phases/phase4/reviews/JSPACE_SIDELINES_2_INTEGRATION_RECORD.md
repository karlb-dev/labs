# J-space sidelines block 2 — integration record

Integrated: `2026-08-03T16:04:20Z`. Target branch:
`interp_jspace_part2`. Evidence tier remains development/methods throughout.

## Outcome

The outstanding Phase-4.4 mainline branch and both Study-2 side branches were
merged with ancestry preserved. Phase 4 advanced once during integration, so
the exact merge order was:

```text
interp_jspace_phase4_4 through the registered runtime boundary
→ interp_jspace_gemma_transport_2
→ interp_jspace_olmo_lineage_2
→ the prospective Phase-4.4 runtime-contract amendment
```

Gemma precedes OLMo among the side branches, preserving the registered
dependency from the G2.1 backend ceiling to the OLMo H6 transport validation.
No branch was squashed. No native `gm2-*` or `ol2-*` event was added to the
Phase-4 registry.

## Ancestry

| Role | Commit |
|---|---|
| Shared parent | `901fb4fc7578a913088c7947a2e6240f7fc45aeb` |
| Phase-4.4 runtime-boundary branch point | `22f6801a750374fa6735420029b57c35471fa3f8` |
| Phase-4.4 initial merge | `82fbcdd6c1d674154dda94031994c66f8c9fc2b7` |
| Gemma Study-2 terminal branch | `e07880084b33fe0f998968bcd5d2394e2ae6465f` |
| Gemma merge | `319c8e0dae9e0e4f1bcdd0df3805a48c11e5d5bc` |
| Append-only import compatibility | `9b9dae4972568cf0cc3b3c2c6996af0448b2bbb4` |
| OLMo Study-2 terminal branch | `9159baff66402f33ee7a907e10acac748a3e6114` |
| OLMo merge | `0118e50bd07236e23681998b5fa9534c860ca85a` |
| OLMo merged-worktree verifier compatibility | `8a7e3de49d55b7703e9a2f4955cd74a2f8856238` |
| Gemma merged-worktree verifier compatibility | `e6e2b4f7e4b33d4894db9a22a00e94a5636c51e8` |
| Phase-4.4 terminal branch | `92831d31e37e76a50b573a99c9f19bf55005531c` |
| Phase-4.4 prospective-amendment merge | `db802baf6a709087d01f9e7d54180b0a27dac3cc` |

The merge commits have the expected second parents, so both isolated histories
remain directly reachable from `interp_jspace_part2`.

## Registry isolation

| Registry | Shared-parent SHA-256 | Integrated SHA-256 | Integrated state |
|---|---|---|---|
| Phase 4 | `831395cfef107c6af454816a21efa732024058f69e046ea35a09cc2f48b918c3` | `8986df09179ef41b36456064e1464bce6427e5559f8fc9427055fb259610c8e4` | 71 lines; 61 origins; 54 live; 284 live outputs |
| Gemma | `acd78247f3e367a1fb81693b7c7b811f98b7a7d166cd3d8a5e46a1deff409173` | `c7e76cac59bbfcf30f5d29edfa5c747bdae9aea46523fad9a8bd49db9cca124b` | 24 lines; 23 origins/live; 71 live outputs |
| OLMo | `7cb7ece414092722e925c4fadecdc3f7aade00190848e6a13f74ff77f69d4307` | `d15391ac733d785f583c62e812688efc56268c70ad9aff430b50bb49741196a6` | 41 lines; 38 origins; 37 live; 163 live outputs |

An exact namespace scan finds zero `gm2-*` or `ol2-*` evidence IDs in the
Phase-4 registry. No Study-2 side result was promoted into Phase-4 evidence,
and no Phase-4 import event was created during integration.

## Exact side bundles

### Gemma

- Terminal event: `gm2-sidelines2-import-bundle-v1`.
- Bundle JSON SHA-256:
  `9ef48b8ab1d99d52a756ddea1e285a9d61e781fd054cbf92702bfe81be56f5b0`.
- Bundle Markdown SHA-256:
  `547da552fee0057fa304b1dffe657eb269315007073b0a5c3cbb29c96577b315`.
- Frozen registry-prefix SHA-256:
  `2a144bcf0e7be0ac4307f7e2a2984c1879340b9a7e9278d10143d122a14fd30a`.
- Prefix boundary: 36,279 bytes, 23 events, through
  `gm2-stage1-relicense-v1`.

### OLMo

- Terminal event: `ol2-sidelines2-import-bundle-v1`.
- Bundle JSON SHA-256:
  `c213dc74aa78dcd6613c8bd1562dd07d2e2a0345409ee6da585001693d8e6b1c`.
- Bundle Markdown SHA-256:
  `9e1d756c7b167e0cef356083e99f667182de3fa50cc5b81c9fd98a8845a1e989`.
- Frozen registry-prefix SHA-256:
  `0a8973e01d562a82fa88da650ab8597c140050f6caf46c8bbd72e2b58acffb58`.
- Prefix boundary: 86,627 bytes, 40 events, through
  `ol2-bank-w-olmo-pair-power-v1`.

The bundle files and their frozen prefix snapshots retain their branch-release
hashes after integration. Drive release copies were not rewritten.

## Integration-only compatibility changes

Three non-scientific compatibility changes were required after the clean
merges:

1. Phase-4 side-import validation now accepts the exact registry bytes tracked
   at an import bundle's pinned source commit when the live append-only source
   registry has later, valid appended events. The frozen hash must still match;
   mutation, an unsafe path, or an unavailable source snapshot still fails.
2. The OLMo live-evidence verifier can materialize a registered absolute
   producer-worktree repo path from the identical merged repository file while
   retaining the OLMo package/run-root isolation check.
3. The Gemma live-evidence verifier applies the same exact-hash
   producer-worktree materialization.

No registry row, bundle, result, threshold, paper source, or scientific output
was edited by these changes. Regression tests cover valid append-only registry
growth, unsafe/hash-drift rejection through the existing suite, repository
materialization, and preservation of external paths.

## Verification

| Scope | Result |
|---|---|
| Phase-4 full suite | 285 passed |
| Gemma full suite | 71 passed; one pre-existing Torch warning |
| OLMo full suite | 89 passed |
| Gemma live evidence | 23 live events / 71 outputs; zero failures |
| OLMo live evidence | 37 live events / 163 outputs; zero failures |
| Phase-4.4 newly merged result outputs | 52/52 strict SHA-256 checks pass |
| Phase-4 side-import and namespace checks | included in the 285-test suite; pass |
| Phase-4 amended-runtime inventory/shape contract | included in the 285-test suite; pass |
| `git diff --check` | pass |

The Phase-4.4 branch's registered durability boundary remains 230/232 with its
two explicitly known historical deficits and zero unexpected failures. The
integration separately rehashed every output of the two newly merged
functional and selection-margin events: 18 files / 303,287,915 bytes and 34
files / 14,411,038 bytes, respectively. A redundant full historical Drive
rehash was not used as a merge gate because it blocks for long periods on the
largest DriveFS objects; no new Phase-4 artifact was inferred from that choice.

## Paper and figure outputs

### Gemma

- TeX SHA-256:
  `afaf90af4f3b96f5e1b267607e52839725a38cbf25b4713e343863d0383187e2`.
- Deterministic three-page PDF SHA-256:
  `518b1fac1997469e364ebe641fcc40b99b513c0a57144a47668793a3882fd5c9`.
- Calibration PNG SHA-256:
  `9c21f13c90a8b9d3d0325a699e025f1267dde8548bc3e74eb6f592ff0bb773ff`.

The parent-worktree two-pass build reproduces the registered PDF hash.

### OLMo

- TeX SHA-256:
  `5783560a9938061efce9f47df85cde7bb021584a1bcef406598fda0763a8cc51`.
- Deterministic 16-page PDF SHA-256:
  `7809b76ab4cec90a5878c3542167eadb83399c6a9f3567fe28bc93db47e2f809`.
- Stage-wedge PNG SHA-256:
  `add4b84aa436779d9a4b491a726d6116783b4c76c5e74107c9537ded33a9273d`.
- H6 PNG SHA-256:
  `03564b351088aac7bda3dc9a4c00fa02f07456543349766e5afdd4e39e8ca7b1`.
- Pair-power PNG SHA-256:
  `476ea88627f1d710b03e1bb8a310f54c46683af337980b90890b0ab57128364d`.

The parent-worktree deterministic build reproduces the registered PDF hash.

### Phase 4

The Phase-4.4 handoff explicitly forbids regenerating the compiled handout
while the canonical event is absent. Integration preserves its branch bytes:

- TeX SHA-256:
  `7745b4f104c4373cf05476232b2ea92d056a0f9d46a5319bf3a7fde4a64dc917`.
- PDF SHA-256:
  `43b5ca44be660642bdeb1059059547aa0bf896b1abf7f71cdc83f2fe534c7391`.

Both hashes match the exact terminal `92831d3` branch objects. The prospective
runtime-contract amendment is SHA-256
`bac298d527f364b302f74a5e6ccf526a369e522b0658ce59bbd660a8c43dac2b`.

## Scientific boundaries retained

- Phase 4 now has a prospectively frozen, exact-content current-runtime shape
  contract, recorded before any new prompt-323 contribution or influence
  outcome. It is ready for a fresh attempt but has not run. Q-L4 remains
  provisional, no canonical lens event exists, and M2/M3/M4 remain closed.
- Gemma's five-layer classifier is a scoped finite-scale methods result; it is
  not a nondifferentiability, missing-information, named-mechanism, workspace,
  intervention, or confirmatory result.
- OLMo stage effects remain missing rather than zero; H6 fails in-band with a
  checkpoint-specific late anchor; registered-dose coverage is unavailable;
  and the OLMo-pair redesign remains underpowered at current support.
- No intervention, confirmatory/replication cell, independent-review field, or
  PI sign-off was opened by integration.

## Release check

This record is the final pre-push integration state. After committing it,
`interp_jspace_part2` is pushed and fetched back, with local/remote equality
required. The shared Drive handoff records the resulting exact pushed head.
