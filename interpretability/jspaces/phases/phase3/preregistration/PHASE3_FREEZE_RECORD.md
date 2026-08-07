# Phase 3 immutable freeze record

This file clarifies the already-valid Phase 3 freeze without modifying its
historical preregistration or tag.

## Binding freeze

- Freeze tag:
  [`jspace-phase3-freeze-v1`](https://github.com/karlb-dev/labs/tree/jspace-phase3-freeze-v1/interpretability/jspaces/phases/phase3)
- Freeze commit:
  [`df4d45ac91b908d6009dfd6632ab1ce386df7a9a`](https://github.com/karlb-dev/labs/commit/df4d45ac91b908d6009dfd6632ab1ce386df7a9a)
- Parent commit:
  `7d0fc5c6383f6e9bc8fe9b18121126d9981a8f2c`
- Derived partition seed:
  `int("7d0fc5c6", 16) % 100000 = 85670`
- Exact gate event ID: `p3-partition-freeze-v1`
- Frozen P3-P3 model decision: `qwen36-27b`

## Frozen artifacts and gates

- Partition file SHA-256:
  `27a177ffa28085b32916b63ffab35286debfc391fec7fb6d7de07459a8a8769d`
- Partition payload SHA-256:
  `15724b6f7ea3862e0bb699fc2ac0e6c19649ff449a8ec1d419b6bb0e5842bf27`
- Frozen preregistration SHA-256:
  `0a83a9318d45199a9ccd3f0291a889511e70182d2534a909e897b08c021a18b1`
- Family counts: 36 confirmatory and 36 replication.
- Bank F two-hop-family counts: 24 confirmatory and 24 replication.
- Intersection-family counts: 17 confirmatory and 17 replication.

The partition artifact and its registered freeze event are authoritative. The
freeze commit subject says “intersection 16/18”; that subject-line count is a
clerical defect. The artifact committed by the tag, its payload, and the gate
event all record the actual 17/17 allocation, above.

## Historical-document clarification

The frozen preregistration still contains the header “CANDIDATE - NOT FROZEN”
and unchecked generation-time checklist boxes. `freeze_phase3.py` renamed the
candidate without rewriting its body after all executable gates passed. This is
a document-generation defect, not evidence that the freeze failed. The
historical file and `jspace-phase3-freeze-v1` remain immutable.

The current clarification is
[`PHASE3_FREEZE_RECORD.md`](https://github.com/karlb-dev/labs/blob/interp_jspace_part2/interpretability/jspaces/phases/phase3/preregistration/PHASE3_FREEZE_RECORD.md).
The separate
[`jspace-phase3-pre-release-audit-v1`](https://github.com/karlb-dev/labs/tree/jspace-phase3-pre-release-audit-v1/interpretability/jspaces/phases/phase3)
tag preserves the state immediately before release-audit implementation.

