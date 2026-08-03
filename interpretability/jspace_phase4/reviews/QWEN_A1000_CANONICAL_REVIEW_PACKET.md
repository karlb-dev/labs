# Qwen A1000 canonical review packet

**EXTERNAL REVIEW COPY — IMPLEMENTATION AGENT HAS NOT APPROVED OR SIGNED**

State date: 2026-08-03. No confirmatory or replication intervention outcome
was opened. The review object is the mechanical development decision, not a
request to select a more favorable lens.

## Bound transaction chain

| Transaction | Event / commit | Bound result |
|---|---|---|
| A1000 cumulative fit | `p4-qwen-lens-fit-drawA-n1000-dev-v1` | Lens `6e48c773...f6bd6`; checkpoint `fd5a4ae...bf20`; exact 1,000-prompt cumulative draw A. |
| Structural A500--A1000 | `p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1` | Result `eaf8a63e...`; task q50/q05 0.998702/0.998122, both pass. |
| Functional A500--A1000 | `p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1`, `603bbcec...` | Result `7625ae1f...`, manifest `cddb0502...`; ID Jaccard 0.538462, projector overlap 0.709818, bridge-rescue difference -0.294028 nat fail. |
| All-position margin audit | `p4-qwen-selection-margin-a500-a1000-dev-v1`, `0ce3519b...` | Result `8a85a715...`; 17,381 positions, 15,536 near-tie, 1,845 stable-core, zero rank-deficient; stable-core Jaccard 0.666667. |
| Prompt-323 retained influence | `p4-qwen-lens-influence-prompt323-dev-v1`, `01236a3...` | Result `a5359b02...`; all A500/A1000 materiality metrics negligible; current-runtime repeat difference 0.004572 <= 0.5. |
| Canonical decision | `p4-qwen-canonical-lens-decision-a1000-dev-v1`, `28e25fe...` | Result `549a3f42...`; **Q-L4; no canonical sparse lens**. |

The prompt-323 transaction follows the prospective amendment committed at
`92831d31...`. Its current-runtime primary maximum is 181.776618 and its
discarded repeat maximum is 181.777423. The historical 173.345 fit-log value
is a non-gating reproducibility mismatch because the old record lacks exact
distribution-content and compiled-kernel identities. The influence result may
be cited only as an exact-pinned current-runtime sensitivity shape.

## Frozen-gate interpretation

Structural convergence passes, but both sparse-selection gates and the
load-bearing bridge-rescue causal endpoint fail. Prompt-323 influence is too
small to make that decision sensitive: the closest frozen metric is more than
3,800-fold below its threshold. The mechanical Q-L table therefore emits Q-L4
without reviewer discretion. Q-L4 blocks one canonical lens and every
Q-L1/Q-L2-dependent Phase 4 intervention.

Informational trend context, not a gate input:

| Rung | Projector overlap | ID Jaccard | Bridge-rescue status |
|---|---:|---:|---|
| A120--A250 | 0.67479 | 0.53846 | Selection failure; historical state partly non-durable. |
| A250--A500 | 0.70302 | 0.53846 | Difference 0.55891 nat fails 0.25; span-safe-specific difference -0.02432 with non-TOST interval. |
| A500--A1000 | 0.709818 | 0.538462 | Difference -0.294028 nat fails. |

The third 0.53846 Jaccard is treated as a repeated structural fact about
near-tied sparse rows, not as grounds to soften a frozen threshold.

## External reviewer checklist

- [ ] Verify event ancestry, result hashes, and registered output hashes.
- [ ] Verify the prospective runtime amendment predates the influence event.
- [ ] Verify historical-runtime reproducibility is not claimed.
- [ ] Verify Q-L4 is reproduced mechanically from unchanged thresholds.
- [ ] Verify no canonical lens, A2000, pilot, or untouched outcome exists.
- [ ] Verify the no-primary paper route follows without multiplicity transfer.

Independent reviewer: **EXTERNAL SIGNATURE REQUIRED**

PI disposition: **EXTERNAL SIGNATURE REQUIRED**
