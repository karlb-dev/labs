# Phase 4.4 execution record

**DEVELOPMENT-ONLY DECISION BLOCK — NOT A FREEZE RECORD**

State date: 2026-08-03. This record stops at the assembled freeze-review
packet. No independent reviewer or PI field is self-signed, and no freeze
commit or tag is created.

## Launch and Terminal-B commitment

The isolated `interp_jspace_phase4_4` branch was created from clean, freshly
pulled `interp_jspace_part2` commit
`901fb4fc7578a913088c7947a2e6240f7fc45aeb`. The required two-paragraph
Terminal-B success-state commitment was committed and pushed at
`6f23a29896e61cc367ed884a2d840f0e08857f40`, before the A1000 functional
outcome was opened.

## M0 and sealed A1000 queue

The corrected frozen-package invocation passed 279/279 tests before
execution. The pinned Qwen snapshot at revision `6a9e13bd...25e8644`, all 23
files and 55,563,006,400 weight bytes, all 48 fused linear-attention bindings,
the external published lens at `a4114d7...`, and the registered A1000 lens,
checkpoint, and header passed exact identity and tensor audits. The model was
downloaded directly from Hugging Face rather than copied from Drive.

The functional observer initially stopped before evidence because a diagnostic
`topk(32)[:10]` did not reproduce the inherited `topk(10)` at an exact score
tie. The repair retained an exact separate top-10 replay and allowed prefix-ID
difference only at the tied boundary. The full suite then passed 282/282; the
partial incompatible state was retained only as an unregistered incident
artifact.

The repaired functional transaction registered
`p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1` at
`603bbcec...`. Structural q50/q05 are 0.998702/0.998122, but selected-ID
Jaccard 0.538462, normalized projector overlap 0.709818, and bridge-rescue
difference -0.294028 nat fail their frozen gates. The subsequent registered
selection-margin audit at `0ce3519b...` retained all 17,381 positions: 15,536
near-tie, 1,845 stable-core, and zero rank-deficient. It exactly reconstructed
the intervention and did not rescue sparse-selection stability.

## Prospective runtime amendment and retained-prompt result

The first prompt-323 attempt stopped before writing evidence because current
runtime Jacobian norms disagreed with the historical fit-log control. Two
current prompt-323 evaluations were mutually stable but differed from 173.345;
the registered prompt-112 control independently showed a larger historical
versus current mismatch. Nominal software versions, model, corpus, bindings,
and lens revision matched, while the historical record lacked distribution-
content and compiled-kernel identities. No cause beyond that provenance limit
is asserted.

PI direction authorized a prospective amendment because retained-prompt
influence cannot change Q-L1--Q-L5. Before any new influence output, the exact
current distribution contents, primary computation, repeatability gate, and
historical-runtime limitation were fixed in
`PROMPT323_RUNTIME_CONTRACT_AMENDMENT.md`, committed and pushed at
`92831d31e37e76a50b573a99c9f19bf55005531c`. Banks, model, endpoints,
lenses, samples, thresholds, SESOIs, wording, and retention were unchanged.

The amended transaction registered
`p4-qwen-lens-influence-prompt323-dev-v1` at
`01236a3f9246847ea9c58d06adaf7479c846661e`. Its 71 outputs and
6,610,029,624-byte local backup verify. Primary and discarded-repeat maxima
are 181.776618 and 181.777423; worst per-layer normalized-norm repeat
difference is 0.004572 against the unchanged 0.5 current-runtime ceiling. All
tensors are finite with sequence length 128 and 111 valid tokens. Every frozen
materiality metric is negligible: the closest is more than 3,800-fold below
its threshold. Influence is largest early and essentially zero through assay
layers 20--44. Figure `p4f29` was copied byte-for-byte into the report and
visually checked. This is a current-runtime sensitivity-shape result; it does
not establish historical-runtime reproducibility.

## Mechanical Q-L branch

The unchanged canonical producer registered
`p4-qwen-canonical-lens-decision-a1000-dev-v1` at
`28e25feb6acff1d247881ed9cc231060b12dd496`. It binds the structural result
`eaf8a63e...`, functional result `7625ae1f...`, functional manifest
`cddb0502...`, selection-margin result `8a85a715...`, and prompt-323 result
`a5359b02...`. The canonical result is Q-L4: no single sparse Qwen lens is
nominated. P4-P2 is therefore blocked by decision-sensitive causal endpoints.
The Bank-B orthogonal shot is also not applicable. No P4-P2 pilot,
confirmatory intervention, replication intervention, M3, or M4 ran.

## M2 admit-then-cite queue

The post-canonical queue ran in the frozen transactional order:

1. OLMo early Bank-W capability import
   `p4-import-olmo-bank-w-capability-v1` (`bcf537e...`).
2. Fresh registered mainline replay
   `p4-bank-w-capability-joint-imported-dev-v1` (`fddab68...`, producer
   materialized at `922ef98...`). Exact common-capable support is 16/20, so
   P4-P3 remains blocked.
3. Gemma terminal transport import `p4-import-gemma-transport-v1`
   (`6ac62d2...`), source `b0425a4...`, at side-development-import tier. It is
   a methods blocker: selected-slot tangents are bit-identical, but the frozen
   all-slot relative-error criterion fails (0.002458 versus 1e-5). It licenses
   no mechanism or intervention claim.
4. OLMo terminal lineage import `p4-import-olmo-lineage-final-v1`
   (`a8e218c...`), source `a28cdd5...`, at side-development-import tier. It
   preserves the 16/20 service block, descriptive O2/O3 development patterns,
   and the no-identifiable-O5-estimand disposition; it opens no intervention.

Bank W's load intervention is not lawfully runnable as a cross-model Phase 4
primary at current capability support; it routes to Phase 5B as per-model
estimation.

No native side-study event was created in the Phase 4 registry. M5 was not run:
the optional retained-extremes producer/configuration had not been frozen
prospectively before Q-L4 and may not be authored after the branch result.

## Durability and pre-freeze boundary

Exact A120 capacity recovery reproduced registered SHA-256
`6b0399df...c651b6f` and was backed up locally. The historical A120--A250
operational `state.json` remains irrecoverable. The append-only methods event
`p4-qwen-a120-a250-state-permanent-deficit-v1` at `e282879...` classifies the
source as partially durable with its gate role superseded by the later live
A250--A500 and A500--A1000 functional events. Its external-review and PI
fields remain explicitly unsigned.

Two same-mounted durability passes agree exactly: 78 registry rows, 61 live
events, 419 live output references, 418 verified, one known historical state
deficit, zero unexpected deficits, and zero pin conflicts. Pass 1 is
`38759c9d...`; pass 2 is `7643abae...`; the registry hash is
`fbb84008...`. The second pass is a same-mount repeat, not an independent
Drive rematerialization.

The required pre-freeze inventory payload is
`0fbd2d4dd4a61adffb721f2959d8fee138be9f03ea4c5856bb91c4b980a00736`.
It finds all commits reachable, exact policy, no native side IDs, no path
violations, no temporary/recovery leakage, and no pin conflict. Its honest
mechanical status is `NOT_REVIEW_READY` solely because 418/419 is not a clean
whole-registry rematerialization. The assembled review packet therefore stops
with the release gate red pending a fresh independent remount plus real
reviewer and PI signatures.

There is no A2000, new bank, model, endpoint, or SESOI change in this block.
