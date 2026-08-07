# Phase 4 permanent-deficit review packet

**ONE PERMANENT HISTORICAL DEFICIT — EXTERNAL ACCEPT/REJECT DECISION
REQUIRED BEFORE FREEZE**

Assembled: 2026-08-04, Phase 4.5 closeout block. Sources:
`reviews/A120_A250_STATE_ARCHIVAL_DISPOSITION.md`,
`reviews/A120_STATE_EXACT_COPY_SEARCH_20260802.md` (SHA-256
`4dd156eb2bb7f55a55e6facc2daea30f0d952ebcb9a4e8ab4052e473725bae1c`),
`reports/PHASE4_PART4_A120_CAPACITY_RECOVERY.json`, the append-only event
`p4-qwen-a120-a250-state-permanent-deficit-v1`, and the fresh pass
`reports/PHASE4_PART5_DURABILITY_FRESH.{json,md}`.

## 1. Original evidence event and output role

Event `p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1`
(phase4-development tier) registered 17 outputs. The missing object is its
operational state file:

```text
.../metrics/qwen36-27b-multilens-functional-gate/functional_gate/
p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1/state.json
SHA-256 361bda08e9ffbe1d333fd3cfaf3c7b9545e6a3504246a16dd8b0c07ad26f45e8
```

Its role was operational (timing and peak-allocation bookkeeping for the
A120–A250 functional gate run), not a scientific result table. The gate's
decision role is carried by the surviving scientific payloads below and was
superseded by the later live A250–A500 and A500–A1000 functional gates.

## 2. Exhaustive negative search

`A120_STATE_EXACT_COPY_SEARCH_20260802.md` records the exhausted surfaces:
mounted Drive, live and trashed Drive objects, Drive revisions, preserved
pre-incident DriveFS metadata, pending operations, mirror databases,
repository trees, and available local staging/backup roots. No exact byte
string or recoverable cloud object was found. The route "restore a genuinely
exact backup" remains admissible if the exact bytes are ever discovered and
the full hash matches.

## 3. Why synthesis is invalid

The file contains wall-clock timing and peak-allocation fields that cannot
be derived exactly from any surviving payload. A schema-compatible surrogate
would necessarily fabricate operational history under a registered hash it
cannot reproduce. Forbidden routes remain forbidden: synthesizing timestamps
or allocation fields, editing the old registry row, copying another run's
state, or describing a compatible reconstruction as exact recovery.

## 4. Superseding events

The deficit event classifies the source as **partially durable, role
superseded** by:

- `p4-qwen-multilens-functional-gate-a250-a500-published-dev-v1`
- `p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1`

No current canonical-lens decision depends on treating the missing
operational state as present; the mechanical Q-L4 chain reads only the
registered structural, functional (A500–A1000), selection-margin, and
influence gates.

## 5. Exact hashes of the sixteen surviving outputs

| Output | SHA-256 |
|---|---|
| functional_gate_result.json | `dd18fb85c5609fec...` |
| input_manifest.json | `e33799ebfb5e5440...` |
| fixed_endpoint_activations.pt | `f8a8bba270e2aec8...` |
| fixed_capacity_activations.pt | `fbb84c37ad09b412...` |
| selection_pair_rows.parquet | `acf482f23a6577c9...` |
| primary_rows.parquet | `ee902838ebb334d1...` |
| selection_rows.parquet | `c9ddfbea68949014...` |
| readout_rows.parquet | `26347782db3b05c7...` |
| prose_rows.parquet | `59ec1f918633a73a...` |
| bridge_rows.parquet | `c77870a2925e7139...` |
| g4_rows.parquet | `8a8edf4bd4ef3e1c...` |
| capacity_reconstructions_published.pt | `ebe0c641115b40db...` |
| capacity_reconstructions_a120.pt | `6b0399df2c57158e...` (exact recovery, 2026-08-03) |
| capacity_reconstructions_a250.pt | `d908cb39e0e36396...` |
| p4f13_qwen_multilens_functional_gate.png | `4696552188cc7098...` |
| p4f13_qwen_multilens_functional_gate.pdf | `70a1a6e40dede6b7...` |

Full 64-hex values are in the registry row and in
`PHASE4_PART5_DURABILITY_FRESH.json`; every one of these sixteen verifies on
the fresh materialization.

## 6. Fresh-materialization proof

The independent pass `phase4-part5-fresh-materialization` (fresh VM, fresh
mount, fresh clone; 2026-08-04) verified 520/521 live references with this
one deficit as the only failure, zero unexpected deficits, zero wrong
hashes, zero pin conflicts, and zero path-policy violations.

## 7. Risk statement

Accepting release with this deficit means the A120–A250 functional gate is
citable only as a partially durable historical event: its scientific tables
and figures reproduce, but its exact operational timing/allocation record
does not. The risk is bounded because (i) the event's decision role is
superseded by fully durable later gates, (ii) no terminal Phase 4 claim
reads the missing file, and (iii) the deficit is declared in the release
manifest and known-limitations file rather than averaged away. Rejecting
release on this ground would block the freeze without any recoverable
remedy, since the exact-copy search is exhausted.

## 8. Decision fields

- Independent reviewer decision on the permanent-deficit disposition
  (ACCEPT WITH EXPLICIT LIMITATION / REJECT): **pending — see
  `reviews/PHASE4_INDEPENDENT_REVIEW_20260804.md`**
- PI decision on release with the explicit limitation
  (ACCEPT / REJECT): **pending — see
  `reviews/PHASE4_PI_DISPOSITION_20260804.md`**

Neither field may be filled by the implementation agent that assembled this
packet.
