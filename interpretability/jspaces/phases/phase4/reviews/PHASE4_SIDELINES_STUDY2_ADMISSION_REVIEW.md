# Phase 4 Study-2 sidelines admission review packet

**METHODS-ONLY ADMISSION — TIERS AND FORBIDDEN USES PRESERVED**

Assembled: 2026-08-04. Under review: the two Phase 4.5 closeout import
events and their claim boundaries.

## Events

| Event | Source | Bundle SHA-256 | Frozen prefix SHA-256 |
|---|---|---|---|
| `p4-import-gemma-transport-study2-v1` | `jspace-gemma-transport-study2` @ `aba2e01460dd...`, branch `interp_jspace_gemma_transport_2` | `9ef48b8ab1d99d52a756ddea1e285a9d61e781fd054cbf92702bfe81be56f5b0` | `2a144bcf0e7be0ac4307f7e2a2984c1879340b9a7e9278d10143d122a14fd30a` |
| `p4-import-olmo-lineage-study2-v1` | `jspace-olmo-lineage` @ `80213290125a...`, branch `interp_jspace_olmo_lineage_2` | `c213dc74aa78dcd6613c8bd1562dd07d2e2a0345409ee6da585001693d8e6b1c` | `0a8973e01d562a82fa88da650ab8597c140050f6caf46c8bbd72e2b58acffb58` |

Importer: `experiments/p4_import_sidelines_study2.py` with configs
`configs/p4_import_gemma_transport_study2.yaml` and
`configs/p4_import_olmo_lineage_study2.yaml`; saved validations
`reports/gemma_transport_study2_import_validation_v1.json` and
`reports/olmo_lineage_study2_import_validation_v1.json`; acceptance tests in
`tests/test_sidelines2_imports.py`.

## What the reviewer must verify

1. Both bundle files hash to the pins above; both payload envelopes are
   self-consistent (`payload_sha256` equals the canonical-JSON hash).
2. Both frozen prefix snapshots hash exactly, and each live side registry
   still begins byte-exactly with its frozen prefix.
3. Every admitted event keeps a native tier in {development, methods}; no
   admitted event is superseded or withdrawn inside its frozen prefix.
4. The Phase 4 registry contains no native `gm-*`, `gm2-*`, `ol-*`, or
   `ol2-*` origin ID.
5. The Study-1 imports (`p4-import-gemma-transport-v1`,
   `p4-import-olmo-lineage-final-v1`) remain live and unmodified; the
   Study-2 events depend on them rather than replacing them.
6. The imported meanings do not exceed the source claim ledgers
   (`jspace_gemma/release/gemma_transport_claim_ledger_v2.md`,
   `jspace_olmo_lineage/reports/OLMO_LINEAGE_CLAIMS_TABLE_V2.md`).

## Imported meaning (Gemma)

The prior 1e-5 parity blocker remains historically correct under its own
contract. A later target-blind calibration froze an all-batches
mixed-precision ceiling of 0.07870368901355948; the preserved all-slot error
0.0024581113830208778 lies below it and the selected slot is bit-identical.
The unchanged five-layer local_tangent_mismatch classifier is therefore a
closed exact-JVP finite-scale methods result over the tested scope.

Forbidden uses: not confirmatory; not replication; not a Gemma workspace
result; not a late-layer intervention license; not proof of curvature as the
unique cause; not permission to reopen G2–G5 inside Phase 4.

## Imported meaning (OLMo)

The official Think-SFT/DPO wedge is technically valid but capability-gated
with empty prospective Bank-S cohorts; stage effects are missing, not zero,
and SFT versus DPO is unresolved. H6 licenses no in-band finite-dose regime
at L24/L32/L40 on either mandatory checkpoint; Think passes only the L56
epsilon-0.10 late anchor. The exact site-dose coverage needed to map
registered causal interventions onto the H6 ladder is unavailable in the
historical archive. The OLMo Think/Instruct Bank-W pair design has power
0.7788 at 16 common capable families; 18 is the first simulated passing
count, and no intervention is authorized from the current design.

Forbidden uses: not stage attribution; not evidence for no SFT or DPO
effect; not an externalization result; not a Bank-W null; not permission to
reauthor or subset families inside Phase 4; not invalidation of the
registered paired ablation effects; not a transport license for O5 or
crossed-lens patching.

## Explicit non-effects on the mainline

These admissions update only the terminal methods/development boundary. They
do not alter Q-L4, restore a retired primary, change a threshold, convert a
capability-gated absence into a zero, or open any confirmatory, replication,
or intervention outcome.
