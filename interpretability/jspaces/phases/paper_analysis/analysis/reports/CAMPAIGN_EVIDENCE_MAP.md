# CAMPAIGN_EVIDENCE_MAP.md — P2 output

Source: `data/master_evidence_events.parquet` / `master_evidence_live.parquet` (builder `scripts/build_evidence_graph.py`; registries read-only, hashes in `ANALYSIS_FOUNDATION.json`).

## Registry totals (event-sourced)

| Registry | Events | Ids | Live | Superseded | Withdrawn | Imports |
|---|---:|---:|---:|---:|---:|---:|
| part2_events | 137 | 104 | 78 | 25 | 1 | 0 |
| phase3 | 123 | 87 | 57 | 29 | 1 | 0 |
| phase4 | 82 | 72 | 65 | 3 | 4 | 6 |
| gemma | 24 | 23 | 23 | 0 | 0 | 9 |
| olmo_lineage | 41 | 38 | 37 | 1 | 0 | 1 |

## Live evidence by tier

| Registry | Tier | Live ids |
|---|---|---:|
| gemma | historical-development-import | 9 |
| gemma | methods | 14 |
| olmo_lineage | development | 13 |
| olmo_lineage | methods | 24 |
| part2_events | confirmatory | 15 |
| part2_events | dev | 14 |
| part2_events | exploratory | 3 |
| part2_events | exploratory-pilot | 2 |
| part2_events | methods | 3 |
| part2_events | pilot | 41 |
| phase3 | methods | 21 |
| phase3 | phase3-confirmatory | 7 |
| phase3 | phase3-development | 26 |
| phase3 | phase3-replication | 3 |
| phase4 | methods | 19 |
| phase4 | phase3-confirmatory-import | 1 |
| phase4 | phase4-development | 40 |
| phase4 | side-development-import | 5 |

## Import edges (native tier preserved at source)

| Importing registry | Import event | Source event | Source study |
|---|---|---|---|
| phase4 | `p4-import-phase3-release-v1` | `p3-release-manifest-v1` | jspace-phase3 |
| phase4 | `p4-import-olmo-bank-w-capability-v1` | `jspace-olmo-lineage-phase4-early-v1` | jspace-olmo-lineage |
| phase4 | `p4-import-gemma-transport-v1` | `jspace-gemma-transport-phase4-v1` | jspace-gemma-transport |
| phase4 | `p4-import-olmo-lineage-final-v1` | `jspace-olmo-lineage-phase4-final-v1` | jspace-olmo-lineage |
| phase4 | `p4-import-gemma-transport-study2-v1` | `gm2-sidelines2-import-bundle-v1` | jspace-gemma-transport-study2 |
| phase4 | `p4-import-olmo-lineage-study2-v1` | `ol2-sidelines2-import-bundle-v1` | jspace-olmo-lineage |
| gemma | `gm-import-a3-gemma-fullfit-v1` | `a3-gemma-fullfit-v1` | jspace-part2 |
| gemma | `gm-import-a3-gemma-identification-v1` | `a3-gemma-identification-v1` | jspace-part2 |
| gemma | `gm-import-a3-gemma-readout-verdict-v1` | `a3-gemma-readout-verdict-v1` | jspace-part2 |
| gemma | `gm-import-a3-gemma-deepband-logit-v1` | `a3-gemma-deepband-logit-v1` | jspace-part2 |
| gemma | `gm-import-local-linearity-v3-gemma4-31b` | `local-linearity-v3-gemma4-31b` | jspace-part2 |
| gemma | `gm-import-linearization-faithfulness-gemma4-31b-v2` | `linearization-faithfulness-gemma4-31b-v2` | jspace-part2 |
| gemma | `gm-import-readout-control-olmo3think-v1` | `readout-control-olmo3think-v1` | jspace-part2 |
| gemma | `gm-import-local-linearity-v3-olmo3-think` | `local-linearity-v3-olmo3-think` | jspace-part2 |
| gemma | `gm-import-linearization-faithfulness-olmo3-think-v2` | `linearization-faithfulness-olmo3-think-v2` | jspace-part2 |
| olmo_lineage | `ol2-gemma-backend-calibration-import-v1` | `gm2-backend-parity-calibration-v1` | jspace-gemma-transport-study2 |

## Anomaly register

- none detected

## Flat part2 registry duplicate rows (non-event file)

The flat `evidence_registry.jsonl` mirror contains append-duplicates; the event-sourced file is authoritative for status. Duplicated ids:
- `a1-fit-olmo31think-v1` ×2
- `g6-power-sim-v1` ×2
- `h7-context-j-olmo3-think-v1` ×2
- `linearization-faithfulness-gemma4-31b-v1` ×2
- `local-linearity-epssweep-olmo3-think-v1` ×2
- `local-linearity-gemma4-31b-v1` ×2
- `local-linearity-olmo3-think-v1` ×2
- `n6-confirmatory-grid-olmo31-think-v1` ×5
- `r7-protected-dynamic-pilot-olmo31instruct-lensB-v1` ×2
- `r7-protected-dynamic-pilot-olmo31instruct-v1` ×2
- `r7-tail-mechanism-olmo3-think-v1` ×2
- `tailrate-endpoint-crossmodel-v1` ×2
