# GhostType DATASET_REPORT (Tier 1)

run: `ghosttype-cpu-dev-20260823T213801Z` · generated 2026-08-23T22:16:32.030Z · lab: aidataapps/ghosttype (Mac local campaign)

> Regenerated from GhostTypeControl by `npm run reports`. The database is the source of record.

## Role × class composition (post-disposition)

| role | class | cases |
|---|---|---|
| calibration | cursor_fragment | 6 |
| calibration | intent_query | 38 |
| test_capability | intent_query | 10 |
| test_id | cursor_fragment | 58 |
| test_id | intent_query | 277 |
| test_schema_holdout | intent_query | 20 |
| test_sparse_catalog | cursor_fragment | 18 |
| test_suffix_holdout | cursor_fragment | 12 |
| test_suffix_holdout | intent_query | 13 |
| test_template_holdout | intent_query | 26 |
| train | cursor_fragment | 64 |
| train | intent_query | 46 |

## Catalog composition

| catalog snapshot | cases |
|---|---|
| fitnessapp_test-legacy-v1.1.0 | 73 |
| ninjadb_a-legacy-v1.1.0 | 57 |
| ghost_editor_core-v2 | 56 |
| ghost_parser_edges-v2 | 55 |
| ghost_vector_search-v2 | 54 |
| ghost_ops_azure-v2 | 54 |
| ghost_sparse_tenant-v2 | 52 |
| ghost_temporal_audit-v2 | 52 |
| adventure_azure-legacy-v1.1.0 | 34 |
| master_onprem-legacy-v1.1.0 | 33 |
| fabric_warehouse-legacy-v1.1.0 | 29 |
| msdb_onprem-legacy-v1.1.0 | 22 |
| ninjadb_b-legacy-v1.1.0 | 10 |
| azure_master-legacy-v1.1.0 | 7 |

## Oracle status (GT-3, ScriptDom pinned 170.191.0)

| oracle | status | cases |
|---|---|---|
| parse | fail | 8 |
| parse | not_applicable | 101 |
| parse | pass | 479 |
| parser_generation | not_applicable | 101 |
| parser_generation | pass | 487 |

### Adjudicated package defects (golds untouched; logged for dataset v2.1)

| case | role | finding |
|---|---|---|
| gt-cur-case-03 | train | gold is syntactically invalid T-SQL |
| gt-cur-case-04 | train | gold is syntactically invalid T-SQL |
| gt-edge-cont-05 | test_suffix_holdout | cursor placed after statement terminator |
| gt-edge-proc-05 | test_suffix_holdout | gold is syntactically invalid T-SQL |
| gt-sparse-alias-01 | train | cursor placed after statement terminator |
| gt-sparse-alias-02 | train | cursor placed after statement terminator |
| gt-sparse-alias-03 | train | cursor placed after statement terminator |
| gt-sparse-alias-04 | train | cursor placed after statement terminator |

Plus two split-group role-governance dispositions (`config/dataset-dispositions.json`).

## B2 category-detection coverage (addendum K-1 requirement)

| category | rows |
|---|---|
| no_category | 545 |
| dot_unresolved | 14 |
| column_of_alias | 8 |
| table_of_schema | 6 |
| table_source | 6 |
| table_source_empty | 4 |
| procedure | 4 |
| procedure_empty | 1 |
