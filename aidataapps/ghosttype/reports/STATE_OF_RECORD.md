# GhostType STATE_OF_RECORD (Tier 1, mac campaign)

run: `ghosttype-cpu-dev-20260823T213801Z` · generated 2026-08-24T16:09:28.978Z · lab: aidataapps/ghosttype (Mac local campaign)

> Regenerated from GhostTypeControl by `npm run reports`. The database is the source of record.

## Campaign identity

- parser: Microsoft.SqlServer.TransactSql.ScriptDom **[170.191.0]** (TSql170Parser)
- frozen rules: normalize-v1, recompose-v1, candidate-extract-v1
- transport: raw text, deterministic reference decode (T=0, top_p=1, no stop strings)
- SQL host: SQL Server 2025 CU8 under Rosetta emulation — every SQL timing is DEV-tier
- model host: mlx_lm.server :8020 / mlx_vlm.server :8021 (single resident model)

## Model profiles / port gates

| profile | served model | port gate |
|---|---|---|
| gemma-4-26b-a4b-mlx | mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit | PASS |
| gemma-4-e4b-mlx | mlx-community/gemma-4-e4b-it-OptiQ-4bit | PASS |
| muse-glimmer-30b-mlx | mlx-community/Muse-Glimmer-30B-4bit | WARN |
| olmo-3.1-32b-mlx | lmstudio-community/Olmo-3.1-32B-Instruct-MLX-4bit | PASS |
| qwen-3.8-27b-mlx | mlx-community/Qwen3.8-27B-4bit | PASS |

## Evidence event log

| stage | disposition | event key | at (UTC) |
|---|---|---|---|
| GT-1 | FAIL | `gt1-dataset-validate:75be363a4414c201` | 2026-08-23T21:39:21.614Z |
| GT-1 | FAIL | `gt1-dataset-validate:027453ccf01318f7` | 2026-08-23T21:41:01.146Z |
| GT-1 | FAIL | `gt1-dataset-validate:9e5d3c26446269b3` | 2026-08-23T21:41:17.398Z |
| GT-1 | PASS | `gt1-dataset-validate:d1399207086c3132` | 2026-08-23T21:41:51.577Z |
| GT-3 | PASS_WITH_FINDINGS | `gt3-parse-oracle:1de942e577350310` | 2026-08-23T21:47:46.753Z |
| GT-3 | PASS_WITH_FINDINGS | `gt3-parse-oracle:963a3bb9dc2ab366` | 2026-08-23T21:50:59.218Z |
| GT-3 | PASS_WITH_FINDINGS | `gt3-parse-oracle:08f605c2620a5d1c` | 2026-08-23T21:52:39.364Z |
| GT-2 | PASS | `gt2-catalog-snapshot:7d63ef3c32cf8d18` | 2026-08-23T21:57:09.720Z |
| GT-2 | PASS | `gt2-catalog-snapshot:ff6b58fa84380389` | 2026-08-23T21:57:40.489Z |
| GT-2 | PASS | `gt2-catalog-snapshot:4827a14de9dfc904` | 2026-08-23T21:57:57.257Z |
| GT-5 | PASS | `gt5-baselines:bf76f6cf8ed1a38b` | 2026-08-23T22:01:47.508Z |
| GT-5 | PASS | `gt5-baselines:bf561057459b3240` | 2026-08-23T22:02:17.281Z |
| GT-8 | PASS | `gt8-metrics:88d1d5d17a9b2105` | 2026-08-23T22:11:36.808Z |
| GT-8 | PASS | `gt8-metrics:83d7a36e82a19ed6` | 2026-08-23T23:11:37.122Z |
| GT-11 | PASS_WITH_FINDINGS | `gt11-port-gate:45a624ad4a3b934b` | 2026-08-23T23:21:25.222Z |
| GT-11 | PASS | `gt11-port-gate:deaa2184eb9b4872` | 2026-08-23T23:24:41.331Z |
| GT-8 | PASS | `gt8-metrics:705f96216c398016` | 2026-08-24T01:14:09.528Z |
| GT-8 | PASS | `gt8-metrics:6e5de2021951df1a` | 2026-08-24T09:01:53.801Z |
| GT-8 | PASS | `gt8-metrics:e77d5083d66dd7fc` | 2026-08-24T14:26:37.944Z |
| GT-8 | PASS | `gt8-metrics:12ca5f88d16e42c7` | 2026-08-24T16:09:28.515Z |

## Inherited scars honored

Autocommit DDL for preview features; explicit isolation hygiene; BACPAC restore re-applies preview/compat; unpaired-surrogate policy in insertion integrity; DiskANN INT-key rule reserved for the Tier-2 ANN comparator; statement-relative cursor offsets documented in migration 004.
