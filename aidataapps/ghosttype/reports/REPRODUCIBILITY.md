# GhostType REPRODUCIBILITY (Tier 1)

run: `ghosttype-cpu-dev-20260823T213801Z` · generated 2026-08-23T23:12:50.202Z · lab: aidataapps/ghosttype (Mac local campaign)

> Regenerated from GhostTypeControl by `npm run reports`. The database is the source of record.

## Row-only reproduction (Tier-1 closeout gate)

```bash
docker compose -f compose.yaml -f compose.mac.yaml up -d   # SQL on :1435
./scripts/repro.sh --mode rows
```

Rebuilds every row-level artifact from the vendored, hash-verified package: schema migrations (drift-refusing), dataset import (588 record hashes re-verified in-process), all GT-1 gates, catalog snapshots, the GT-3 parse oracle under the pinned ScriptDom [170.191.0], deterministic baselines B0–B4, aggregate metrics, and the unit suite for the frozen rules.

## Database backup/restore

Stage-boundary backups only (no rolling dumps): `npm run db:backup`. Restore re-applies PREVIEW_FEATURES and compatibility level 170 (inherited ModelPrint scar): `./scripts/export-database.sh restore <bak>`.

## Model campaign reproduction

Model rows depend on the frozen MLX servings on this machine (registry `config/models.mac.json`; per-profile pins recorded by the port gate in `control.model_profiles`). Re-running `npm run quality:run -- --profile <id>` after wiping that profile's requests reproduces the campaign at temperature 0; exact token-for-token stability is recorded per profile by the port-gate determinism canary.
