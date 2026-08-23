# ModelPrint Reproducibility

ModelPrint retains four independent recovery layers for the authoritative run:
the pushed Git branch, a verified Git bundle plus worktree patch, immutable run
artifacts in Google Drive, and checksum-recorded SQL Server backups/BACPACs.
The scientific system of record is SQL Server; aggregate reports are rebuilt
from retained predictions, metrics, and frozen manifests rather than copied
from notebook output.

## Authoritative run

- run ID: `modelprint-full-20260822T230728Z`
- branch: `aidataapps-modelprint`
- local run: `runs/modelprint-full-20260822T230728Z`
- Drive mirror: `aidataapps/lab02/runs/modelprint-full-20260822T230728Z`
- campaign freeze: `manifests/campaign-freeze.json`
- artifact inventory: `ARTIFACT_INVENTORY.json`

## Independent reconstruction

After the final native backup, BACPAC, report build, and run inventory exist,
run:

```bash
source scripts/runtime-env.sh
npm run repro
```

The reproduction command verifies every inventoried digest, recomputes probe
metrics from `tables/predictions.parquet`, selects a checksum-valid Drive backup
when the mirror is mounted, restores it into a temporary database, regenerates
the report pack, and requires `reports/HEADLINE.md` to match byte-for-byte. It
drops the temporary database on exit and writes
`metrics/reproduction.json` only after every check passes.

## Current disposition

`IN_PROGRESS`. The final PASS receipt and completion tag are created only after
all evaluation stages and database exports finish; a partial run is never
described as reproduced.
