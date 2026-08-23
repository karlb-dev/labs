#!/usr/bin/env bash
# Stage-boundary database export (NOT a timer — Karl's rule: no rolling
# 20-minute dumps; back up at stage boundaries only).
#   ./scripts/export-database.sh backup            -> .bak via docker exec
#   ./scripts/export-database.sh restore <file>    -> restore + re-apply
#                                                     preview/compat (ModelPrint scar)
set -euo pipefail
cd "$(dirname "$0")/.."
source .env

CONTAINER="aidataapps-ghosttype-sqlserver-1"
DB="${CONTROL_DATABASE:-GhostTypeControl}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="backups"
mkdir -p "$OUT_DIR"

sqlcmd_exec() {
  # container listens on MSSQL_TCP_PORT (1435), not the default 1433
  docker exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S "localhost,${SQLSERVER_INTERNAL_PORT:-1435}" -U sa -P "$MSSQL_SA_PASSWORD" -C -b -Q "$1"
}

case "${1:-}" in
  backup)
    IN_CONTAINER="/var/opt/mssql/backup/${DB}_${STAMP}.bak"
    docker exec "$CONTAINER" mkdir -p /var/opt/mssql/backup
    sqlcmd_exec "BACKUP DATABASE [$DB] TO DISK = N'$IN_CONTAINER' WITH INIT, COMPRESSION, CHECKSUM"
    docker exec "$CONTAINER" chmod a+r "$IN_CONTAINER"   # docker-cp root-owned scar (Lab 02)
    docker cp "$CONTAINER:$IN_CONTAINER" "$OUT_DIR/"
    docker exec "$CONTAINER" rm -f "$IN_CONTAINER"        # keep the VM disk lean
    shasum -a 256 "$OUT_DIR/${DB}_${STAMP}.bak" | tee "$OUT_DIR/${DB}_${STAMP}.bak.sha256"
    echo "[backup] $OUT_DIR/${DB}_${STAMP}.bak"
    ;;
  restore)
    BAK="${2:?usage: export-database.sh restore <local .bak>}"
    BASE="$(basename "$BAK")"
    docker exec "$CONTAINER" mkdir -p /var/opt/mssql/backup
    docker cp "$BAK" "$CONTAINER:/var/opt/mssql/backup/$BASE"
    sqlcmd_exec "ALTER DATABASE [$DB] SET SINGLE_USER WITH ROLLBACK IMMEDIATE"
    sqlcmd_exec "RESTORE DATABASE [$DB] FROM DISK = N'/var/opt/mssql/backup/$BASE' WITH REPLACE"
    sqlcmd_exec "ALTER DATABASE [$DB] SET MULTI_USER"
    # ModelPrint scar: restores/imports drop preview + compat; re-apply.
    sqlcmd_exec "ALTER DATABASE [$DB] SET COMPATIBILITY_LEVEL = 170"
    sqlcmd_exec "ALTER DATABASE SCOPED CONFIGURATION SET PREVIEW_FEATURES = ON"
    echo "[restore] $DB restored from $BASE (preview/compat re-applied)"
    ;;
  *)
    echo "usage: $0 backup | restore <bak>" >&2
    exit 2
    ;;
esac
