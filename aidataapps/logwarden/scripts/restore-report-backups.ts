import { execFile } from "node:child_process";
import { mkdir, readFile, stat } from "node:fs/promises";
import { basename } from "node:path";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashFile, hashJson } from "../src/hash.js";
import { connect, sqlIdentifier } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";

interface BackupArtifact {
  kind: "control" | "workload";
  fileName: string;
  byteCount: number;
  sha256: string;
}

interface BackupReceipt {
  schemaVersion: number;
  runId: string;
  artifacts: BackupArtifact[];
  receiptSha256: string;
  [key: string]: unknown;
}

interface BackupFileRow { LogicalName: string; Type: "D" | "L" | "F" | "S" }

const runDirectory = resolveRunDirectory();
const receiptPath = valueAfter("--receipt") ?? `${runDirectory}/database/backup-receipt.json`;
const receipt = JSON.parse(await readFile(receiptPath, "utf8")) as BackupReceipt;
const receivedReceiptHash = receipt.receiptSha256;
const { receiptSha256: _receiptSha256, ...receiptBody } = receipt;
if (hashJson(receiptBody) !== receivedReceiptHash) throw new Error("Backup receipt hash drift");
if (receipt.runId !== basename(runDirectory)) throw new Error("Backup receipt run mismatch");
if (receipt.artifacts.length !== 2 || new Set(receipt.artifacts.map((item) => item.kind)).size !== 2) {
  throw new Error("Expected exactly one control and one workload backup");
}

const config = loadConfig();
const composeProject = process.env.COMPOSE_PROJECT_NAME ?? "aidataapps-logwarden";
if (!/^[a-z0-9][a-z0-9_-]{0,62}$/.test(composeProject)) throw new Error(`Unsafe Compose project: ${composeProject}`);
const container = process.env.LOGWARDEN_REPRO_SQL_CONTAINER ?? `${composeProject}-sqlserver-1`;
if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(container)) throw new Error(`Unsafe SQL container: ${container}`);
const mountedBackups = process.env.LOGWARDEN_REPRO_BACKUPS_DIR !== undefined;
if (!mountedBackups) await command("docker", ["exec", container, "mkdir", "-p", "/var/opt/mssql/backup"]);

const master = await connect(config.databases.admin, "master", 600_000, 2);
const restored: Array<Record<string, unknown>> = [];
try {
  for (const artifact of [...receipt.artifacts].sort((left, right) => left.kind.localeCompare(right.kind))) {
    if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\.bak$/.test(artifact.fileName)) throw new Error(`Unsafe backup file name: ${artifact.fileName}`);
    const localPath = `${runDirectory}/database/backups/${artifact.fileName}`;
    const information = await stat(localPath);
    if (information.size !== artifact.byteCount || await hashFile(localPath) !== artifact.sha256) {
      throw new Error(`Backup artifact drift: ${artifact.fileName}`);
    }
    const containerPath = mountedBackups
      ? `/var/opt/mssql/repro-backups/${artifact.fileName}`
      : `/var/opt/mssql/backup/repro-${artifact.kind}.bak`;
    if (!mountedBackups) await command("docker", ["cp", localPath, `${container}:${containerPath}`]);
    const target = artifact.kind === "control" ? config.databases.controlName : config.databases.workloadName;
    const targetIdentifier = sqlIdentifier(target);
    const present = await master.request().input("name", sql.NVarChar(128), target)
      .query<{ present: number }>("SELECT CONVERT(int,CASE WHEN DB_ID(@name) IS NULL THEN 0 ELSE 1 END) present;");
    if (present.recordset[0]?.present !== 0) throw new Error(`Restore target already exists in fresh service: ${target}`);
    const files = await master.request().input("path", sql.NVarChar(500), containerPath)
      .query<BackupFileRow>("RESTORE FILELISTONLY FROM DISK=@path;");
    if (files.recordset.length === 0) throw new Error(`Backup has no database files: ${artifact.fileName}`);
    const moves = files.recordset.map((file, index) => {
      const extension = file.Type === "L" ? "ldf" : index === 0 ? "mdf" : "ndf";
      return `MOVE ${literal(file.LogicalName)} TO ${literal(`/var/opt/mssql/data/${target}-${index}.${extension}`)}`;
    });
    await master.request().input("path", sql.NVarChar(500), containerPath)
      .query(`RESTORE DATABASE ${targetIdentifier} FROM DISK=@path WITH ${moves.join(",")},RECOVERY;`);
    const database = await connect(config.databases.admin, target, 600_000, 2);
    try {
      const check = await database.request().query<{ table_count: number; migration_count: number | null }>(`
        DBCC CHECKDB WITH NO_INFOMSGS,PHYSICAL_ONLY;
        SELECT CONVERT(int,COUNT(*)) table_count,
          ${artifact.kind === "control" ? "(SELECT CONVERT(int,COUNT(*)) FROM control.schema_migrations)" : "NULL"} migration_count
        FROM sys.tables WHERE is_ms_shipped=0;`);
      if (artifact.kind === "control") await waitForFullText(database);
      restored.push({
        kind: artifact.kind,
        database: target,
        backupSha256: artifact.sha256,
        backupBytes: artifact.byteCount,
        checkDb: "PASS",
        tableCount: check.recordset[0]?.table_count,
        migrationCount: check.recordset[0]?.migration_count,
        fullTextPopulation: artifact.kind === "control" ? "PASS" : "not_applicable",
      });
    } finally {
      await database.close();
    }
    if (!mountedBackups) await command("docker", ["exec", container, "rm", "--", containerPath]);
  }
} finally {
  await master.close();
}

const body = {
  schemaVersion: 1,
  runId: receipt.runId,
  restoredAtUtc: new Date().toISOString(),
  sourceBackupReceiptSha256: receivedReceiptHash,
  sqlServer: `${process.env.SQLSERVER_HOST ?? "127.0.0.1"}:${process.env.SQLSERVER_PORT ?? "1434"}`,
  composeProject,
  restored,
  disposition: "PASS",
};
const result = { ...body, receiptSha256: hashJson(body) };
const stamp = body.restoredAtUtc.replaceAll(/[-:.]/g, "");
await mkdir(`${runDirectory}/repro/restore-executions`, { recursive: true });
await atomicWrite(`${runDirectory}/repro/restore-executions/${stamp}-database.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(result, null, 2));

async function waitForFullText(pool: sql.ConnectionPool): Promise<void> {
  for (let attempt = 0; attempt < 90; attempt += 1) {
    const result = await pool.request().query<{ active_count: number }>(`
      SELECT CONVERT(int,COUNT(*)) active_count FROM sys.fulltext_catalogs
      WHERE FULLTEXTCATALOGPROPERTY(name,'PopulateStatus') <> 0;`);
    if (result.recordset[0]?.active_count === 0) return;
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 2_000));
  }
  throw new Error("Full-text catalogs did not reach an idle population state after restore");
}

function literal(value: string): string {
  return `N'${value.replaceAll("'", "''")}'`;
}

function command(executable: string, arguments_: string[]): Promise<void> {
  return new Promise((resolvePromise, reject) => {
    execFile(executable, arguments_, { encoding: "utf8", maxBuffer: 8 * 1024 * 1024 }, (error, _stdout, stderr) => {
      if (error === null) resolvePromise();
      else reject(new Error(`${executable} ${arguments_.join(" ")} failed: ${stderr.trim() || error.message}`));
    });
  });
}
