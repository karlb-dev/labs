import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { copyFile, mkdir, stat, unlink } from "node:fs/promises";
import { basename } from "node:path";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { connect, sqlIdentifier } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

interface BackupFileRow { LogicalName: string; Type: "D" | "L" | "F" | "S" }

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const runId = basename(runDirectory);
if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$/.test(runId)) throw new Error(`Unsafe run ID: ${runId}`);
const restoreTest = process.argv.includes("--restore-test");
const stamp = new Date().toISOString().replaceAll(/[-:.]/g, "");
const composeProject = process.env.COMPOSE_PROJECT_NAME ?? "aidataapps-logwarden";
const backupDirectory = `${runDirectory}/database/backups`;
await mkdir(backupDirectory, { recursive: true, mode: 0o700 });
const sqlVolumePath = (await commandOutput("docker", ["volume", "inspect", `${composeProject}-sqlserver-data`, "--format", "{{.Mountpoint}}"])).trim();
if (!sqlVolumePath.startsWith("/")) throw new Error("SQL data-volume mountpoint was not absolute");

const master = await connect(config.databases.admin, "master");
const artifacts: Array<Record<string, unknown>> = [];
const restoreResults: Array<Record<string, unknown>> = [];
try {
  await master.request().query("EXEC master.dbo.xp_create_subdir N'/var/opt/mssql/backup';").catch((error: unknown) => {
    // xp_create_subdir reports an OS error when the directory already exists.
    const message = (error as { message?: string }).message ?? String(error);
    if (!/already exists|error 183/i.test(message)) throw error;
  });
  for (const [kind, database] of [["control", config.databases.controlName], ["workload", config.databases.workloadName]] as const) {
    const fileName = `${runId}-${kind}-${stamp}.bak`;
    const containerPath = `/var/opt/mssql/backup/${fileName}`;
    const localPath = `${backupDirectory}/${fileName}`;
    await master.request()
      .input("path", sql.NVarChar(500), containerPath)
      .query(`BACKUP DATABASE ${sqlIdentifier(database)} TO DISK=@path WITH COPY_ONLY, INIT, CHECKSUM, COMPRESSION; RESTORE VERIFYONLY FROM DISK=@path WITH CHECKSUM;`);
    try {
      await copyFile(`${sqlVolumePath}/backup/${fileName}`, localPath);
    } catch (error) {
      // Docker Desktop (mac profile): volume mountpoints live inside the VM,
      // not on the host filesystem — copy out of the container instead.
      if ((error as { code?: string }).code !== "ENOENT") throw error;
      await commandOutput("docker", ["cp", `${composeProject}-sqlserver-1:${containerPath}`, localPath]);
    }
    const file = await stat(localPath);
    const artifact = {
      kind,
      database,
      fileName,
      byteCount: file.size,
      sha256: await fileSha256(localPath),
      sqlBackupPath: containerPath,
      localPath,
      verifyOnly: "passed",
    };
    artifacts.push(artifact);
    await master.request()
      .input("run", sql.VarChar(120), runId)
      .input("path", sql.NVarChar(500), `database/backups/${fileName}`)
      .input("bytes", sql.BigInt, file.size)
      .input("hash", sql.Char(64), artifact.sha256)
      .query(`
        IF NOT EXISTS (SELECT 1 FROM ${sqlIdentifier(config.databases.controlName)}.control.run_artifacts WHERE run_id=@run AND relative_path=@path)
          INSERT ${sqlIdentifier(config.databases.controlName)}.control.run_artifacts
            (run_id,relative_path,artifact_kind,byte_count,sha256)
          VALUES(@run,@path,'database_backup',@bytes,@hash);
      `);
    if (restoreTest) restoreResults.push(await restoreAndProbe(master, kind, database, containerPath, stamp));
    await removeSqlStagingBackup(`${sqlVolumePath}/backup/${fileName}`, containerPath);
  }
} finally {
  await master.close();
}

const receiptBody = {
  schemaVersion: 1,
  runId,
  createdAtUtc: new Date().toISOString(),
  backupOptions: ["COPY_ONLY", "INIT", "CHECKSUM", "COMPRESSION"],
  artifacts,
  restoreTest,
  restoreResults,
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/database/checkpoints/${stamp}-backup-receipt.json`, `${JSON.stringify(receipt, null, 2)}\n`);
await atomicWrite(`${runDirectory}/database/backup-receipt.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({
  runId,
  backups: artifacts.map((artifact) => ({ kind: artifact.kind, bytes: artifact.byteCount, sha256: artifact.sha256 })),
  restoreTest,
  restoreResults,
  receiptSha256: receipt.receiptSha256,
}, null, 2));

async function restoreAndProbe(
  masterPool: sql.ConnectionPool,
  kind: "control" | "workload",
  sourceDatabase: string,
  backupPath: string,
  stampValue: string,
): Promise<Record<string, unknown>> {
  const shortStamp = stampValue.replace(/T.*/, "").slice(-8) + stampValue.slice(-7, -1);
  const restoreName = `LW_Restore_${kind}_${shortStamp}`.slice(0, 63);
  const restoreIdentifier = sqlIdentifier(restoreName);
  const fileList = await masterPool.request().input("path", sql.NVarChar(500), backupPath)
    .query<BackupFileRow>("RESTORE FILELISTONLY FROM DISK=@path;");
  const moveClauses = fileList.recordset.map((file, index) => {
    const extension = file.Type === "L" ? "ldf" : index === 0 ? "mdf" : "ndf";
    return `MOVE ${sqlLiteral(file.LogicalName)} TO ${sqlLiteral(`/var/opt/mssql/data/${restoreName}-${index}.${extension}`)}`;
  });
  if (moveClauses.length === 0) throw new Error(`Backup contained no files: ${sourceDatabase}`);
  try {
    await masterPool.request().input("path", sql.NVarChar(500), backupPath)
      .query(`RESTORE DATABASE ${restoreIdentifier} FROM DISK=@path WITH ${moveClauses.join(", ")}, RECOVERY, REPLACE;`);
    const restored = await connect(config.databases.admin, restoreName);
    try {
      const integrity = await restored.request().query<{ table_count: number; migration_count: number | null }>(`
        DBCC CHECKDB WITH NO_INFOMSGS, PHYSICAL_ONLY;
        SELECT CONVERT(int,COUNT(*)) AS table_count,
               ${kind === "control" ? "(SELECT CONVERT(int,COUNT(*)) FROM control.schema_migrations)" : "NULL"} AS migration_count
        FROM sys.tables WHERE is_ms_shipped=0;
      `);
      return {
        kind,
        restoreDatabase: restoreName,
        checkDb: "passed",
        tableCount: integrity.recordset[0]?.table_count,
        migrationCount: integrity.recordset[0]?.migration_count,
      };
    } finally {
      await restored.close();
    }
  } finally {
    if ((await masterPool.request().input("name", sql.NVarChar(128), restoreName).query<{ present: number }>("SELECT CONVERT(int,CASE WHEN DB_ID(@name) IS NULL THEN 0 ELSE 1 END) AS present;")).recordset[0]?.present === 1) {
      await masterPool.request().batch(`ALTER DATABASE ${restoreIdentifier} SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE ${restoreIdentifier};`);
    }
  }
}

function sqlLiteral(value: string): string {
  return `N'${value.replaceAll("'", "''")}'`;
}

function fileSha256(path: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

function commandOutput(command: string, arguments_: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(command, arguments_, { encoding: "utf8", maxBuffer: 4 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error !== null) reject(new Error(`${command} failed: ${stderr.trim() || error.message}`));
      else resolve(stdout);
    });
  });
}

async function removeSqlStagingBackup(hostPath: string, containerPath: string): Promise<void> {
  try {
    await unlink(hostPath);
  } catch (error) {
    if ((error as { code?: string }).code !== "ENOENT") throw error;
    await commandOutput("docker", ["exec", `${composeProject}-sqlserver-1`, "rm", "--", containerPath]);
  }
}
