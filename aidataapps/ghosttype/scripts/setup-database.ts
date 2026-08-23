import { readFile, readdir } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { sha256 } from "../src/hash.js";
import { LAB_ROOT } from "../src/run.js";

// Creates both databases (compat 170, Query Store per inherited Lab 03
// settings, PREVIEW_FEATURES on autocommit per ModelPrint's scar), applies
// hash-tracked idempotent migrations, and refuses drifted migrations.

const config = loadConfig();

async function connect(database: string): Promise<sql.ConnectionPool> {
  return new sql.ConnectionPool({ ...config.databases.admin, database }).connect();
}

const master = await connect("master");
for (const name of [config.databases.controlName, config.databases.workloadName]) {
  await master.request().query(`IF DB_ID(N'${name}') IS NULL CREATE DATABASE [${name}];`);
  await master.request().batch(`ALTER DATABASE [${name}] SET COMPATIBILITY_LEVEL = 170;`);
  // Autocommit batch per ModelPrint: driver transactions are rejected for scoped-configuration DDL.
  await master.request().batch(`ALTER DATABASE [${name}] SET QUERY_STORE = ON (OPERATION_MODE = READ_WRITE, QUERY_CAPTURE_MODE = ALL, INTERVAL_LENGTH_MINUTES = 1, DATA_FLUSH_INTERVAL_SECONDS = 60, MAX_STORAGE_SIZE_MB = 1024, WAIT_STATS_CAPTURE_MODE = ON);`);
}
const control = await connect(config.databases.controlName);
await control.request().batch(`ALTER DATABASE SCOPED CONFIGURATION SET PREVIEW_FEATURES = ON;`).catch((error: unknown) => {
  console.warn(`PREVIEW_FEATURES not enabled: ${(error as Error).message}`);
});

const dir = `${LAB_ROOT}/db/migrations`;
const files = (await readdir(dir)).filter((name) => name.endsWith(".sql")).sort();
const applied: Array<Record<string, string>> = [];
for (const file of files) {
  const text = await readFile(`${dir}/${file}`, "utf8");
  const hash = sha256(text);
  const migrationId = file.replace(/\.sql$/, "");
  const existing = await control.request().input("id", sql.VarChar(120), migrationId)
    .query("SELECT migration_sha256 FROM control.schema_migrations WHERE migration_id=@id")
    .catch(() => ({ recordset: [] as Array<{ migration_sha256: string }> }));
  const row = existing.recordset[0];
  if (row) {
    if (row.migration_sha256 !== hash) throw new Error(`Migration drift: ${migrationId} recorded ${row.migration_sha256.slice(0, 12)} != on-disk ${hash.slice(0, 12)}`);
    applied.push({ migrationId, status: "already-applied" });
    continue;
  }
  for (const batch of text.split(/^\s*GO\s*$/m).map((chunk) => chunk.trim()).filter(Boolean)) {
    await control.request().batch(batch);
  }
  await control.request()
    .input("id", sql.VarChar(120), migrationId)
    .input("hash", sql.Char(64), hash)
    .query("INSERT control.schema_migrations(migration_id, migration_sha256) VALUES(@id, @hash)");
  applied.push({ migrationId, status: "applied" });
}
console.log(JSON.stringify({ databases: [config.databases.controlName, config.databases.workloadName], migrations: applied }, null, 2));
await Promise.allSettled([master.close(), control.close()]);
