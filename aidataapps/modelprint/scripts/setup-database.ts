import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashFile } from "../src/hash.js";

const config = loadConfig();
const allowHistoricalDrift = process.argv.includes("--allow-historical-drift");
const options = {
  server: config.database.server,
  port: config.database.port,
  user: config.database.user,
  password: config.database.password,
  options: { encrypt: false, trustServerCertificate: true },
  requestTimeout: 600_000,
};

const master = await new sql.ConnectionPool({ ...options, database: "master" }).connect();
try {
  const database = config.database.database;
  await master.request().query(`IF DB_ID(N'${database}') IS NULL EXEC(N'CREATE DATABASE [${database}]');`);
} finally {
  await master.close();
}

const pool = await new sql.ConnectionPool({ ...options, database: config.database.database }).connect();
try {
  await pool.request().batch(`ALTER DATABASE [${config.database.database}] SET COMPATIBILITY_LEVEL = 170;`);
  const migrationsDirectory = fileURLToPath(new URL("../db/migrations/", import.meta.url));
  const names = (await readdir(migrationsDirectory)).filter((name) => name.endsWith(".sql")).sort();
  for (const name of names) {
    const path = `${migrationsDirectory}/${name}`;
    const text = await readFile(path, "utf8");
    const digest = await hashFile(path);
    const tableExists = await pool.request().query<{ present: number }>("SELECT CASE WHEN OBJECT_ID(N'dbo.schema_migrations', N'U') IS NULL THEN 0 ELSE 1 END AS present;");
    if (tableExists.recordset[0]?.present) {
      const existing = await pool.request().input("id", sql.VarChar(80), name).query<{ migration_sha256: string }>("SELECT migration_sha256 FROM dbo.schema_migrations WHERE migration_id=@id;");
      const previous = existing.recordset[0]?.migration_sha256;
      if (previous && previous !== digest) {
        if (!allowHistoricalDrift) throw new Error(`Migration drift: ${name} was ${previous}, now ${digest}`);
        console.warn(JSON.stringify({ disposition: "HISTORICAL_MIGRATION_DRIFT_ACCEPTED", migration: name, recorded: previous, source: digest }));
      }
      if (previous) continue;
    }
    for (const batch of text.split(/^\s*GO\s*$/gim)) if (batch.trim()) await pool.request().batch(batch);
    await pool.request().input("id", sql.VarChar(80), name).input("sha", sql.Char(64), digest)
      .query("INSERT dbo.schema_migrations(migration_id,migration_sha256) VALUES(@id,@sha);");
    console.log(`Applied ${name} ${digest.slice(0, 12)}`);
  }
  const row = await pool.request().query(`SELECT @@VERSION AS version, DB_NAME() AS databaseName, compatibility_level AS compatibilityLevel FROM sys.databases WHERE name=DB_NAME();`);
  console.log(JSON.stringify(row.recordset[0], null, 2));
} finally {
  await pool.close();
}
