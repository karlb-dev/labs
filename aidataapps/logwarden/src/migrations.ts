import { readdir, readFile } from "node:fs/promises";
import { basename } from "node:path";
import sql from "mssql";
import { sha256 } from "./hash.js";

export interface AppliedMigration {
  migrationId: string;
  sha256: string;
  status: "applied" | "already_applied";
  mode: "transactional" | "non_transactional";
}

export function migrationMode(source: string): AppliedMigration["mode"] {
  return /^\s*--\s*logwarden:migration-mode=non-transactional\s*$/im.test(source)
    ? "non_transactional"
    : "transactional";
}

export function splitSqlBatches(source: string): string[] {
  return source
    .split(/^\s*GO\s*(?:--.*)?$/gim)
    .map((batch) => batch.trim())
    .filter(Boolean);
}

export async function migrationFiles(directory: string): Promise<string[]> {
  return (await readdir(directory, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && /^\d{3}_[a-z0-9_]+\.sql$/.test(entry.name))
    .map((entry) => `${directory}/${entry.name}`)
    .sort();
}

export async function applyControlMigrations(
  pool: sql.ConnectionPool,
  directory: string,
  runId: string,
): Promise<AppliedMigration[]> {
  const outcomes: AppliedMigration[] = [];
  for (const path of await migrationFiles(directory)) {
    const migrationId = basename(path, ".sql");
    const source = await readFile(path, "utf8");
    const migrationSha256 = sha256(source);
    const mode = migrationMode(source);
    const tableExists = await pool.request().query<{ present: number }>(
      "SELECT CONVERT(int, CASE WHEN OBJECT_ID(N'control.schema_migrations', N'U') IS NULL THEN 0 ELSE 1 END) AS present",
    );
    if (tableExists.recordset[0]?.present === 1) {
      const prior = await pool.request()
        .input("migration_id", sql.VarChar(100), migrationId)
        .query<{ migration_sha256: string }>(
          "SELECT migration_sha256 FROM control.schema_migrations WHERE migration_id = @migration_id",
        );
      const recorded = prior.recordset[0]?.migration_sha256;
      if (recorded !== undefined) {
        if (recorded !== migrationSha256) {
          throw new Error(`Historical migration drift for ${migrationId}: database=${recorded} file=${migrationSha256}`);
        }
        outcomes.push({ migrationId, sha256: migrationSha256, status: "already_applied", mode });
        continue;
      }
    }

    if (mode === "non_transactional") {
      try {
        for (const batch of splitSqlBatches(source)) await pool.request().batch(batch);
        await pool.request()
          .input("migration_id", sql.VarChar(100), migrationId)
          .input("migration_sha256", sql.Char(64), migrationSha256)
          .input("run_id", sql.VarChar(120), runId)
          .query(`
            INSERT control.schema_migrations(migration_id, migration_sha256, applied_by_run_id)
            VALUES(@migration_id, @migration_sha256, @run_id);
          `);
      } catch (error) {
        throw new Error(`Non-transactional migration ${migrationId} failed; it must be idempotent before retry`, { cause: error });
      }
      outcomes.push({ migrationId, sha256: migrationSha256, status: "applied", mode });
      continue;
    }

    const transaction = new sql.Transaction(pool);
    await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
    try {
      for (const batch of splitSqlBatches(source)) await new sql.Request(transaction).batch(batch);
      await new sql.Request(transaction)
        .input("migration_id", sql.VarChar(100), migrationId)
        .input("migration_sha256", sql.Char(64), migrationSha256)
        .input("run_id", sql.VarChar(120), runId)
        .query(`
          INSERT control.schema_migrations(migration_id, migration_sha256, applied_by_run_id)
          VALUES(@migration_id, @migration_sha256, @run_id);
        `);
      await transaction.commit();
    } catch (error) {
      await transaction.rollback().catch(() => undefined);
      throw new Error(`Migration ${migrationId} failed`, { cause: error });
    }
    outcomes.push({ migrationId, sha256: migrationSha256, status: "applied", mode });
  }
  return outcomes;
}

export async function applyUntrackedMigrations(
  pool: sql.ConnectionPool,
  directory: string,
): Promise<AppliedMigration[]> {
  const outcomes: AppliedMigration[] = [];
  for (const path of await migrationFiles(directory)) {
    const migrationId = basename(path, ".sql");
    const source = await readFile(path, "utf8");
    const migrationSha256 = sha256(source);
    for (const batch of splitSqlBatches(source)) await pool.request().batch(batch);
    outcomes.push({ migrationId, sha256: migrationSha256, status: "applied", mode: migrationMode(source) });
  }
  return outcomes;
}
