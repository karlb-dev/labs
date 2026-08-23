import sql from "mssql";
import type { AppConfig } from "./config.js";

type Connection = AppConfig["databases"]["admin"];

export function sqlIdentifier(value: string): string {
  if (!/^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(value)) throw new Error(`Unsafe SQL identifier: ${value}`);
  return `[${value}]`;
}

export async function connect(connection: Connection, database: string, timeoutMs = 600_000): Promise<sql.ConnectionPool> {
  return new sql.ConnectionPool({
    ...connection,
    database,
    requestTimeout: timeoutMs,
    connectionTimeout: 30_000,
    pool: { min: 0, max: 10, idleTimeoutMillis: 30_000 },
  }).connect();
}

export async function withPool<T>(connection: Connection, database: string, operation: (pool: sql.ConnectionPool) => Promise<T>): Promise<T> {
  const pool = await connect(connection, database);
  try {
    return await operation(pool);
  } finally {
    await pool.close();
  }
}
