import { basename } from "node:path";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { resolveRunDirectory } from "../src/run.js";

const config = loadConfig().database;
const runId = basename(resolveRunDirectory());
if (!/^[a-z0-9-]+$/i.test(runId)) throw new Error("Unsafe run ID");
const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
const fileName = `${runId}-${stamp}.bak`;
const containerPath = `/var/opt/mssql/data/${fileName}`;
const pool = await new sql.ConnectionPool({ server: config.server, port: config.port, user: config.user, password: config.password,
  database: "master", options: { encrypt: false, trustServerCertificate: true }, requestTimeout: 3_600_000 }).connect();
try {
  const escapedPath = containerPath.replaceAll("'", "''");
  const database = config.database;
  await pool.request().batch(`BACKUP DATABASE [${database}] TO DISK=N'${escapedPath}' WITH COPY_ONLY, COMPRESSION, CHECKSUM, INIT, STATS=10;`);
  const verify = await pool.request().batch(`RESTORE VERIFYONLY FROM DISK=N'${escapedPath}' WITH CHECKSUM;`);
  console.log(JSON.stringify({ runId, fileName, containerPath, verified: true, resultSets: verify.recordsets.length }));
} finally { await pool.close(); }
