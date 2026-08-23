import sql from "mssql";
import { loadConfig } from "../src/config.js";

const config = loadConfig().database;
const escapedDatabase = config.database.replaceAll("]", "]]");
const pool = await new sql.ConnectionPool({
  server: config.server,
  port: config.port,
  user: config.user,
  password: config.password,
  database: config.database,
  options: { encrypt: false, trustServerCertificate: true },
  requestTimeout: 600_000,
}).connect();

try {
  await pool.request().batch(`ALTER DATABASE [${escapedDatabase}] SET COMPATIBILITY_LEVEL = 170;`);
  await pool.request().batch("ALTER DATABASE SCOPED CONFIGURATION SET PREVIEW_FEATURES = ON;");
  const result = await pool.request().query(`SELECT DB_NAME() database_name,d.compatibility_level,
    (SELECT value FROM sys.database_scoped_configurations WHERE name='PREVIEW_FEATURES') preview_features
    FROM sys.databases d WHERE d.name=DB_NAME()`);
  console.log(JSON.stringify(result.recordset[0]));
} finally {
  await pool.close();
}
