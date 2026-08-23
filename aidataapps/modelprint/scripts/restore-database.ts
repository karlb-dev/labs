import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { valueAfter } from "../src/run.js";

const target=valueAfter("--target");const backup=valueAfter("--backup");const drop=process.argv.includes("--drop");
if (!target || (target!=="ModelPrint" && !/^ModelPrintRepro_[A-Za-z0-9_]+$/.test(target))) throw new Error("--target must be ModelPrint or match ModelPrintRepro_[A-Za-z0-9_]+");
if (target==="ModelPrint" && drop) throw new Error("Refusing --drop for the primary ModelPrint database");
const config=loadConfig().database;const pool=await new sql.ConnectionPool({server:config.server,port:config.port,user:config.user,password:config.password,database:"master",
 options:{encrypt:false,trustServerCertificate:true},requestTimeout:3_600_000}).connect();
try {
 const escapedTarget=target;
 if (drop) { await pool.request().batch(`IF DB_ID(N'${target}') IS NOT NULL BEGIN ALTER DATABASE [${escapedTarget}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE [${escapedTarget}]; END;`);console.log(JSON.stringify({target,dropped:true})); }
 else {
  if (!backup || !/^\/var\/opt\/mssql\/data\/[A-Za-z0-9_.-]+\.bak$/.test(backup)) throw new Error("--backup must be an explicit .bak under /var/opt/mssql/data");
  const exists=await pool.request().query(`SELECT DB_ID(N'${target}') id;`);if (exists.recordset[0]?.id) throw new Error(`Refusing to replace existing ${target}`);
  const escapedBackup=backup.replaceAll("'","''");const listing=await pool.request().batch(`RESTORE FILELISTONLY FROM DISK=N'${escapedBackup}';`);
  const rows=listing.recordset as Array<{LogicalName:string;Type:string}>;const data=rows.find((row)=>row.Type==="D");const log=rows.find((row)=>row.Type==="L");
  if (!data || !log) throw new Error("Backup did not expose one data and one log logical file");
  await pool.request().batch(`RESTORE DATABASE [${escapedTarget}] FROM DISK=N'${escapedBackup}' WITH CHECKSUM,RECOVERY,
    MOVE N'${data.LogicalName.replaceAll("'","''")}' TO N'/var/opt/mssql/data/${target}.mdf',
    MOVE N'${log.LogicalName.replaceAll("'","''")}' TO N'/var/opt/mssql/data/${target}_log.ldf',STATS=10;`);
  console.log(JSON.stringify({target,backup,restored:true}));
 }
} finally {await pool.close();}
