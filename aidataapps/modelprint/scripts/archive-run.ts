import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readdir, stat, writeFile } from "node:fs/promises";
import { basename, relative } from "node:path";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { resolveRunDirectory } from "../src/run.js";

const runDirectory = resolveRunDirectory(); const runId = basename(runDirectory); const config = loadConfig().database;
async function files(root: string): Promise<string[]> {
  const output: string[] = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = `${root}/${entry.name}`;
    if (entry.isDirectory()) output.push(...await files(path));
    else if (entry.isFile() && !["ARTIFACT_INVENTORY.json", "MIRROR_RECEIPT.json"].includes(entry.name)) output.push(path);
  }
  return output;
}
async function digest(path: string): Promise<string> {
  return new Promise((resolve, reject) => { const hash = createHash("sha256"); const stream = createReadStream(path);
    stream.on("data", (value) => hash.update(value)); stream.on("error", reject); stream.on("end", () => resolve(hash.digest("hex"))); });
}
const paths = (await files(runDirectory)).sort(); const artifacts=[];
for (const path of paths) { const info=await stat(path); const relativePath=relative(runDirectory,path); artifacts.push({ path:relativePath,kind:relativePath.split("/")[0] ?? "root",bytes:info.size,sha256:await digest(path) }); }
const inventory={schemaVersion:1,runId,createdAt:new Date().toISOString(),artifacts};
await writeFile(`${runDirectory}/ARTIFACT_INVENTORY.json`,`${JSON.stringify(inventory,null,2)}\n`);
const pool=await new sql.ConnectionPool({server:config.server,port:config.port,user:config.user,password:config.password,database:config.database,
  options:{encrypt:false,trustServerCertificate:true},requestTimeout:3_600_000}).connect();
try {
  for (let start=0;start<artifacts.length;start+=500) await pool.request().input("run",sql.VarChar(120),runId).input("rows",sql.NVarChar(sql.MAX),JSON.stringify(artifacts.slice(start,start+500))).query(`
    MERGE dbo.run_artifacts AS target USING OPENJSON(@rows) WITH(relative_path nvarchar(500) '$.path',artifact_kind varchar(80) '$.kind',byte_count bigint '$.bytes',sha256 char(64) '$.sha256') source
    ON target.run_id=@run AND target.relative_path=source.relative_path WHEN MATCHED THEN UPDATE SET artifact_kind=source.artifact_kind,byte_count=source.byte_count,sha256=source.sha256,created_at=SYSUTCDATETIME()
    WHEN NOT MATCHED THEN INSERT(run_id,relative_path,artifact_kind,byte_count,sha256) VALUES(@run,source.relative_path,source.artifact_kind,source.byte_count,source.sha256);`);
  const status=await pool.request().query<{ campaign_id:number; model_profile_id:string; status:string; rows:number }>(`SELECT j.campaign_id,j.model_profile_id,j.status,COUNT(*) rows FROM dbo.generation_jobs j
    JOIN dbo.campaigns c ON c.campaign_id=j.campaign_id WHERE c.run_id='${runId.replaceAll("'","''")}' GROUP BY j.campaign_id,j.model_profile_id,j.status ORDER BY j.campaign_id,j.model_profile_id,j.status;`);
  const lines=["# ModelPrint Resume Record","",`- Run: \`${runId}\``,`- Run directory: \`${runDirectory}\``,`- Generated: \`${new Date().toISOString()}\``,"","## Generation status","","| Profile | Status | Rows |","|---|---:|---:|",
    ...status.recordset.map((row)=>`| campaign ${row.campaign_id}: ${row.model_profile_id} | ${row.status} | ${row.rows} |`),"","## Resume","","```bash",`cd ${process.cwd()}`,"source scripts/runtime-env.sh",
    `export MODELPRINT_RUN_DIR=${runDirectory}`,"npm run doctor","# Resume the next incomplete residency using config/models.json, then:","npm run features:build","npm run derived:build","npm run evaluate:probes","npm run reports","npm run db:bacpac","```","",
    "Generation, scoring, feature, and export commands are idempotent and use the retained manifests. Never create a replacement freeze when resuming this run.",""];
  await writeFile(`${runDirectory}/RESUME.md`,lines.join("\n"));
} finally { await pool.close(); }
console.log(JSON.stringify({runId,inventory:`${runDirectory}/ARTIFACT_INVENTORY.json`,artifacts:artifacts.length},null,2));
