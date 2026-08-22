import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { normalizeText } from "../src/prompt-bank.js";
import { sha256 } from "../src/hash.js";

const config = loadConfig().database;
const pool = await new sql.ConnectionPool({ server: config.server, port: config.port, user: config.user, password: config.password,
  database: config.database, options: { encrypt: false, trustServerCertificate: true }, requestTimeout: 600_000 }).connect();
const checks: Record<string, unknown> = {};
try {
  await pool.request().batch(`
    DROP TABLE IF EXISTS dbo.__mp_sql_integration;
    CREATE TABLE dbo.__mp_sql_integration(id int NOT NULL PRIMARY KEY CLUSTERED, embedding vector(3) NOT NULL);
    INSERT dbo.__mp_sql_integration VALUES (1,'[1,0,0]'),(2,'[0.8,0.2,0]'),(3,'[0,1,0]');`);
  const exact = await pool.request().input("query", sql.NVarChar(sql.MAX), "[1,0,0]").query<{ id: number; distance: number }>(`
    SELECT id,VECTOR_DISTANCE('cosine',embedding,CAST(@query AS vector(3))) AS distance
    FROM dbo.__mp_sql_integration ORDER BY distance,id;`);
  if (exact.recordset.map((row) => row.id).join(",") !== "1,2,3") throw new Error("Exact vector ranking fixture failed");
  if (Math.abs(Number(exact.recordset[0]?.distance)) > 1e-7) throw new Error("Self distance fixture failed");
  checks.exactVectorRanking = exact.recordset;
  let rejectedDimension = false;
  try { await pool.request().query("INSERT dbo.__mp_sql_integration VALUES(4,'[1,2]');"); } catch { rejectedDimension = true; }
  if (!rejectedDimension) throw new Error("Bad vector dimension was accepted");
  checks.dimensionRejection = true;

  const generation = await pool.request().query<{ job_key: string; final_text: string }>(`
    SELECT TOP (1) j.job_key,g.final_text FROM dbo.generations g JOIN dbo.generation_jobs j ON j.generation_job_id=g.generation_job_id ORDER BY g.generation_id DESC;`);
  if (generation.recordset[0]) {
    const row = generation.recordset[0]; const text = normalizeText(row.final_text);
    const artifactRows = [{ jobKey: row.job_key, textView: "sql-integration-v1", hash: sha256(text), text }];
    const transaction = new sql.Transaction(pool); await transaction.begin();
    try {
      await new sql.Request(transaction).input("rows", sql.NVarChar(sql.MAX), JSON.stringify(artifactRows)).query(`
        INSERT dbo.text_artifacts(text_view_id,normalized_sha256,artifact_text)
        SELECT s.text_view,s.hash,MIN(s.text) FROM OPENJSON(@rows) WITH (text_view varchar(80) '$.textView', hash char(64) '$.hash', text nvarchar(max) '$.text') s
        WHERE NOT EXISTS (SELECT 1 FROM dbo.text_artifacts t WHERE t.text_view_id=s.text_view AND t.normalized_sha256=s.hash)
        GROUP BY s.text_view,s.hash;
        INSERT dbo.generation_text_artifacts(generation_id,text_artifact_id)
        SELECT DISTINCT g.generation_id,t.text_artifact_id FROM OPENJSON(@rows) WITH (jobKey char(64) '$.jobKey', text_view varchar(80) '$.textView', hash char(64) '$.hash') s
        JOIN dbo.generation_jobs j ON j.job_key=s.jobKey JOIN dbo.generations g ON g.generation_job_id=j.generation_job_id
        JOIN dbo.text_artifacts t ON t.text_view_id=s.text_view AND t.normalized_sha256=s.hash
        WHERE NOT EXISTS (SELECT 1 FROM dbo.generation_text_artifacts m WHERE m.generation_id=g.generation_id AND m.text_artifact_id=t.text_artifact_id);`);
      const mapped = await new sql.Request(transaction).input("job", sql.Char(64), row.job_key).query<{ count: number }>(`
        SELECT COUNT(*) AS count FROM dbo.generation_text_artifacts m JOIN dbo.generations g ON g.generation_id=m.generation_id
        JOIN dbo.generation_jobs j ON j.generation_job_id=g.generation_job_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id
        WHERE j.job_key=@job AND t.text_view_id='sql-integration-v1';`);
      if (mapped.recordset[0]?.count !== 1) throw new Error("Set-based text-artifact mapping failed");
      checks.setBasedArtifactPersistence = true;
    } finally { await transaction.rollback(); }
  } else checks.setBasedArtifactPersistence = "skipped-no-generation-fixture";
} finally {
  await pool.request().batch("DROP TABLE IF EXISTS dbo.__mp_sql_integration;");
  await pool.close();
}
console.log(JSON.stringify({ disposition: "PASS", checks }, null, 2));
