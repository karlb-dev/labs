import { readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson, sha256 } from "../src/hash.js";
import { appendExperimentLog, resolveRunDirectory } from "../src/run.js";
import { SqlServerRepository } from "../src/repository.js";

type Probe = { supported: boolean; detail?: unknown; error?: string; errorNumber?: number };

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(readFileSync(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const repository = await SqlServerRepository.connect(config.database);
const pool = repository.pool;

async function probe(operation: () => Promise<unknown>): Promise<Probe> {
  try { return { supported: true, detail: await operation() }; }
  catch (error) {
    const value = error as { message?: string; number?: number; originalError?: { info?: { number?: number } } };
    const errorNumber = value.number ?? value.originalError?.info?.number;
    return errorNumber === undefined
      ? { supported: false, error: value.message ?? String(error) }
      : { supported: false, error: value.message ?? String(error), errorNumber };
  }
}

async function rows(query: string): Promise<unknown> {
  const result = await pool.request().query(query);
  return result.recordsets;
}

const capabilities: Record<string, unknown> = { schemaVersion: 1, runId: run.runId, checkedAt: new Date().toISOString() };
try {
  const server = await pool.request().query(`
    SELECT
      @@VERSION AS full_version,
      CONVERT(VARCHAR(64), SERVERPROPERTY('ProductVersion')) AS product_version,
      CONVERT(VARCHAR(128), SERVERPROPERTY('Edition')) AS edition,
      CONVERT(VARCHAR(128), SERVERPROPERTY('ProductLevel')) AS product_level,
      d.compatibility_level,
      (SELECT value FROM sys.database_scoped_configurations WHERE name='PREVIEW_FEATURES') AS preview_features,
      (SELECT value_in_use FROM sys.configurations WHERE name='external rest endpoint enabled') AS external_rest_endpoint_enabled
    FROM sys.databases d WHERE d.name=DB_NAME();
  `);
  capabilities.server = server.recordset[0];
  capabilities.previewEnable = await probe(async () => rows("ALTER DATABASE SCOPED CONFIGURATION SET PREVIEW_FEATURES = ON;"));
  capabilities.vectorType = await probe(async () => rows("DECLARE @v VECTOR(3)=CAST(N'[1,2,3]' AS VECTOR(3)); SELECT CAST(@v AS NVARCHAR(MAX)) AS value;"));
  capabilities.vectorDistance = await probe(async () => rows("SELECT VECTOR_DISTANCE('cosine', CAST(N'[1,0,0]' AS VECTOR(3)), CAST(N'[0,1,0]' AS VECTOR(3))) AS distance;"));
  capabilities.vectorFunctions = await probe(async () => rows(`
    DECLARE @v VECTOR(3)=CAST(N'[3,4,0]' AS VECTOR(3));
    SELECT VECTOR_NORM(@v,'norm2') AS norm,
      CAST(VECTOR_NORMALIZE(@v,'norm2') AS NVARCHAR(MAX)) AS normalized,
      VECTORPROPERTY(@v,'Dimensions') AS dimensions,
      VECTORPROPERTY(@v,'BaseType') AS base_type;
  `));
  capabilities.regex = await probe(async () => rows(`SELECT CASE WHEN EXISTS(SELECT 1 WHERE REGEXP_LIKE(N'Alpha 123',N'^[A-Za-z]+')) THEN 1 ELSE 0 END AS like_ok, REGEXP_COUNT(N'a a a',N'a') AS count_ok, REGEXP_SUBSTR(N'abc123',N'[0-9]+') AS substr_ok;`));
  capabilities.regexSplit = await probe(async () => rows(`SELECT * FROM REGEXP_SPLIT_TO_TABLE(N'a,b,c',N',');`));
  capabilities.aiGenerateChunks = await probe(async () => rows(`
    WITH n AS (SELECT TOP (100) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS id FROM sys.all_objects)
    SELECT COUNT(*) AS source_rows, SUM(c.chunk_length) AS chunk_chars, MAX(c.chunk_order) AS max_order
    FROM n CROSS APPLY AI_GENERATE_CHUNKS(SOURCE=CONCAT(N'row ',n.id,N' alpha beta gamma delta'),CHUNK_TYPE=FIXED,CHUNK_SIZE=12,OVERLAP=10,ENABLE_CHUNK_SET_ID=1) c;
  `));
  capabilities.createExternalModelParses = await probe(async () => rows(`
    SET PARSEONLY ON;
    CREATE EXTERNAL MODEL ModelPrintParseProbe
    WITH (LOCATION='https://example.invalid/v1/embeddings', API_FORMAT='OpenAI', MODEL_TYPE=EMBEDDINGS, MODEL='probe');
    SET PARSEONLY OFF;
  `));
  capabilities.maxVectorDimension = {
    dimension1998: await probe(async () => rows("CREATE TABLE dbo.__mp_dim_1998(id INT PRIMARY KEY, v VECTOR(1998)); DROP TABLE dbo.__mp_dim_1998;")),
    dimension1999: await probe(async () => rows("CREATE TABLE dbo.__mp_dim_1999(id INT PRIMARY KEY, v VECTOR(1999)); DROP TABLE dbo.__mp_dim_1999;")),
  };

  await pool.request().batch(`
    IF OBJECT_ID(N'dbo.__mp_vector_probe',N'U') IS NOT NULL DROP TABLE dbo.__mp_vector_probe;
    CREATE TABLE dbo.__mp_vector_probe(id INT NOT NULL PRIMARY KEY CLUSTERED, label VARCHAR(20) NOT NULL, embedding VECTOR(3) NOT NULL);
    WITH n AS (SELECT TOP (100) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS id FROM sys.all_objects)
    INSERT dbo.__mp_vector_probe(id,label,embedding)
    SELECT id, CASE WHEN id%2=0 THEN 'even' ELSE 'odd' END,
      CAST(CONCAT('[',CONVERT(VARCHAR(30),id/100.0),',',CONVERT(VARCHAR(30),(101-id)/100.0),',',CONVERT(VARCHAR(30),(id%17)/17.0),']') AS VECTOR(3))
    FROM n;
  `);
  capabilities.createVectorIndex = await probe(async () => rows("CREATE VECTOR INDEX __mp_vector_probe_idx ON dbo.__mp_vector_probe(embedding) WITH (TYPE='DISKANN',METRIC='COSINE');"));
  const indexMetadata = await probe(async () => rows(`
    SELECT i.name AS index_name, t.name AS table_name,
      JSON_VALUE(v.build_parameters,'$.Version') AS documented_index_version,
      COALESCE(JSON_VALUE(v.build_parameters,'$.Version'),JSON_VALUE(v.build_parameters,'$.version')) AS index_version,
      v.build_parameters, v.vector_index_type, v.distance_metric
    FROM sys.vector_indexes v
    JOIN sys.indexes i ON i.object_id=v.object_id AND i.index_id=v.index_id
    JOIN sys.tables t ON t.object_id=v.object_id
    WHERE i.name='__mp_vector_probe_idx';
  `));
  capabilities.vectorIndexMetadata = indexMetadata;
  capabilities.vectorIndexMaintenanceDmv = await probe(async () => rows(`SELECT * FROM sys.dm_db_vector_indexes WHERE object_id=OBJECT_ID(N'dbo.__mp_vector_probe');`));
  const metadataRows = indexMetadata.supported
    ? (indexMetadata.detail as Array<Array<Record<string, unknown>>> | undefined)
    : undefined;
  const version = metadataRows?.[0]?.[0]?.index_version;
  capabilities.detectedIndexVersion = typeof version === "string" && version ? version : "legacy-unversioned";
  capabilities.vectorSearchLegacy = await probe(async () => rows(`
    DECLARE @q VECTOR(3)=CAST(N'[0.1,0.9,0.2]' AS VECTOR(3));
    SELECT p.id,p.label,s.distance FROM VECTOR_SEARCH(TABLE=dbo.__mp_vector_probe AS p,COLUMN=embedding,SIMILAR_TO=@q,METRIC='cosine',TOP_N=10) s ORDER BY s.distance;
  `));
  capabilities.vectorSearchV3 = await probe(async () => rows(`
    DECLARE @q VECTOR(3)=CAST(N'[0.1,0.9,0.2]' AS VECTOR(3));
    SELECT TOP (10) WITH APPROXIMATE p.id,p.label,s.distance FROM VECTOR_SEARCH(TABLE=dbo.__mp_vector_probe AS p,COLUMN=embedding,SIMILAR_TO=@q,METRIC='cosine') s ORDER BY s.distance;
  `));
  capabilities.forceAnnOnly = await probe(async () => rows(`
    DECLARE @q VECTOR(3)=CAST(N'[0.1,0.9,0.2]' AS VECTOR(3));
    SELECT TOP (10) WITH APPROXIMATE p.id,s.distance FROM VECTOR_SEARCH(TABLE=dbo.__mp_vector_probe AS p,COLUMN=embedding,SIMILAR_TO=@q,METRIC='cosine') s WITH (FORCE_ANN_ONLY) ORDER BY s.distance;
  `));

  const planProbe = await probe(async () => {
    const result = await pool.request().query(`
      SET STATISTICS XML ON;
      DECLARE @q VECTOR(3)=CAST(N'[0.1,0.9,0.2]' AS VECTOR(3));
      SELECT p.id,s.distance FROM VECTOR_SEARCH(TABLE=dbo.__mp_vector_probe AS p,COLUMN=embedding,SIMILAR_TO=@q,METRIC='cosine',TOP_N=10) s ORDER BY s.distance;
      SET STATISTICS XML OFF;
    `);
    const serialized = JSON.stringify(result.recordsets);
    await writeFile(`${runDirectory}/environment/sql-vector-probe-plan.json`, `${serialized}\n`);
    return { planSha256: sha256(serialized), indexOperatorPresent: serialized.includes("__mp_vector_probe_idx") && serialized.includes('PhysicalOp=\\"Vector Index Seek\\"'), bytes: serialized.length };
  });
  capabilities.queryPlanEvidence = planProbe;
  capabilities.dmlAfterIndex = {
    insert: await probe(async () => rows("INSERT dbo.__mp_vector_probe VALUES(101,'new',CAST(N'[1,0,0]' AS VECTOR(3)));")),
    update: await probe(async () => rows("UPDATE dbo.__mp_vector_probe SET label='updated' WHERE id=1;")),
    delete: await probe(async () => rows("DELETE dbo.__mp_vector_probe WHERE id=2;")),
  };
  capabilities.filtering = await probe(async () => rows(`
    DECLARE @q VECTOR(3)=CAST(N'[0.1,0.9,0.2]' AS VECTOR(3));
    SELECT p.id,p.label,s.distance FROM VECTOR_SEARCH(TABLE=dbo.__mp_vector_probe AS p,COLUMN=embedding,SIMILAR_TO=@q,METRIC='cosine',TOP_N=10) s WHERE p.label='even' ORDER BY s.distance;
  `));

  await pool.request().batch(`
    IF EXISTS (SELECT 1 FROM sys.vector_indexes v JOIN sys.indexes i ON i.object_id=v.object_id AND i.index_id=v.index_id WHERE i.name='__mp_vector_probe_idx') DROP INDEX __mp_vector_probe_idx ON dbo.__mp_vector_probe;
    IF OBJECT_ID(N'dbo.__mp_vector_probe',N'U') IS NOT NULL DROP TABLE dbo.__mp_vector_probe;
    IF OBJECT_ID(N'dbo.__mp_vector_probe_99',N'U') IS NOT NULL DROP TABLE dbo.__mp_vector_probe_99;
    CREATE TABLE dbo.__mp_vector_probe_99(id INT NOT NULL PRIMARY KEY CLUSTERED, embedding VECTOR(3) NOT NULL);
    WITH n AS (SELECT TOP (99) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS id FROM sys.all_objects)
    INSERT dbo.__mp_vector_probe_99 SELECT id,CAST(CONCAT('[',id/100.0,',',1-id/100.0,',0.1]') AS VECTOR(3)) FROM n;
  `);
  capabilities.minimumRowIndex = await probe(async () => rows("CREATE VECTOR INDEX __mp_vector_probe_99_idx ON dbo.__mp_vector_probe_99(embedding) WITH (TYPE='DISKANN',METRIC='COSINE');"));
  capabilities.minimumRowRequirementObserved = !(capabilities.minimumRowIndex as Probe).supported && (capabilities.minimumRowIndex as Probe).errorNumber === 42266;
  capabilities.syntaxSelection =
    typeof version === "string" && Number(version) >= 3 && (capabilities.vectorSearchV3 as Probe).supported
      ? "ann_v3"
      : (capabilities.createVectorIndex as Probe).supported && (capabilities.vectorSearchLegacy as Probe).supported
        ? "ann_legacy"
        : "exact";
  capabilities.primaryAllowStaleVectorIndex = false;
} finally {
  await probe(async () => rows(`
    IF EXISTS (SELECT 1 FROM sys.vector_indexes v JOIN sys.indexes i ON i.object_id=v.object_id AND i.index_id=v.index_id WHERE i.name='__mp_vector_probe_idx') DROP INDEX __mp_vector_probe_idx ON dbo.__mp_vector_probe;
    IF EXISTS (SELECT 1 FROM sys.vector_indexes v JOIN sys.indexes i ON i.object_id=v.object_id AND i.index_id=v.index_id WHERE i.name='__mp_vector_probe_99_idx') DROP INDEX __mp_vector_probe_99_idx ON dbo.__mp_vector_probe_99;
    IF OBJECT_ID(N'dbo.__mp_vector_probe',N'U') IS NOT NULL DROP TABLE dbo.__mp_vector_probe;
    IF OBJECT_ID(N'dbo.__mp_vector_probe_99',N'U') IS NOT NULL DROP TABLE dbo.__mp_vector_probe_99;
    IF OBJECT_ID(N'dbo.__mp_dim_1998',N'U') IS NOT NULL DROP TABLE dbo.__mp_dim_1998;
    IF OBJECT_ID(N'dbo.__mp_dim_1999',N'U') IS NOT NULL DROP TABLE dbo.__mp_dim_1999;
  `));
}

const snapshot: Record<string, unknown> & { snapshotHash: string } = { ...capabilities, snapshotHash: hashJson(capabilities) };
await writeFile(`${runDirectory}/environment/sql-server-capabilities.json`, `${JSON.stringify(snapshot, null, 2)}\n`);
await pool.request().input("runId", sql.VarChar(120), run.runId).input("hash", sql.Char(64), snapshot.snapshotHash)
  .input("json", sql.NVarChar(sql.MAX), JSON.stringify(snapshot))
  .query("IF NOT EXISTS(SELECT 1 FROM dbo.capability_snapshots WHERE snapshot_sha256=@hash) INSERT dbo.capability_snapshots(run_id,snapshot_sha256,snapshot_json) VALUES(@runId,@hash,@json);");
await repository.close();
await appendExperimentLog(`MP-0 SQL capability doctor completed for ${run.runId}; mode=${String(snapshot.syntaxSelection)}; hash=${snapshot.snapshotHash}.`);
console.log(JSON.stringify({ path: `${runDirectory}/environment/sql-server-capabilities.json`, snapshotHash: snapshot.snapshotHash, mode: snapshot.syntaxSelection }, null, 2));
