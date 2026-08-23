import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { applyControlMigrations, applyUntrackedMigrations } from "../src/migrations.js";
import { loadModelRegistry } from "../src/models.js";
import { connect, sqlIdentifier } from "../src/repository.js";
import { LAB_ROOT, atomicWrite, resolveRunDirectory } from "../src/run.js";

interface RunManifest {
  runId: string;
  campaign: string;
  startedAt: string;
  runManifestHash: string;
  inputs: {
    specSha256: string;
    addendumSha256: string;
    modelRegistrySha256: string;
  };
  git: { commit: string | null };
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const controlName = sqlIdentifier(config.databases.controlName);
const workloadName = sqlIdentifier(config.databases.workloadName);

const master = await connect(config.databases.admin, "master");
try {
  await master.request().batch(`
    IF DB_ID(N'${config.databases.controlName}') IS NULL CREATE DATABASE ${controlName};
    IF DB_ID(N'${config.databases.workloadName}') IS NULL CREATE DATABASE ${workloadName};
    ALTER DATABASE ${controlName} SET COMPATIBILITY_LEVEL = 170;
    ALTER DATABASE ${workloadName} SET COMPATIBILITY_LEVEL = 170;
    ALTER DATABASE ${controlName} SET READ_COMMITTED_SNAPSHOT ON WITH ROLLBACK IMMEDIATE;
    ALTER DATABASE ${workloadName} SET READ_COMMITTED_SNAPSHOT ON WITH ROLLBACK IMMEDIATE;
    ALTER DATABASE ${controlName} SET AUTO_UPDATE_STATISTICS_ASYNC OFF;
    ALTER DATABASE ${workloadName} SET AUTO_UPDATE_STATISTICS_ASYNC OFF;
    ALTER DATABASE ${controlName} SET QUERY_STORE = ON;
    ALTER DATABASE ${workloadName} SET QUERY_STORE = ON;
    ALTER DATABASE ${controlName} SET QUERY_STORE
    (
      OPERATION_MODE = READ_WRITE,
      QUERY_CAPTURE_MODE = ALL,
      DATA_FLUSH_INTERVAL_SECONDS = 60,
      INTERVAL_LENGTH_MINUTES = 1,
      MAX_STORAGE_SIZE_MB = 1024,
      WAIT_STATS_CAPTURE_MODE = ON
    );
    ALTER DATABASE ${workloadName} SET QUERY_STORE
    (
      OPERATION_MODE = READ_WRITE,
      QUERY_CAPTURE_MODE = ALL,
      DATA_FLUSH_INTERVAL_SECONDS = 60,
      INTERVAL_LENGTH_MINUTES = 1,
      MAX_STORAGE_SIZE_MB = 1024,
      WAIT_STATS_CAPTURE_MODE = ON
    );
    EXEC sys.sp_configure N'show advanced options', 1;
    RECONFIGURE;
    EXEC sys.sp_configure N'blocked process threshold (s)', 5;
    RECONFIGURE;
  `);

  await upsertLogin(master, "lw_lab", config.databases.lab.password, controlName);
  await upsertLogin(master, "lw_agent", config.databases.agent.password, controlName);
  await master.request().batch(`
    GRANT ALTER TRACE TO [lw_lab];
    GRANT VIEW SERVER STATE TO [lw_lab];
    GRANT VIEW SERVER PERFORMANCE STATE TO [lw_lab];
    GRANT ALTER ANY EVENT SESSION TO [lw_lab];
    GRANT CREATE ANY DATABASE TO [lw_lab];
    GRANT ALTER ANY DATABASE TO [lw_lab];
  `);
  await master.request().batch(`
    USE ${workloadName};
    IF USER_ID(N'lw_lab') IS NULL CREATE USER [lw_lab] FOR LOGIN [lw_lab];
    IF IS_ROLEMEMBER(N'db_owner', N'lw_lab') <> 1 ALTER ROLE [db_owner] ADD MEMBER [lw_lab];
  `);
} finally {
  await master.close();
}

const controlPool = await connect(config.databases.admin, config.databases.controlName);
let controlMigrations;
try {
  const masterKeyPassword = `Lw!${sha256(`logwarden-control-master-key:${config.databases.lab.password}`)}`;
  await controlPool.request()
    .input("password", sql.NVarChar(128), masterKeyPassword)
    .query(`
      IF NOT EXISTS (SELECT 1 FROM sys.symmetric_keys WHERE name=N'##MS_DatabaseMasterKey##')
      BEGIN
        DECLARE @statement nvarchar(max) = N'CREATE MASTER KEY ENCRYPTION BY PASSWORD=' + QUOTENAME(@password,N'''');
        EXEC sys.sp_executesql @statement;
      END;
    `);
  controlMigrations = await applyControlMigrations(controlPool, `${LAB_ROOT}/db/migrations`, run.runId);
  await seedControlMetadata(controlPool, run);
} finally {
  await controlPool.close();
}

const workloadPool = await connect(config.databases.admin, config.databases.workloadName);
let workloadMigrations;
try {
  workloadMigrations = await applyUntrackedMigrations(workloadPool, `${LAB_ROOT}/db/workload`);
} finally {
  await workloadPool.close();
}

const serverPool = await connect(config.databases.admin, "master");
let serverMigrations;
try {
  serverMigrations = await applyUntrackedMigrations(serverPool, `${LAB_ROOT}/db/server`);
} finally {
  await serverPool.close();
}

const receipt = {
  schemaVersion: 1,
  runId: run.runId,
  createdAt: new Date().toISOString(),
  databases: {
    control: config.databases.controlName,
    workload: config.databases.workloadName,
    compatibilityLevel: 170,
    queryStoreIntervalMinutes: 1,
  },
  principals: ["lw_lab", "lw_agent"],
  controlMigrations,
  workloadMigrations,
  serverMigrations,
};
await atomicWrite(`${runDirectory}/database/setup.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));

async function upsertLogin(
  pool: sql.ConnectionPool,
  login: "lw_lab" | "lw_agent",
  password: string,
  defaultDatabase: string,
): Promise<void> {
  await pool.request()
    .input("password", sql.NVarChar(128), password)
    .query(`
      DECLARE @statement nvarchar(max);
      IF SUSER_ID(N'${login}') IS NULL
        SET @statement = N'CREATE LOGIN [${login}] WITH PASSWORD = ' + QUOTENAME(@password, N'''') +
          N', CHECK_POLICY = ON, CHECK_EXPIRATION = OFF, DEFAULT_DATABASE = ${defaultDatabase}';
      ELSE
        SET @statement = N'ALTER LOGIN [${login}] WITH PASSWORD = ' + QUOTENAME(@password, N'''') +
          N', CHECK_POLICY = ON, CHECK_EXPIRATION = OFF, DEFAULT_DATABASE = ${defaultDatabase}';
      EXEC sys.sp_executesql @statement;
    `);
}

async function seedControlMetadata(pool: sql.ConnectionPool, run: RunManifest): Promise<void> {
  await seedIngestionSources(pool);
  const registry = loadModelRegistry();
  for (const [profileId, profile] of Object.entries(registry.profiles)) {
    const profileJson = canonicalJson(profile);
    await pool.request()
      .input("id", sql.VarChar(80), profileId)
      .input("hash", sql.Char(64), hashJson(profile))
      .input("model", sql.NVarChar(300), profile.modelId)
      .input("revision", sql.Char(40), profile.revision)
      .input("image", sql.VarChar(160), profile.vllmImage)
      .input("json", sql.NVarChar(sql.MAX), profileJson)
      .query(`
        IF NOT EXISTS (SELECT 1 FROM control.model_profiles WHERE model_profile_id = @id)
          INSERT control.model_profiles(model_profile_id, profile_hash, model_id, revision, tokenizer_revision, image_digest, config_json)
          VALUES(@id, @hash, @model, @revision, @revision, @image, @json);
      `);
  }
  for (const [profileId, profile] of Object.entries(registry.embeddingProfiles)) {
    const profileJson = canonicalJson(profile);
    await pool.request()
      .input("id", sql.VarChar(80), profileId)
      .input("hash", sql.Char(64), hashJson(profile))
      .input("model", sql.NVarChar(300), profile.modelId)
      .input("revision", sql.Char(40), profile.revision)
      .input("image", sql.VarChar(160), profile.vllmImage)
      .input("dimensions", sql.Int, profile.dimensions)
      .input("json", sql.NVarChar(sql.MAX), profileJson)
      .query(`
        IF NOT EXISTS (SELECT 1 FROM control.embedding_profiles WHERE embedding_profile_id = @id)
          INSERT control.embedding_profiles(embedding_profile_id, profile_hash, model_id, revision, image_digest, dimensions, config_json)
          VALUES(@id, @hash, @model, @revision, @image, @dimensions, @json);
      `);
  }

  const decode = {
    temperature: 0,
    top_p: 1,
    seed: 0,
    max_tokens: 512,
    stream: false,
    transport: "structured_json",
  };
  await pool.request()
    .input("hash", sql.Char(64), hashJson(decode))
    .input("json", sql.NVarChar(sql.MAX), canonicalJson(decode))
    .query(`
      IF NOT EXISTS (SELECT 1 FROM control.decode_configs WHERE decode_config_id = 'primary-json-v1')
        INSERT control.decode_configs(decode_config_id, decode_hash, requested_json)
        VALUES('primary-json-v1', @hash, @json);
    `);

  const primaryDecode = {
    temperature: 0,
    top_p: 1,
    seed: 0,
    max_tokens: 900,
    stream: false,
    transport: "structured_json",
  };
  await pool.request()
    .input("hash", sql.Char(64), hashJson(primaryDecode))
    .input("json", sql.NVarChar(sql.MAX), canonicalJson(primaryDecode))
    .query(`
      IF NOT EXISTS (SELECT 1 FROM control.decode_configs WHERE decode_config_id = 'primary-json-v2')
        INSERT control.decode_configs(decode_config_id, decode_hash, requested_json)
        VALUES('primary-json-v2', @hash, @json);
    `);

  const campaignIdentity = {
    name: "logwarden-tier1-building",
    tier: run.campaign,
    governingSpecHash: run.inputs.addendumSha256,
    modelRegistryHash: run.inputs.modelRegistrySha256,
  };
  const campaign = await pool.request()
    .input("name", sql.VarChar(120), campaignIdentity.name)
    .input("tier", sql.VarChar(24), run.campaign)
    .input("hash", sql.Char(64), hashJson(campaignIdentity))
    .input("spec_hash", sql.Char(64), run.inputs.addendumSha256)
    .input("registry_hash", sql.Char(64), run.inputs.modelRegistrySha256)
    .input("json", sql.NVarChar(sql.MAX), canonicalJson(campaignIdentity))
    .query<{ campaign_id: number }>(`
      IF NOT EXISTS (SELECT 1 FROM control.campaigns WHERE campaign_hash = @hash)
        INSERT control.campaigns(campaign_name, tier, campaign_hash, governing_spec_hash, model_registry_hash, status, config_json)
        VALUES(@name, @tier, @hash, @spec_hash, @registry_hash, 'building', @json);
      SELECT campaign_id FROM control.campaigns WHERE campaign_hash = @hash;
    `);
  const campaignId = campaign.recordset[0]?.campaign_id;
  if (campaignId === undefined) throw new Error("Failed to resolve building campaign");

  await pool.request()
    .input("run_id", sql.VarChar(120), run.runId)
    .input("campaign_id", sql.BigInt, campaignId)
    .input("config_hash", sql.Char(64), run.runManifestHash)
    .input("git_commit", sql.Char(40), run.git.commit ?? "0000000000000000000000000000000000000000")
    .input("started", sql.DateTime2(7), new Date(run.startedAt))
    .query(`
      IF NOT EXISTS (SELECT 1 FROM control.runs WHERE run_id = @run_id)
        INSERT control.runs(run_id, campaign_id, run_kind, status, config_hash, git_commit, started_at_utc, notes)
        VALUES(@run_id, @campaign_id, 'capture', 'initialized', @config_hash, @git_commit, @started, 'LW-0 foundation run; no scientific freeze');
    `);
}

async function seedIngestionSources(pool: sql.ConnectionPool): Promise<void> {
  const xePath = `${LAB_ROOT}/db/server/007_xe_capture_contract_v3.sql`;
  const errorlogParserPath = `${LAB_ROOT}/src/errorlog.ts`;
  const errorlogParserSha256 = sha256(await readFile(errorlogParserPath, "utf8"));
  const definitions = [
    {
      id: "xe-logwarden-capture",
      kind: "xe_event_file",
      name: "logwarden_capture",
      hash: sha256(await readFile(xePath, "utf8")),
      config: {
        schemaVersion: 1,
        targetPattern: "/var/opt/mssql/log/logwarden_capture*.xel",
        cursor: "file_name+file_offset",
        dispatchLatencySeconds: 1,
        requestedEventRetentionMode: "NO_EVENT_LOSS",
        actualEventRetentionMode: "ALLOW_SINGLE_EVENT_LOSS",
        retentionAdaptation: "SQL Server error 25643: error_reported cannot be added to a NO_EVENT_LOSS session",
        maxFileSizeMb: 16,
        maxRolloverFiles: 20,
        definitionPath: "db/server/007_xe_capture_contract_v3.sql",
      },
    },
    {
      id: "errorlog",
      kind: "errorlog_file_tail",
      name: "SQL Server ERRORLOG",
      hash: hashJson({ parser: "errorlog-file-v1", parserSha256: errorlogParserSha256, cursor: "generation+ordinal", transport: "docker-exec-read-only" }),
      config: {
        schemaVersion: 1,
        files: "/var/opt/mssql/log/errorlog*",
        parserVersion: "errorlog-file-v1",
        parserSha256: errorlogParserSha256,
        parserPath: "src/errorlog.ts",
        cursor: "generation_identity+ordinal",
        transport: "docker-exec-read-only",
      },
    },
  ];
  for (const definition of definitions) {
    await pool.request()
      .input("id", sql.VarChar(80), definition.id)
      .input("kind", sql.VarChar(40), definition.kind)
      .input("name", sql.VarChar(120), definition.name)
      .input("hash", sql.Char(64), definition.hash)
      .input("json", sql.NVarChar(sql.MAX), canonicalJson(definition.config))
      .query(`
        IF NOT EXISTS (SELECT 1 FROM ingest.sources WHERE source_id=@id)
          INSERT ingest.sources(source_id, source_kind, source_name, definition_sha256, config_json)
          VALUES(@id, @kind, @name, @hash, @json);
        ELSE
          UPDATE ingest.sources
          SET source_kind=@kind, source_name=@name, definition_sha256=@hash, config_json=@json
          WHERE source_id=@id;
        IF NOT EXISTS (SELECT 1 FROM ingest.source_cursors WHERE source_id=@id)
          INSERT ingest.source_cursors(source_id, cursor_json) VALUES(@id, N'{}');
      `);
  }
}
