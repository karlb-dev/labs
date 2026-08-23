-- GhostType control plane: schemas, migrations ledger, runs, campaigns, evidence.
IF SCHEMA_ID(N'control') IS NULL EXEC(N'CREATE SCHEMA control');
IF SCHEMA_ID(N'dataset') IS NULL EXEC(N'CREATE SCHEMA dataset');
IF SCHEMA_ID(N'completion') IS NULL EXEC(N'CREATE SCHEMA completion');
IF SCHEMA_ID(N'retrieval') IS NULL EXEC(N'CREATE SCHEMA retrieval');
IF SCHEMA_ID(N'validation') IS NULL EXEC(N'CREATE SCHEMA validation');
IF SCHEMA_ID(N'eval') IS NULL EXEC(N'CREATE SCHEMA eval');
IF SCHEMA_ID(N'telemetry') IS NULL EXEC(N'CREATE SCHEMA telemetry');
GO

IF OBJECT_ID(N'control.schema_migrations', N'U') IS NULL
  CREATE TABLE control.schema_migrations
  (
    migration_id varchar(120) NOT NULL CONSTRAINT pk_control_schema_migrations PRIMARY KEY,
    migration_sha256 char(64) NOT NULL,
    applied_at_utc datetime2(3) NOT NULL CONSTRAINT df_migrations_applied DEFAULT SYSUTCDATETIME()
  );

IF OBJECT_ID(N'control.runs', N'U') IS NULL
  CREATE TABLE control.runs
  (
    run_id varchar(120) NOT NULL CONSTRAINT pk_control_runs PRIMARY KEY,
    campaign varchar(80) NOT NULL,
    git_commit char(40) NULL,
    started_at_utc datetime2(3) NOT NULL,
    manifest_sha256 char(64) NOT NULL,
    detail_json nvarchar(max) NOT NULL
  );

IF OBJECT_ID(N'control.campaigns', N'U') IS NULL
  CREATE TABLE control.campaigns
  (
    campaign_id int IDENTITY(1,1) NOT NULL CONSTRAINT pk_control_campaigns PRIMARY KEY,
    campaign_name varchar(80) NOT NULL CONSTRAINT uq_campaign_name UNIQUE,
    status varchar(20) NOT NULL CONSTRAINT df_campaign_status DEFAULT 'building',
    dataset_version varchar(40) NOT NULL,
    registry_sha256 char(64) NULL,
    freeze_sha256 char(64) NULL,
    frozen_at_utc datetime2(3) NULL,
    detail_json nvarchar(max) NOT NULL CONSTRAINT df_campaign_detail DEFAULT N'{}'
  );

IF OBJECT_ID(N'control.capability_snapshots', N'U') IS NULL
  CREATE TABLE control.capability_snapshots
  (
    snapshot_id int IDENTITY(1,1) NOT NULL CONSTRAINT pk_capability_snapshots PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    sql_product_version nvarchar(80) NOT NULL,
    sql_edition nvarchar(160) NOT NULL,
    emulated bit NOT NULL,
    vector_supported bit NOT NULL,
    fulltext_supported bit NOT NULL,
    preview_features bit NOT NULL,
    snapshot_json nvarchar(max) NOT NULL,
    snapshot_sha256 char(64) NOT NULL,
    created_at_utc datetime2(3) NOT NULL CONSTRAINT df_capsnap_created DEFAULT SYSUTCDATETIME()
  );

IF OBJECT_ID(N'control.model_profiles', N'U') IS NULL
  CREATE TABLE control.model_profiles
  (
    profile_id varchar(80) NOT NULL CONSTRAINT pk_model_profiles PRIMARY KEY,
    family varchar(40) NOT NULL,
    base_model_id nvarchar(200) NOT NULL,
    served_model_id nvarchar(200) NOT NULL,
    base_url nvarchar(200) NOT NULL,
    quantization varchar(40) NOT NULL,
    registry_role varchar(40) NOT NULL,
    port_gate_status varchar(20) NULL,
    port_gate_json nvarchar(max) NULL,
    profile_json nvarchar(max) NOT NULL
  );

IF OBJECT_ID(N'control.evidence_events', N'U') IS NULL
  CREATE TABLE control.evidence_events
  (
    event_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_evidence_events PRIMARY KEY,
    event_key varchar(160) NOT NULL CONSTRAINT uq_evidence_key UNIQUE,
    run_id varchar(120) NOT NULL,
    stage varchar(40) NOT NULL,
    disposition varchar(40) NOT NULL,
    detail_json nvarchar(max) NOT NULL,
    detail_sha256 char(64) NOT NULL,
    created_at_utc datetime2(3) NOT NULL CONSTRAINT df_evidence_created DEFAULT SYSUTCDATETIME()
  );

IF OBJECT_ID(N'control.run_artifacts', N'U') IS NULL
  CREATE TABLE control.run_artifacts
  (
    run_id varchar(120) NOT NULL,
    relative_path nvarchar(500) NOT NULL,
    artifact_kind varchar(80) NOT NULL,
    byte_count bigint NOT NULL,
    sha256 char(64) NOT NULL,
    created_at datetime2(3) NOT NULL CONSTRAINT df_artifact_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_run_artifacts PRIMARY KEY (run_id, relative_path)
  );
GO
