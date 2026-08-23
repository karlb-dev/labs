SET XACT_ABORT ON;

IF SCHEMA_ID(N'control') IS NULL EXEC(N'CREATE SCHEMA control');
IF SCHEMA_ID(N'workload') IS NULL EXEC(N'CREATE SCHEMA workload');
IF SCHEMA_ID(N'ingest') IS NULL EXEC(N'CREATE SCHEMA ingest');
IF SCHEMA_ID(N'ops') IS NULL EXEC(N'CREATE SCHEMA ops');
IF SCHEMA_ID(N'kb') IS NULL EXEC(N'CREATE SCHEMA kb');
IF SCHEMA_ID(N'agent') IS NULL EXEC(N'CREATE SCHEMA agent');
IF SCHEMA_ID(N'telemetry') IS NULL EXEC(N'CREATE SCHEMA telemetry');
IF SCHEMA_ID(N'eval') IS NULL EXEC(N'CREATE SCHEMA eval');
IF SCHEMA_ID(N'reporting') IS NULL EXEC(N'CREATE SCHEMA reporting');

IF OBJECT_ID(N'control.schema_migrations', N'U') IS NULL
BEGIN
  CREATE TABLE control.schema_migrations
  (
    migration_id varchar(100) NOT NULL CONSTRAINT pk_control_schema_migrations PRIMARY KEY,
    migration_sha256 char(64) NOT NULL,
    applied_at_utc datetime2(7) NOT NULL CONSTRAINT df_control_schema_migrations_applied DEFAULT SYSUTCDATETIME(),
    applied_by_run_id varchar(120) NOT NULL
  );
END;

IF OBJECT_ID(N'control.capability_snapshots', N'U') IS NULL
BEGIN
  CREATE TABLE control.capability_snapshots
  (
    capability_snapshot_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_control_capability_snapshots PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    sql_product_version varchar(80) NOT NULL,
    sql_edition varchar(160) NOT NULL,
    database_compatibility_level int NOT NULL,
    preview_features_enabled bit NOT NULL,
    vector_supported bit NOT NULL,
    vector_index_mode varchar(32) NOT NULL,
    json_type_supported bit NOT NULL,
    json_index_supported bit NOT NULL,
    ai_chunks_supported bit NOT NULL,
    ai_embeddings_supported bit NOT NULL,
    fulltext_supported bit NOT NULL,
    query_store_state varchar(32) NOT NULL,
    xe_permissions_ok bit NOT NULL,
    snapshot_json nvarchar(max) NOT NULL,
    snapshot_sha256 char(64) NOT NULL,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_control_capabilities_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_control_capabilities_hash UNIQUE(snapshot_sha256),
    CONSTRAINT ck_control_capabilities_json CHECK (ISJSON(snapshot_json) = 1)
  );
END;

IF OBJECT_ID(N'control.campaigns', N'U') IS NULL
BEGIN
  CREATE TABLE control.campaigns
  (
    campaign_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_control_campaigns PRIMARY KEY,
    campaign_name varchar(120) NOT NULL,
    tier varchar(24) NOT NULL,
    campaign_hash char(64) NOT NULL CONSTRAINT uq_control_campaigns_hash UNIQUE,
    governing_spec_hash char(64) NOT NULL,
    model_registry_hash char(64) NOT NULL,
    scenario_manifest_hash char(64) NULL,
    runbook_manifest_hash char(64) NULL,
    agent_arms_hash char(64) NULL,
    status varchar(24) NOT NULL,
    config_json nvarchar(max) NOT NULL,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_control_campaigns_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT ck_control_campaigns_tier CHECK (tier IN ('smoke','dev','standard','full')),
    CONSTRAINT ck_control_campaigns_status CHECK (status IN ('building','frozen','running','complete','stopped','invalid')),
    CONSTRAINT ck_control_campaigns_config_json CHECK (ISJSON(config_json) = 1)
  );
END;

IF OBJECT_ID(N'control.campaign_freezes', N'U') IS NULL
BEGIN
  CREATE TABLE control.campaign_freezes
  (
    freeze_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_control_campaign_freezes PRIMARY KEY,
    campaign_id bigint NOT NULL,
    freeze_hash char(64) NOT NULL CONSTRAINT uq_control_campaign_freezes_hash UNIQUE,
    manifests_json nvarchar(max) NOT NULL,
    git_commit char(40) NOT NULL,
    database_checkpoint_id varchar(120) NOT NULL,
    frozen_at_utc datetime2(7) NOT NULL CONSTRAINT df_control_freezes_at DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_control_freezes_campaign FOREIGN KEY(campaign_id) REFERENCES control.campaigns(campaign_id),
    CONSTRAINT ck_control_freezes_json CHECK (ISJSON(manifests_json) = 1)
  );
END;

IF OBJECT_ID(N'control.model_profiles', N'U') IS NULL
BEGIN
  CREATE TABLE control.model_profiles
  (
    model_profile_id varchar(80) NOT NULL CONSTRAINT pk_control_model_profiles PRIMARY KEY,
    profile_hash char(64) NOT NULL CONSTRAINT uq_control_model_profiles_hash UNIQUE,
    model_id nvarchar(300) NOT NULL,
    revision char(40) NOT NULL,
    tokenizer_revision char(40) NOT NULL,
    image_digest varchar(160) NOT NULL,
    chat_template_hash char(64) NULL,
    config_json nvarchar(max) NOT NULL,
    CONSTRAINT ck_control_model_profiles_json CHECK (ISJSON(config_json) = 1)
  );
END;

IF OBJECT_ID(N'control.embedding_profiles', N'U') IS NULL
BEGIN
  CREATE TABLE control.embedding_profiles
  (
    embedding_profile_id varchar(80) NOT NULL CONSTRAINT pk_control_embedding_profiles PRIMARY KEY,
    profile_hash char(64) NOT NULL CONSTRAINT uq_control_embedding_profiles_hash UNIQUE,
    model_id nvarchar(300) NOT NULL,
    revision char(40) NOT NULL,
    image_digest varchar(160) NOT NULL,
    dimensions int NOT NULL,
    config_json nvarchar(max) NOT NULL,
    CONSTRAINT ck_control_embedding_profiles_dimensions CHECK (dimensions > 0),
    CONSTRAINT ck_control_embedding_profiles_json CHECK (ISJSON(config_json) = 1)
  );
END;

IF OBJECT_ID(N'control.decode_configs', N'U') IS NULL
BEGIN
  CREATE TABLE control.decode_configs
  (
    decode_config_id varchar(40) NOT NULL CONSTRAINT pk_control_decode_configs PRIMARY KEY,
    decode_hash char(64) NOT NULL CONSTRAINT uq_control_decode_configs_hash UNIQUE,
    requested_json nvarchar(max) NOT NULL,
    effective_json nvarchar(max) NULL,
    CONSTRAINT ck_control_decode_requested_json CHECK (ISJSON(requested_json) = 1),
    CONSTRAINT ck_control_decode_effective_json CHECK (effective_json IS NULL OR ISJSON(effective_json) = 1)
  );
END;

IF OBJECT_ID(N'control.agent_arms', N'U') IS NULL
BEGIN
  CREATE TABLE control.agent_arms
  (
    agent_arm_id varchar(80) NOT NULL CONSTRAINT pk_control_agent_arms PRIMARY KEY,
    arm_hash char(64) NOT NULL CONSTRAINT uq_control_agent_arms_hash UNIQUE,
    prompt_sha256 char(64) NOT NULL,
    tool_registry_sha256 char(64) NOT NULL,
    policy_sha256 char(64) NOT NULL,
    contract_sha256 char(64) NOT NULL,
    packet_version varchar(40) NOT NULL,
    retrieval_mode varchar(40) NOT NULL,
    correlation_mode varchar(40) NOT NULL,
    config_json nvarchar(max) NOT NULL,
    CONSTRAINT ck_control_agent_arms_json CHECK (ISJSON(config_json) = 1)
  );
END;

IF OBJECT_ID(N'control.runs', N'U') IS NULL
BEGIN
  CREATE TABLE control.runs
  (
    run_id varchar(120) NOT NULL CONSTRAINT pk_control_runs PRIMARY KEY,
    campaign_id bigint NULL,
    run_kind varchar(40) NOT NULL,
    model_profile_id varchar(80) NULL,
    agent_arm_id varchar(80) NULL,
    decode_config_id varchar(40) NULL,
    packet_role varchar(40) NULL,
    status varchar(24) NOT NULL,
    parent_run_id varchar(120) NULL,
    config_hash char(64) NOT NULL,
    git_commit char(40) NOT NULL,
    started_at_utc datetime2(7) NOT NULL,
    finished_at_utc datetime2(7) NULL,
    stop_disposition varchar(80) NULL,
    notes nvarchar(2000) NULL,
    CONSTRAINT fk_control_runs_campaign FOREIGN KEY(campaign_id) REFERENCES control.campaigns(campaign_id),
    CONSTRAINT fk_control_runs_model FOREIGN KEY(model_profile_id) REFERENCES control.model_profiles(model_profile_id),
    CONSTRAINT fk_control_runs_arm FOREIGN KEY(agent_arm_id) REFERENCES control.agent_arms(agent_arm_id),
    CONSTRAINT fk_control_runs_decode FOREIGN KEY(decode_config_id) REFERENCES control.decode_configs(decode_config_id),
    CONSTRAINT fk_control_runs_parent FOREIGN KEY(parent_run_id) REFERENCES control.runs(run_id),
    CONSTRAINT ck_control_runs_status CHECK (status IN ('initialized','running','complete','stopped','invalid','failed'))
  );
END;

IF OBJECT_ID(N'control.jobs', N'U') IS NULL
BEGIN
  CREATE TABLE control.jobs
  (
    job_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_control_jobs PRIMARY KEY,
    job_key char(64) NOT NULL CONSTRAINT uq_control_jobs_key UNIQUE,
    campaign_id bigint NOT NULL,
    run_kind varchar(40) NOT NULL,
    episode_id varchar(120) NULL,
    model_profile_id varchar(80) NULL,
    agent_arm_id varchar(80) NULL,
    decode_config_id varchar(40) NULL,
    sample_index int NOT NULL CONSTRAINT df_control_jobs_sample DEFAULT(0),
    priority int NOT NULL CONSTRAINT df_control_jobs_priority DEFAULT(0),
    status varchar(24) NOT NULL,
    attempt_count int NOT NULL CONSTRAINT df_control_jobs_attempt DEFAULT(0),
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_control_jobs_created DEFAULT SYSUTCDATETIME(),
    started_at_utc datetime2(7) NULL,
    completed_at_utc datetime2(7) NULL,
    error_class varchar(100) NULL,
    error_detail nvarchar(max) NULL,
    CONSTRAINT fk_control_jobs_campaign FOREIGN KEY(campaign_id) REFERENCES control.campaigns(campaign_id),
    CONSTRAINT fk_control_jobs_model FOREIGN KEY(model_profile_id) REFERENCES control.model_profiles(model_profile_id),
    CONSTRAINT fk_control_jobs_arm FOREIGN KEY(agent_arm_id) REFERENCES control.agent_arms(agent_arm_id),
    CONSTRAINT fk_control_jobs_decode FOREIGN KEY(decode_config_id) REFERENCES control.decode_configs(decode_config_id),
    CONSTRAINT ck_control_jobs_status CHECK (status IN ('pending','running','complete','failed','stopped','STOP_BUDGET','STOP_PORT','STOP_DATA','STOP_SAFETY'))
  );
  CREATE INDEX ix_control_jobs_status ON control.jobs(status, priority DESC, job_id);
END;

IF OBJECT_ID(N'control.job_attempts', N'U') IS NULL
BEGIN
  CREATE TABLE control.job_attempts
  (
    job_attempt_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_control_job_attempts PRIMARY KEY,
    job_id bigint NOT NULL,
    attempt_number int NOT NULL,
    worker_id varchar(120) NOT NULL,
    request_manifest_json nvarchar(max) NOT NULL,
    started_at_utc datetime2(7) NOT NULL CONSTRAINT df_control_job_attempts_started DEFAULT SYSUTCDATETIME(),
    finished_at_utc datetime2(7) NULL,
    status varchar(24) NOT NULL,
    error_class varchar(100) NULL,
    error_detail nvarchar(max) NULL,
    CONSTRAINT uq_control_job_attempt UNIQUE(job_id, attempt_number),
    CONSTRAINT fk_control_job_attempts_job FOREIGN KEY(job_id) REFERENCES control.jobs(job_id),
    CONSTRAINT ck_control_job_attempts_json CHECK (ISJSON(request_manifest_json) = 1)
  );
END;

IF OBJECT_ID(N'control.evidence_events', N'U') IS NULL
BEGIN
  CREATE TABLE control.evidence_events
  (
    evidence_event_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_control_evidence_events PRIMARY KEY,
    event_key varchar(120) NOT NULL CONSTRAINT uq_control_evidence_events_key UNIQUE,
    run_id varchar(120) NULL,
    stage varchar(80) NOT NULL,
    scientific_tier varchar(24) NOT NULL,
    disposition varchar(40) NOT NULL,
    detail_json nvarchar(max) NOT NULL,
    detail_sha256 char(64) NOT NULL,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_control_evidence_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT ck_control_evidence_json CHECK (ISJSON(detail_json) = 1)
  );
END;

IF OBJECT_ID(N'control.run_artifacts', N'U') IS NULL
BEGIN
  CREATE TABLE control.run_artifacts
  (
    run_id varchar(120) NOT NULL,
    relative_path nvarchar(500) NOT NULL,
    artifact_kind varchar(80) NOT NULL,
    byte_count bigint NOT NULL,
    sha256 char(64) NOT NULL,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_control_artifacts_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_control_run_artifacts PRIMARY KEY(run_id, relative_path),
    CONSTRAINT ck_control_artifacts_bytes CHECK (byte_count >= 0)
  );
END;
