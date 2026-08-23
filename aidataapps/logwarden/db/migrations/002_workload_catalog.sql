SET XACT_ABORT ON;

IF OBJECT_ID(N'workload.scenario_definitions', N'U') IS NULL
BEGIN
  CREATE TABLE workload.scenario_definitions
  (
    scenario_id varchar(100) NOT NULL CONSTRAINT pk_workload_scenario_definitions PRIMARY KEY,
    scenario_group_id varchar(100) NOT NULL,
    family varchar(80) NOT NULL,
    regime char(1) NOT NULL,
    scenario_version int NOT NULL,
    description nvarchar(1000) NOT NULL,
    injector_procedure sysname NULL,
    driver_id varchar(80) NULL,
    max_runtime_seconds int NOT NULL,
    safety_class varchar(40) NOT NULL,
    cleanup_procedure sysname NOT NULL,
    expected_class varchar(80) NOT NULL,
    expected_severity varchar(24) NOT NULL,
    should_abstain bit NOT NULL,
    is_multi_event bit NOT NULL,
    is_context_dependent bit NOT NULL,
    config_json nvarchar(max) NOT NULL,
    ground_truth_json nvarchar(max) NOT NULL,
    scenario_sha256 char(64) NOT NULL CONSTRAINT uq_workload_scenarios_hash UNIQUE,
    enabled bit NOT NULL CONSTRAINT df_workload_scenarios_enabled DEFAULT(1),
    CONSTRAINT ck_workload_scenarios_regime CHECK (regime IN ('K','C','U','M','N')),
    CONSTRAINT ck_workload_scenarios_runtime CHECK (max_runtime_seconds BETWEEN 1 AND 3600),
    CONSTRAINT ck_workload_scenarios_config_json CHECK (ISJSON(config_json) = 1),
    CONSTRAINT ck_workload_scenarios_truth_json CHECK (ISJSON(ground_truth_json) = 1)
  );
  CREATE INDEX ix_workload_scenarios_group ON workload.scenario_definitions(scenario_group_id, family, regime);
END;

IF OBJECT_ID(N'workload.scenario_variants', N'U') IS NULL
BEGIN
  CREATE TABLE workload.scenario_variants
  (
    scenario_variant_id varchar(120) NOT NULL CONSTRAINT pk_workload_scenario_variants PRIMARY KEY,
    scenario_id varchar(100) NOT NULL,
    variant_group_id varchar(120) NOT NULL,
    split_role varchar(40) NOT NULL,
    parameter_json nvarchar(max) NOT NULL,
    message_view_policy varchar(40) NOT NULL,
    rate_context_id varchar(80) NULL,
    variant_sha256 char(64) NOT NULL CONSTRAINT uq_workload_variants_hash UNIQUE,
    CONSTRAINT fk_workload_variants_scenario FOREIGN KEY(scenario_id) REFERENCES workload.scenario_definitions(scenario_id),
    CONSTRAINT ck_workload_variants_role CHECK (split_role IN ('dev','calibration','test_id','test_variant_holdout','test_unknown','test_live_parity','test_storm')),
    CONSTRAINT ck_workload_variants_json CHECK (ISJSON(parameter_json) = 1)
  );
  CREATE INDEX ix_workload_variants_split ON workload.scenario_variants(split_role, scenario_id);
END;

IF OBJECT_ID(N'workload.schedules', N'U') IS NULL
BEGIN
  CREATE TABLE workload.schedules
  (
    schedule_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_workload_schedules PRIMARY KEY,
    campaign_id bigint NOT NULL,
    schedule_name varchar(80) NOT NULL,
    seed bigint NOT NULL,
    rate_profile varchar(40) NOT NULL,
    schedule_hash char(64) NOT NULL CONSTRAINT uq_workload_schedules_hash UNIQUE,
    status varchar(24) NOT NULL,
    config_json nvarchar(max) NOT NULL,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_workload_schedules_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_workload_schedules_campaign FOREIGN KEY(campaign_id) REFERENCES control.campaigns(campaign_id),
    CONSTRAINT ck_workload_schedules_status CHECK (status IN ('building','frozen','running','complete','invalid')),
    CONSTRAINT ck_workload_schedules_json CHECK (ISJSON(config_json) = 1)
  );
END;

IF OBJECT_ID(N'workload.schedule_items', N'U') IS NULL
BEGIN
  CREATE TABLE workload.schedule_items
  (
    schedule_item_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_workload_schedule_items PRIMARY KEY,
    schedule_id bigint NOT NULL,
    job_key char(64) NOT NULL CONSTRAINT uq_workload_schedule_items_key UNIQUE,
    scenario_variant_id varchar(120) NOT NULL,
    ordinal int NOT NULL,
    planned_offset_ms bigint NOT NULL,
    episode_seed bigint NOT NULL,
    expected_role varchar(40) NOT NULL,
    status varchar(24) NOT NULL CONSTRAINT df_workload_schedule_items_status DEFAULT('pending'),
    CONSTRAINT uq_workload_schedule_ordinal UNIQUE(schedule_id, ordinal),
    CONSTRAINT fk_workload_schedule_items_schedule FOREIGN KEY(schedule_id) REFERENCES workload.schedules(schedule_id),
    CONSTRAINT fk_workload_schedule_items_variant FOREIGN KEY(scenario_variant_id) REFERENCES workload.scenario_variants(scenario_variant_id),
    CONSTRAINT ck_workload_schedule_items_offset CHECK (planned_offset_ms >= 0)
  );
END;

IF OBJECT_ID(N'workload.disposable_databases', N'U') IS NULL
BEGIN
  CREATE TABLE workload.disposable_databases
  (
    disposable_database_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_workload_disposable_databases PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    episode_id varchar(120) NOT NULL,
    database_name sysname NOT NULL CONSTRAINT uq_workload_disposable_database_name UNIQUE,
    creation_token uniqueidentifier NOT NULL,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_workload_disposable_created DEFAULT SYSUTCDATETIME(),
    dropped_at_utc datetime2(7) NULL,
    cleanup_status varchar(24) NOT NULL,
    CONSTRAINT uq_workload_disposable_token UNIQUE(database_name, creation_token),
    CONSTRAINT ck_workload_disposable_name CHECK (database_name LIKE 'LW[_]%')
  );
END;

IF OBJECT_ID(N'workload.injection_executions', N'U') IS NULL
BEGIN
  CREATE TABLE workload.injection_executions
  (
    injection_execution_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_workload_injection_executions PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    schedule_item_id bigint NOT NULL,
    episode_id varchar(120) NOT NULL,
    disposable_database_id bigint NULL,
    injector_request_json nvarchar(max) NOT NULL,
    started_at_utc datetime2(7) NOT NULL,
    finished_at_utc datetime2(7) NULL,
    sql_session_ids_json nvarchar(max) NOT NULL,
    return_code int NULL,
    verified bit NOT NULL CONSTRAINT df_workload_injection_verified DEFAULT(0),
    cleanup_verified bit NOT NULL CONSTRAINT df_workload_injection_cleanup DEFAULT(0),
    error_detail nvarchar(max) NULL,
    CONSTRAINT uq_workload_injection_run_item UNIQUE(run_id, schedule_item_id),
    CONSTRAINT fk_workload_injection_item FOREIGN KEY(schedule_item_id) REFERENCES workload.schedule_items(schedule_item_id),
    CONSTRAINT fk_workload_injection_database FOREIGN KEY(disposable_database_id) REFERENCES workload.disposable_databases(disposable_database_id),
    CONSTRAINT ck_workload_injection_request_json CHECK (ISJSON(injector_request_json) = 1),
    CONSTRAINT ck_workload_injection_sessions_json CHECK (ISJSON(sql_session_ids_json) = 1)
  );
  CREATE INDEX ix_workload_injection_episode ON workload.injection_executions(episode_id);
END;

IF OBJECT_ID(N'workload.expected_evidence', N'U') IS NULL
BEGIN
  CREATE TABLE workload.expected_evidence
  (
    expected_evidence_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_workload_expected_evidence PRIMARY KEY,
    scenario_variant_id varchar(120) NOT NULL,
    source_kind varchar(40) NOT NULL,
    event_name varchar(120) NULL,
    error_number int NULL,
    minimum_count int NOT NULL,
    maximum_count int NULL,
    match_rule_json nvarchar(max) NOT NULL,
    required bit NOT NULL,
    CONSTRAINT fk_workload_expected_variant FOREIGN KEY(scenario_variant_id) REFERENCES workload.scenario_variants(scenario_variant_id),
    CONSTRAINT ck_workload_expected_counts CHECK (minimum_count >= 0 AND (maximum_count IS NULL OR maximum_count >= minimum_count)),
    CONSTRAINT ck_workload_expected_json CHECK (ISJSON(match_rule_json) = 1)
  );
END;

IF OBJECT_ID(N'workload.backup_attempts', N'U') IS NULL
BEGIN
  CREATE TABLE workload.backup_attempts
  (
    backup_attempt_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_workload_backup_attempts PRIMARY KEY,
    episode_id varchar(120) NOT NULL,
    database_name sysname NOT NULL,
    destination_hash char(64) NOT NULL,
    started_at_utc datetime2(7) NOT NULL,
    finished_at_utc datetime2(7) NULL,
    succeeded bit NOT NULL,
    error_number int NULL,
    detail_json nvarchar(max) NOT NULL,
    CONSTRAINT ck_workload_backup_attempts_json CHECK (ISJSON(detail_json) = 1)
  );
END;
