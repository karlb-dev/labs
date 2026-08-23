SET XACT_ABORT ON;

IF OBJECT_ID(N'ingest.sources', N'U') IS NULL
BEGIN
  CREATE TABLE ingest.sources
  (
    source_id varchar(80) NOT NULL CONSTRAINT pk_ingest_sources PRIMARY KEY,
    source_kind varchar(40) NOT NULL,
    source_name varchar(120) NOT NULL,
    definition_sha256 char(64) NOT NULL,
    config_json nvarchar(max) NOT NULL,
    CONSTRAINT ck_ingest_sources_json CHECK (ISJSON(config_json) = 1)
  );
END;

IF OBJECT_ID(N'ingest.source_cursors', N'U') IS NULL
BEGIN
  CREATE TABLE ingest.source_cursors
  (
    source_id varchar(80) NOT NULL CONSTRAINT pk_ingest_source_cursors PRIMARY KEY,
    file_name nvarchar(500) NULL,
    file_offset bigint NULL,
    log_generation int NULL,
    watermark_utc datetime2(7) NULL,
    cursor_json nvarchar(max) NOT NULL,
    updated_at_utc datetime2(7) NOT NULL CONSTRAINT df_ingest_cursor_updated DEFAULT SYSUTCDATETIME(),
    row_version rowversion NOT NULL,
    CONSTRAINT fk_ingest_cursor_source FOREIGN KEY(source_id) REFERENCES ingest.sources(source_id),
    CONSTRAINT ck_ingest_cursor_json CHECK (ISJSON(cursor_json) = 1)
  );
END;

IF OBJECT_ID(N'ingest.ingestion_batches', N'U') IS NULL
BEGIN
  CREATE TABLE ingest.ingestion_batches
  (
    ingestion_batch_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_ingest_batches PRIMARY KEY,
    source_id varchar(80) NOT NULL,
    worker_id varchar(120) NOT NULL,
    start_cursor_json nvarchar(max) NOT NULL,
    end_cursor_json nvarchar(max) NULL,
    rows_read int NOT NULL CONSTRAINT df_ingest_batches_read DEFAULT(0),
    rows_inserted int NOT NULL CONSTRAINT df_ingest_batches_inserted DEFAULT(0),
    rows_duplicate int NOT NULL CONSTRAINT df_ingest_batches_duplicate DEFAULT(0),
    rows_failed int NOT NULL CONSTRAINT df_ingest_batches_failed DEFAULT(0),
    started_at_utc datetime2(7) NOT NULL CONSTRAINT df_ingest_batches_started DEFAULT SYSUTCDATETIME(),
    committed_at_utc datetime2(7) NULL,
    status varchar(24) NOT NULL,
    CONSTRAINT fk_ingest_batches_source FOREIGN KEY(source_id) REFERENCES ingest.sources(source_id),
    CONSTRAINT ck_ingest_batches_start_json CHECK (ISJSON(start_cursor_json) = 1),
    CONSTRAINT ck_ingest_batches_end_json CHECK (end_cursor_json IS NULL OR ISJSON(end_cursor_json) = 1)
  );
END;

IF OBJECT_ID(N'ingest.raw_events', N'U') IS NULL
BEGIN
  CREATE TABLE ingest.raw_events
  (
    raw_event_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_ingest_raw_events PRIMARY KEY,
    source_id varchar(80) NOT NULL,
    ingestion_batch_id bigint NOT NULL,
    source_position_key char(64) NOT NULL CONSTRAINT uq_ingest_raw_position UNIQUE,
    source_file_name nvarchar(500) NULL,
    source_file_offset bigint NULL,
    source_timestamp_utc datetime2(7) NULL,
    captured_at_utc datetime2(7) NOT NULL,
    raw_event_name varchar(120) NULL,
    raw_payload_xml xml NULL,
    raw_payload_text nvarchar(max) NULL,
    raw_payload_json nvarchar(max) NULL,
    raw_sha256 char(64) NOT NULL,
    parse_status varchar(24) NOT NULL,
    CONSTRAINT fk_ingest_raw_source FOREIGN KEY(source_id) REFERENCES ingest.sources(source_id),
    CONSTRAINT fk_ingest_raw_batch FOREIGN KEY(ingestion_batch_id) REFERENCES ingest.ingestion_batches(ingestion_batch_id),
    CONSTRAINT ck_ingest_raw_payload CHECK (raw_payload_xml IS NOT NULL OR raw_payload_text IS NOT NULL OR raw_payload_json IS NOT NULL),
    CONSTRAINT ck_ingest_raw_json CHECK (raw_payload_json IS NULL OR ISJSON(raw_payload_json) = 1)
  );
  CREATE INDEX ix_ingest_raw_source_time ON ingest.raw_events(source_id, source_timestamp_utc, raw_event_id);
END;

IF OBJECT_ID(N'ingest.canonical_events', N'U') IS NULL
BEGIN
  CREATE TABLE ingest.canonical_events
  (
    canonical_event_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_ingest_canonical_events PRIMARY KEY,
    raw_event_id bigint NOT NULL CONSTRAINT uq_ingest_canonical_raw UNIQUE,
    source_kind varchar(40) NOT NULL,
    source_event_name varchar(120) NOT NULL,
    occurred_at_utc datetime2(7) NOT NULL,
    captured_at_utc datetime2(7) NOT NULL,
    database_name sysname NULL,
    session_id int NULL,
    error_number int NULL,
    severity int NULL,
    state int NULL,
    message_raw nvarchar(max) NULL,
    message_normalized nvarchar(max) NULL,
    sql_text_sha256 char(64) NULL,
    object_name nvarchar(512) NULL,
    client_app_name nvarchar(256) NULL,
    login_name nvarchar(256) NULL,
    duration_ms decimal(18,3) NULL,
    cpu_ms decimal(18,3) NULL,
    logical_reads bigint NULL,
    writes bigint NULL,
    wait_type varchar(120) NULL,
    activity_id varchar(100) NULL,
    parser_version varchar(40) NOT NULL,
    normalization_version varchar(40) NOT NULL,
    canonical_json nvarchar(max) NOT NULL,
    normalized_sha256 char(64) NOT NULL,
    CONSTRAINT fk_ingest_canonical_raw FOREIGN KEY(raw_event_id) REFERENCES ingest.raw_events(raw_event_id),
    CONSTRAINT ck_ingest_canonical_json CHECK (ISJSON(canonical_json) = 1)
  );
  CREATE INDEX ix_ingest_canonical_time ON ingest.canonical_events(occurred_at_utc, canonical_event_id);
  CREATE INDEX ix_ingest_canonical_signature ON ingest.canonical_events(error_number, source_event_name, occurred_at_utc);
END;

IF OBJECT_ID(N'ingest.event_fingerprints', N'U') IS NULL
BEGIN
  CREATE TABLE ingest.event_fingerprints
  (
    canonical_event_id bigint NOT NULL CONSTRAINT pk_ingest_event_fingerprints PRIMARY KEY,
    fingerprint_version varchar(40) NOT NULL,
    event_fingerprint char(64) NOT NULL,
    message_template_hash char(64) NOT NULL,
    identifier_masked_text nvarchar(max) NOT NULL,
    error_number_masked_text nvarchar(max) NOT NULL,
    CONSTRAINT fk_ingest_fingerprint_event FOREIGN KEY(canonical_event_id) REFERENCES ingest.canonical_events(canonical_event_id)
  );
  CREATE INDEX ix_ingest_fingerprint_hash ON ingest.event_fingerprints(event_fingerprint, canonical_event_id);
END;

IF OBJECT_ID(N'ingest.injection_event_links', N'U') IS NULL
BEGIN
  CREATE TABLE ingest.injection_event_links
  (
    injection_execution_id bigint NOT NULL,
    canonical_event_id bigint NOT NULL,
    match_role varchar(40) NOT NULL,
    match_score decimal(9,6) NOT NULL,
    match_rule_id varchar(80) NOT NULL,
    audited bit NOT NULL CONSTRAINT df_ingest_links_audited DEFAULT(0),
    CONSTRAINT pk_ingest_injection_event_links PRIMARY KEY(injection_execution_id, canonical_event_id),
    CONSTRAINT fk_ingest_links_injection FOREIGN KEY(injection_execution_id) REFERENCES workload.injection_executions(injection_execution_id),
    CONSTRAINT fk_ingest_links_event FOREIGN KEY(canonical_event_id) REFERENCES ingest.canonical_events(canonical_event_id),
    CONSTRAINT ck_ingest_links_role CHECK (match_role IN ('anchor','supporting','noise','conflicting')),
    CONSTRAINT ck_ingest_links_score CHECK (match_score BETWEEN 0 AND 1)
  );
END;

IF OBJECT_ID(N'ingest.context_snapshots', N'U') IS NULL
BEGIN
  CREATE TABLE ingest.context_snapshots
  (
    context_snapshot_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_ingest_context_snapshots PRIMARY KEY,
    episode_id varchar(120) NOT NULL,
    snapshot_kind varchar(80) NOT NULL,
    tool_id varchar(80) NOT NULL,
    canonical_args_sha256 char(64) NOT NULL,
    canonical_args_json nvarchar(max) NOT NULL,
    procedure_version varchar(80) NOT NULL,
    requested_at_utc datetime2(7) NOT NULL,
    captured_at_utc datetime2(7) NOT NULL,
    result_json nvarchar(max) NOT NULL,
    result_sha256 char(64) NOT NULL,
    row_count int NOT NULL,
    latency_ms decimal(18,3) NOT NULL,
    status varchar(24) NOT NULL,
    CONSTRAINT uq_ingest_snapshot_lookup UNIQUE(episode_id, tool_id, canonical_args_sha256),
    CONSTRAINT ck_ingest_snapshot_args_json CHECK (ISJSON(canonical_args_json) = 1),
    CONSTRAINT ck_ingest_snapshot_result_json CHECK (ISJSON(result_json) = 1)
  );
END;

IF OBJECT_ID(N'ingest.incident_packets', N'U') IS NULL
BEGIN
  CREATE TABLE ingest.incident_packets
  (
    episode_id varchar(120) NOT NULL CONSTRAINT pk_ingest_incident_packets PRIMARY KEY,
    campaign_id bigint NOT NULL,
    scenario_variant_id varchar(120) NOT NULL,
    split_role varchar(40) NOT NULL,
    packet_version varchar(40) NOT NULL,
    packet_json nvarchar(max) NOT NULL,
    packet_sha256 char(64) NOT NULL CONSTRAINT uq_ingest_packets_hash UNIQUE,
    source_manifest_json nvarchar(max) NOT NULL,
    frozen_tool_manifest_json nvarchar(max) NOT NULL,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_ingest_packets_created DEFAULT SYSUTCDATETIME(),
    frozen_at_utc datetime2(7) NULL,
    is_valid bit NOT NULL,
    exclusion_reason nvarchar(1000) NULL,
    CONSTRAINT fk_ingest_packets_campaign FOREIGN KEY(campaign_id) REFERENCES control.campaigns(campaign_id),
    CONSTRAINT fk_ingest_packets_variant FOREIGN KEY(scenario_variant_id) REFERENCES workload.scenario_variants(scenario_variant_id),
    CONSTRAINT ck_ingest_packets_json CHECK (ISJSON(packet_json) = 1),
    CONSTRAINT ck_ingest_packets_source_json CHECK (ISJSON(source_manifest_json) = 1),
    CONSTRAINT ck_ingest_packets_tools_json CHECK (ISJSON(frozen_tool_manifest_json) = 1)
  );
END;

IF OBJECT_ID(N'ingest.packet_events', N'U') IS NULL
BEGIN
  CREATE TABLE ingest.packet_events
  (
    episode_id varchar(120) NOT NULL,
    canonical_event_id bigint NOT NULL,
    ordinal int NOT NULL,
    packet_role varchar(40) NOT NULL,
    CONSTRAINT pk_ingest_packet_events PRIMARY KEY(episode_id, canonical_event_id),
    CONSTRAINT uq_ingest_packet_ordinal UNIQUE(episode_id, ordinal),
    CONSTRAINT fk_ingest_packet_events_packet FOREIGN KEY(episode_id) REFERENCES ingest.incident_packets(episode_id),
    CONSTRAINT fk_ingest_packet_events_event FOREIGN KEY(canonical_event_id) REFERENCES ingest.canonical_events(canonical_event_id)
  );
END;
