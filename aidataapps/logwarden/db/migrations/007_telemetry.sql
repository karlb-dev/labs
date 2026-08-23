SET XACT_ABORT ON;

IF OBJECT_ID(N'telemetry.journal_events', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.journal_events
  (
    journal_event_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_telemetry_journal_events PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    process_epoch_id uniqueidentifier NOT NULL,
    sequence bigint NOT NULL,
    event_id uniqueidentifier NOT NULL,
    semantic_stream varchar(40) NOT NULL,
    event_kind varchar(32) NOT NULL,
    event_name varchar(160) NOT NULL,
    job_id bigint NULL,
    episode_id varchar(120) NULL,
    attempt_id int NULL,
    trace_id char(32) NULL,
    span_id char(16) NULL,
    parent_span_id char(16) NULL,
    event_at_utc datetime2(7) NOT NULL,
    monotonic_ms decimal(20,3) NOT NULL,
    duration_ms decimal(18,3) NULL,
    status varchar(40) NULL,
    previous_record_sha256 char(64) NULL,
    record_sha256 char(64) NOT NULL,
    record_json nvarchar(max) NOT NULL,
    ingested_at_utc datetime2(7) NOT NULL CONSTRAINT df_telemetry_journal_ingested DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_telemetry_journal_sequence UNIQUE(run_id, process_epoch_id, sequence),
    CONSTRAINT uq_telemetry_journal_event_id UNIQUE(event_id),
    CONSTRAINT uq_telemetry_journal_hash UNIQUE(run_id, record_sha256),
    CONSTRAINT fk_telemetry_journal_job FOREIGN KEY(job_id) REFERENCES control.jobs(job_id),
    CONSTRAINT ck_telemetry_journal_stream CHECK (semantic_stream IN ('run','state_transition','transcript','model','tool','validation','decision','metric','system','error')),
    CONSTRAINT ck_telemetry_journal_kind CHECK (event_kind IN ('point','span_start','span_end','metric','raw_snapshot')),
    CONSTRAINT ck_telemetry_journal_sequence CHECK (sequence > 0),
    CONSTRAINT ck_telemetry_journal_json CHECK (ISJSON(record_json) = 1)
  );
  CREATE INDEX ix_telemetry_journal_run_time ON telemetry.journal_events(run_id, event_at_utc, journal_event_id);
  CREATE INDEX ix_telemetry_journal_correlation ON telemetry.journal_events(run_id, job_id, episode_id, trace_id, sequence);
END;

IF OBJECT_ID(N'telemetry.journal_cursors', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.journal_cursors
  (
    run_id varchar(120) NOT NULL,
    journal_path_sha256 char(64) NOT NULL,
    process_epoch_id uniqueidentifier NOT NULL,
    committed_sequence bigint NOT NULL,
    committed_record_sha256 char(64) NULL,
    journal_byte_offset bigint NOT NULL,
    updated_at_utc datetime2(7) NOT NULL CONSTRAINT df_telemetry_journal_cursor_updated DEFAULT SYSUTCDATETIME(),
    row_version rowversion NOT NULL,
    CONSTRAINT pk_telemetry_journal_cursors PRIMARY KEY(run_id, journal_path_sha256, process_epoch_id),
    CONSTRAINT ck_telemetry_journal_cursor_values CHECK (committed_sequence >= 0 AND journal_byte_offset >= 0)
  );
END;

IF OBJECT_ID(N'telemetry.traces', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.traces
  (
    trace_id char(32) NOT NULL CONSTRAINT pk_telemetry_traces PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    job_id bigint NULL,
    episode_id varchar(120) NULL,
    trace_kind varchar(40) NOT NULL,
    started_at_utc datetime2(7) NOT NULL,
    finished_at_utc datetime2(7) NULL,
    duration_ms decimal(18,3) NULL,
    status varchar(32) NOT NULL,
    root_attributes_json nvarchar(max) NOT NULL,
    CONSTRAINT fk_telemetry_traces_job FOREIGN KEY(job_id) REFERENCES control.jobs(job_id),
    CONSTRAINT ck_telemetry_traces_json CHECK (ISJSON(root_attributes_json) = 1)
  );
  CREATE INDEX ix_telemetry_traces_run ON telemetry.traces(run_id, job_id, trace_id);
END;

IF OBJECT_ID(N'telemetry.spans', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.spans
  (
    trace_id char(32) NOT NULL,
    span_id char(16) NOT NULL,
    parent_span_id char(16) NULL,
    span_name varchar(160) NOT NULL,
    span_kind varchar(40) NOT NULL,
    job_id bigint NULL,
    episode_id varchar(120) NULL,
    turn_id bigint NULL,
    model_request_id bigint NULL,
    tool_invocation_id bigint NULL,
    started_at_utc datetime2(7) NOT NULL,
    finished_at_utc datetime2(7) NULL,
    start_monotonic_ms decimal(20,3) NOT NULL,
    duration_ms decimal(18,3) NULL,
    status varchar(32) NOT NULL,
    attributes_json nvarchar(max) NOT NULL,
    error_class varchar(80) NULL,
    CONSTRAINT pk_telemetry_spans PRIMARY KEY(trace_id, span_id),
    CONSTRAINT fk_telemetry_spans_trace FOREIGN KEY(trace_id) REFERENCES telemetry.traces(trace_id),
    CONSTRAINT fk_telemetry_spans_job FOREIGN KEY(job_id) REFERENCES control.jobs(job_id),
    CONSTRAINT fk_telemetry_spans_turn FOREIGN KEY(turn_id) REFERENCES agent.turns(turn_id),
    CONSTRAINT fk_telemetry_spans_request FOREIGN KEY(model_request_id) REFERENCES agent.model_requests(model_request_id),
    CONSTRAINT fk_telemetry_spans_tool FOREIGN KEY(tool_invocation_id) REFERENCES agent.tool_invocations(tool_invocation_id),
    CONSTRAINT ck_telemetry_spans_json CHECK (ISJSON(attributes_json) = 1)
  );
  CREATE INDEX ix_telemetry_spans_waterfall ON telemetry.spans(trace_id, start_monotonic_ms, span_id);
END;

IF OBJECT_ID(N'telemetry.metric_samples', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.metric_samples
  (
    metric_sample_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_telemetry_metric_samples PRIMARY KEY,
    sample_key char(64) NOT NULL CONSTRAINT uq_telemetry_metric_samples_key UNIQUE,
    run_id varchar(120) NOT NULL,
    job_id bigint NULL,
    trace_id char(32) NULL,
    source_kind varchar(40) NOT NULL,
    source_instance varchar(160) NOT NULL,
    metric_name nvarchar(300) NOT NULL,
    metric_type varchar(24) NOT NULL,
    metric_value float NULL,
    unit varchar(40) NULL,
    provenance varchar(32) NOT NULL,
    labels_json nvarchar(max) NOT NULL,
    observed_at_utc datetime2(7) NOT NULL,
    interval_start_utc datetime2(7) NULL,
    interval_end_utc datetime2(7) NULL,
    unavailable_reason nvarchar(500) NULL,
    CONSTRAINT fk_telemetry_metric_job FOREIGN KEY(job_id) REFERENCES control.jobs(job_id),
    CONSTRAINT ck_telemetry_metric_type CHECK (metric_type IN ('counter','gauge','histogram_count','histogram_sum','histogram_bucket','derived')),
    CONSTRAINT ck_telemetry_metric_provenance CHECK (provenance IN ('raw_service','client_observed','sql_dmv','nvidia_smi','derived','estimated')),
    CONSTRAINT ck_telemetry_metric_json CHECK (ISJSON(labels_json) = 1),
    CONSTRAINT ck_telemetry_metric_value CHECK (metric_value IS NOT NULL OR unavailable_reason IS NOT NULL)
  );
  CREATE INDEX ix_telemetry_metric_series ON telemetry.metric_samples(run_id, source_kind, metric_name, observed_at_utc, metric_sample_id);
END;

IF OBJECT_ID(N'telemetry.raw_metric_snapshots', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.raw_metric_snapshots
  (
    raw_metric_snapshot_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_telemetry_raw_metric_snapshots PRIMARY KEY,
    snapshot_key char(64) NOT NULL CONSTRAINT uq_telemetry_raw_metric_snapshot_key UNIQUE,
    run_id varchar(120) NOT NULL,
    source_kind varchar(40) NOT NULL,
    source_instance varchar(160) NOT NULL,
    endpoint_uri nvarchar(1000) NULL,
    phase varchar(40) NOT NULL,
    http_status int NULL,
    content_type varchar(160) NULL,
    payload varbinary(max) NULL,
    payload_sha256 char(64) NULL,
    payload_bytes bigint NULL,
    discovered_metric_names_json nvarchar(max) NOT NULL,
    observed_at_utc datetime2(7) NOT NULL,
    latency_ms decimal(18,3) NULL,
    status varchar(24) NOT NULL,
    error_detail nvarchar(max) NULL,
    CONSTRAINT ck_telemetry_raw_snapshot_names CHECK (ISJSON(discovered_metric_names_json) = 1),
    CONSTRAINT ck_telemetry_raw_snapshot_payload CHECK ((payload IS NULL AND payload_sha256 IS NULL) OR (payload IS NOT NULL AND payload_sha256 IS NOT NULL))
  );
END;

IF OBJECT_ID(N'telemetry.queue_samples', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.queue_samples
  (
    queue_sample_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_telemetry_queue_samples PRIMARY KEY,
    sample_key char(64) NOT NULL CONSTRAINT uq_telemetry_queue_samples_key UNIQUE,
    run_id varchar(120) NOT NULL,
    sampled_at_utc datetime2(7) NOT NULL,
    pending_count int NOT NULL,
    leased_count int NOT NULL,
    retryable_count int NOT NULL,
    terminal_count int NOT NULL,
    oldest_pending_age_ms bigint NULL,
    lease_expired_count int NOT NULL,
    arrivals_since_prior int NULL,
    completions_since_prior int NULL,
    worker_count int NOT NULL,
    sampler_epoch_id uniqueidentifier NOT NULL,
    CONSTRAINT ck_telemetry_queue_counts CHECK (pending_count >= 0 AND leased_count >= 0 AND retryable_count >= 0 AND terminal_count >= 0 AND lease_expired_count >= 0 AND worker_count >= 0)
  );
  CREATE INDEX ix_telemetry_queue_run_time ON telemetry.queue_samples(run_id, sampled_at_utc, queue_sample_id);
END;

IF OBJECT_ID(N'telemetry.model_service_samples', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.model_service_samples
  (
    model_service_sample_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_telemetry_model_service_samples PRIMARY KEY,
    sample_key char(64) NOT NULL CONSTRAINT uq_telemetry_model_samples_key UNIQUE,
    run_id varchar(120) NOT NULL,
    model_profile_id varchar(80) NULL,
    sampled_at_utc datetime2(7) NOT NULL,
    running_requests float NULL,
    waiting_requests float NULL,
    swapped_requests float NULL,
    prompt_tokens_total float NULL,
    generation_tokens_total float NULL,
    prompt_throughput float NULL,
    generation_throughput float NULL,
    kv_cache_usage_ratio float NULL,
    prefix_cache_hits_total float NULL,
    prefix_cache_queries_total float NULL,
    preemptions_total float NULL,
    request_errors_total float NULL,
    cancellations_total float NULL,
    raw_metric_snapshot_id bigint NOT NULL,
    availability_json nvarchar(max) NOT NULL,
    CONSTRAINT fk_telemetry_model_profile FOREIGN KEY(model_profile_id) REFERENCES control.model_profiles(model_profile_id),
    CONSTRAINT fk_telemetry_model_raw FOREIGN KEY(raw_metric_snapshot_id) REFERENCES telemetry.raw_metric_snapshots(raw_metric_snapshot_id),
    CONSTRAINT ck_telemetry_model_availability_json CHECK (ISJSON(availability_json) = 1)
  );
END;

IF OBJECT_ID(N'telemetry.gpu_samples', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.gpu_samples
  (
    gpu_sample_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_telemetry_gpu_samples PRIMARY KEY,
    sample_key char(64) NOT NULL CONSTRAINT uq_telemetry_gpu_samples_key UNIQUE,
    run_id varchar(120) NOT NULL,
    gpu_uuid varchar(120) NOT NULL,
    sampled_at_utc datetime2(7) NOT NULL,
    utilization_gpu_pct float NULL,
    utilization_memory_pct float NULL,
    memory_used_mib float NULL,
    memory_total_mib float NULL,
    power_draw_w float NULL,
    power_limit_w float NULL,
    temperature_c float NULL,
    sm_clock_mhz float NULL,
    memory_clock_mhz float NULL,
    pstate varchar(16) NULL,
    process_json nvarchar(max) NOT NULL,
    unavailable_reason nvarchar(500) NULL,
    CONSTRAINT ck_telemetry_gpu_process_json CHECK (ISJSON(process_json) = 1)
  );
  CREATE INDEX ix_telemetry_gpu_run_time ON telemetry.gpu_samples(run_id, gpu_uuid, sampled_at_utc);
END;

IF OBJECT_ID(N'telemetry.sql_resource_samples', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.sql_resource_samples
  (
    sql_resource_sample_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_telemetry_sql_resource_samples PRIMARY KEY,
    sample_key char(64) NOT NULL CONSTRAINT uq_telemetry_sql_resource_samples_key UNIQUE,
    run_id varchar(120) NOT NULL,
    sampled_at_utc datetime2(7) NOT NULL,
    process_cpu_pct float NULL,
    process_memory_kb bigint NULL,
    target_memory_kb bigint NULL,
    request_count int NULL,
    blocked_request_count int NULL,
    runnable_task_count int NULL,
    pending_io_count int NULL,
    data_file_bytes bigint NULL,
    log_file_bytes bigint NULL,
    log_used_pct float NULL,
    waits_json nvarchar(max) NOT NULL,
    io_json nvarchar(max) NOT NULL,
    unavailable_reason nvarchar(500) NULL,
    CONSTRAINT ck_telemetry_sql_waits_json CHECK (ISJSON(waits_json) = 1),
    CONSTRAINT ck_telemetry_sql_io_json CHECK (ISJSON(io_json) = 1)
  );
END;

IF OBJECT_ID(N'telemetry.query_store_intervals', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.query_store_intervals
  (
    query_store_interval_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_telemetry_query_store_intervals PRIMARY KEY,
    sample_key char(64) NOT NULL CONSTRAINT uq_telemetry_query_store_intervals_key UNIQUE,
    run_id varchar(120) NOT NULL,
    database_name sysname NOT NULL,
    interval_id bigint NOT NULL,
    interval_start_utc datetime2(7) NOT NULL,
    interval_end_utc datetime2(7) NOT NULL,
    execution_count bigint NOT NULL,
    duration_ms float NOT NULL,
    cpu_ms float NOT NULL,
    logical_reads float NOT NULL,
    physical_reads float NOT NULL,
    log_bytes float NOT NULL,
    wait_stats_json nvarchar(max) NOT NULL,
    captured_at_utc datetime2(7) NOT NULL CONSTRAINT df_telemetry_qs_captured DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_telemetry_qs_interval UNIQUE(run_id, database_name, interval_id),
    CONSTRAINT ck_telemetry_qs_waits_json CHECK (ISJSON(wait_stats_json) = 1)
  );
END;

IF OBJECT_ID(N'telemetry.xe_pipeline_samples', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.xe_pipeline_samples
  (
    xe_pipeline_sample_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_telemetry_xe_pipeline_samples PRIMARY KEY,
    sample_key char(64) NOT NULL CONSTRAINT uq_telemetry_xe_pipeline_samples_key UNIQUE,
    run_id varchar(120) NOT NULL,
    source_id varchar(80) NOT NULL,
    sampled_at_utc datetime2(7) NOT NULL,
    cursor_file_name nvarchar(500) NULL,
    cursor_file_offset bigint NULL,
    newest_event_at_utc datetime2(7) NULL,
    ingest_lag_ms bigint NULL,
    batch_rows int NULL,
    parse_errors_total bigint NULL,
    rollover_files int NULL,
    dropped_events_total bigint NULL,
    source_bytes bigint NULL,
    CONSTRAINT fk_telemetry_xe_source FOREIGN KEY(source_id) REFERENCES ingest.sources(source_id)
  );
END;

IF OBJECT_ID(N'telemetry.host_samples', N'U') IS NULL
BEGIN
  CREATE TABLE telemetry.host_samples
  (
    host_sample_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_telemetry_host_samples PRIMARY KEY,
    sample_key char(64) NOT NULL CONSTRAINT uq_telemetry_host_samples_key UNIQUE,
    run_id varchar(120) NOT NULL,
    sampled_at_utc datetime2(7) NOT NULL,
    load_1m float NULL,
    memory_used_bytes bigint NULL,
    disk_free_bytes bigint NULL,
    disk_read_bytes_total bigint NULL,
    disk_write_bytes_total bigint NULL,
    journal_bytes bigint NULL,
    database_bytes bigint NULL,
    mirror_bytes bigint NULL,
    attributes_json nvarchar(max) NOT NULL,
    CONSTRAINT ck_telemetry_host_attributes_json CHECK (ISJSON(attributes_json) = 1)
  );
END;
