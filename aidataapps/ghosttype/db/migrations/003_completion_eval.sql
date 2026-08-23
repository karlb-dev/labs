-- Completion lifecycle, raw-first persistence, evaluation, telemetry.
IF OBJECT_ID(N'completion.requests', N'U') IS NULL
  CREATE TABLE completion.requests
  (
    request_id varchar(120) NOT NULL CONSTRAINT pk_completion_requests PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    campaign varchar(80) NOT NULL,
    case_id varchar(80) NOT NULL,
    profile_id varchar(80) NOT NULL,
    arm varchar(40) NOT NULL,
    decode_config_json nvarchar(max) NOT NULL,
    prompt_sha256 char(64) NOT NULL,
    status varchar(20) NOT NULL CONSTRAINT df_request_status DEFAULT 'pending',
    created_at_utc datetime2(3) NOT NULL CONSTRAINT df_request_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_request_job UNIQUE (run_id, case_id, profile_id, arm)
  );
CREATE INDEX ix_requests_status ON completion.requests (run_id, status) WITH (DATA_COMPRESSION = NONE);

IF OBJECT_ID(N'completion.raw_model_responses', N'U') IS NULL
  CREATE TABLE completion.raw_model_responses
  (
    request_id varchar(120) NOT NULL CONSTRAINT pk_raw_responses PRIMARY KEY,
    http_status int NOT NULL,
    raw_body nvarchar(max) NOT NULL,
    raw_sha256 char(64) NOT NULL,
    latency_ms float NOT NULL,
    prompt_tokens int NULL,
    completion_tokens int NULL,
    reasoning_chars int NULL,
    received_at_utc datetime2(3) NOT NULL CONSTRAINT df_raw_received DEFAULT SYSUTCDATETIME()
  );

IF OBJECT_ID(N'completion.candidates', N'U') IS NULL
  CREATE TABLE completion.candidates
  (
    candidate_id varchar(140) NOT NULL CONSTRAINT pk_candidates PRIMARY KEY,
    request_id varchar(120) NOT NULL,
    ordinal int NOT NULL,
    raw_text nvarchar(max) NOT NULL,
    extracted_text nvarchar(max) NOT NULL,
    extraction_version varchar(40) NOT NULL,
    suffix_overlap_trimmed int NOT NULL CONSTRAINT df_cand_trim DEFAULT 0,
    format_failure varchar(40) NULL,       -- markdown_fence|prose|mode_leak|null
    is_abstention bit NOT NULL CONSTRAINT df_cand_abst DEFAULT 0
  );

IF OBJECT_ID(N'completion.final_decisions', N'U') IS NULL
  CREATE TABLE completion.final_decisions
  (
    request_id varchar(120) NOT NULL CONSTRAINT pk_final_decisions PRIMARY KEY,
    decision varchar(20) NOT NULL,          -- offer|abstain|fail
    selected_candidate_id varchar(140) NULL,
    decided_at_utc datetime2(3) NOT NULL CONSTRAINT df_decision_at DEFAULT SYSUTCDATETIME()
  );

IF OBJECT_ID(N'eval.row_scores', N'U') IS NULL
  CREATE TABLE eval.row_scores
  (
    run_id varchar(120) NOT NULL,
    case_id varchar(80) NOT NULL,
    profile_id varchar(80) NOT NULL,
    arm varchar(40) NOT NULL,
    eligible bit NOT NULL,
    outcome varchar(30) NOT NULL,            -- success|partial|fail|abstain_correct|abstain_wrong|format_fail|missing
    failure_stage varchar(30) NULL,
    normalized_exact bit NULL,
    constraint_pass bit NULL,
    grounding_pass bit NULL,
    empty_policy_pass bit NULL,
    no_markdown_pass bit NULL,
    insertion_integrity_pass bit NULL,
    parse_pass bit NULL,
    semantic_success bit NULL,
    latency_ms float NULL,
    completion_tokens int NULL,
    detail_json nvarchar(max) NULL,
    CONSTRAINT pk_row_scores PRIMARY KEY (run_id, case_id, profile_id, arm)
  );

IF OBJECT_ID(N'eval.metric_results', N'U') IS NULL
  CREATE TABLE eval.metric_results
  (
    metric_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_metric_results PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    profile_id varchar(80) NOT NULL,
    arm varchar(40) NOT NULL,
    suite varchar(60) NOT NULL,
    metric_name varchar(80) NOT NULL,
    metric_value float NULL,
    rows_count int NOT NULL,
    detail_json nvarchar(max) NULL,
    created_at_utc datetime2(3) NOT NULL CONSTRAINT df_metric_created DEFAULT SYSUTCDATETIME()
  );

IF OBJECT_ID(N'eval.claims', N'U') IS NULL
  CREATE TABLE eval.claims
  (
    claim_id int IDENTITY(1,1) NOT NULL CONSTRAINT pk_claims PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    research_question varchar(20) NOT NULL,
    evidence_tag varchar(20) NOT NULL,
    taxonomy varchar(60) NOT NULL,
    supported bit NOT NULL,
    power_label varchar(30) NOT NULL,
    rationale nvarchar(1000) NOT NULL,
    metric_id bigint NULL
  );

IF OBJECT_ID(N'telemetry.model_requests', N'U') IS NULL
  CREATE TABLE telemetry.model_requests
  (
    request_id varchar(120) NOT NULL CONSTRAINT pk_tel_model_requests PRIMARY KEY,
    profile_id varchar(80) NOT NULL,
    queue_ms float NULL,
    model_ms float NOT NULL,
    extract_ms float NULL,
    persist_ms float NULL,
    total_ms float NOT NULL,
    prompt_tokens int NULL,
    completion_tokens int NULL,
    tokens_per_second float NULL,
    server_kind varchar(30) NOT NULL,
    recorded_at_utc datetime2(3) NOT NULL CONSTRAINT df_tel_recorded DEFAULT SYSUTCDATETIME()
  );

IF OBJECT_ID(N'telemetry.host_samples', N'U') IS NULL
  CREATE TABLE telemetry.host_samples
  (
    sample_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_host_samples PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    phase varchar(60) NOT NULL,
    sample_json nvarchar(max) NOT NULL,
    sampled_at_utc datetime2(3) NOT NULL CONSTRAINT df_host_sampled DEFAULT SYSUTCDATETIME()
  );
GO
