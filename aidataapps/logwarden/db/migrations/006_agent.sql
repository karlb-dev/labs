SET XACT_ABORT ON;

IF OBJECT_ID(N'agent.agent_runs', N'U') IS NULL
BEGIN
  CREATE TABLE agent.agent_runs
  (
    agent_run_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_agent_runs PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    job_id bigint NOT NULL,
    job_attempt_id bigint NOT NULL,
    work_item_id bigint NULL,
    episode_id varchar(120) NOT NULL,
    model_profile_id varchar(80) NOT NULL,
    agent_arm_id varchar(80) NOT NULL,
    decode_config_id varchar(40) NOT NULL,
    trace_id char(32) NOT NULL,
    prompt_contract_sha256 char(64) NOT NULL,
    tool_registry_sha256 char(64) NOT NULL,
    status varchar(32) NOT NULL,
    terminal_reason nvarchar(1000) NULL,
    started_at_utc datetime2(7) NOT NULL,
    finished_at_utc datetime2(7) NULL,
    elapsed_ms decimal(18,3) NULL,
    CONSTRAINT uq_agent_runs_attempt UNIQUE(job_attempt_id),
    CONSTRAINT uq_agent_runs_trace UNIQUE(trace_id),
    CONSTRAINT fk_agent_runs_control_run FOREIGN KEY(run_id) REFERENCES control.runs(run_id),
    CONSTRAINT fk_agent_runs_job FOREIGN KEY(job_id) REFERENCES control.jobs(job_id),
    CONSTRAINT fk_agent_runs_attempt FOREIGN KEY(job_attempt_id) REFERENCES control.job_attempts(job_attempt_id),
    CONSTRAINT fk_agent_runs_work_item FOREIGN KEY(work_item_id) REFERENCES ops.work_items(work_item_id),
    CONSTRAINT fk_agent_runs_model FOREIGN KEY(model_profile_id) REFERENCES control.model_profiles(model_profile_id),
    CONSTRAINT fk_agent_runs_arm FOREIGN KEY(agent_arm_id) REFERENCES control.agent_arms(agent_arm_id),
    CONSTRAINT fk_agent_runs_decode FOREIGN KEY(decode_config_id) REFERENCES control.decode_configs(decode_config_id),
    CONSTRAINT ck_agent_runs_status CHECK (status IN ('running','complete','needs_human_review','contract_rejected','policy_rejected','model_timeout','tool_timeout','failed','stopped'))
  );
  CREATE INDEX ix_agent_runs_episode ON agent.agent_runs(run_id, episode_id, agent_run_id);
END;

IF OBJECT_ID(N'agent.turns', N'U') IS NULL
BEGIN
  CREATE TABLE agent.turns
  (
    turn_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_agent_turns PRIMARY KEY,
    agent_run_id bigint NOT NULL,
    turn_ordinal int NOT NULL,
    state_before varchar(40) NOT NULL,
    state_after varchar(40) NULL,
    messages_sha256 char(64) NOT NULL,
    sent_messages_sha256 char(64) NOT NULL,
    prompt_bytes int NOT NULL,
    status varchar(32) NOT NULL,
    started_at_utc datetime2(7) NOT NULL,
    finished_at_utc datetime2(7) NULL,
    elapsed_ms decimal(18,3) NULL,
    CONSTRAINT uq_agent_turn_ordinal UNIQUE(agent_run_id, turn_ordinal),
    CONSTRAINT fk_agent_turns_run FOREIGN KEY(agent_run_id) REFERENCES agent.agent_runs(agent_run_id),
    CONSTRAINT ck_agent_turns_ordinal CHECK (turn_ordinal BETWEEN 0 AND 20),
    CONSTRAINT ck_agent_turns_bytes CHECK (prompt_bytes >= 0)
  );
END;

IF OBJECT_ID(N'agent.model_requests', N'U') IS NULL
BEGIN
  CREATE TABLE agent.model_requests
  (
    model_request_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_agent_model_requests PRIMARY KEY,
    turn_id bigint NOT NULL,
    client_request_id varchar(160) NOT NULL CONSTRAINT uq_agent_model_requests_client UNIQUE,
    endpoint_uri nvarchar(1000) NOT NULL,
    endpoint_identity_sha256 char(64) NOT NULL,
    messages_sha256 char(64) NOT NULL,
    prompt_sha256 char(64) NOT NULL,
    tool_registry_sha256 char(64) NOT NULL,
    response_schema_sha256 char(64) NOT NULL,
    decode_config_sha256 char(64) NOT NULL,
    request_body varbinary(max) NOT NULL,
    request_body_sha256 char(64) NOT NULL,
    request_bytes int NOT NULL,
    retry_ordinal int NOT NULL,
    started_at_utc datetime2(7) NOT NULL,
    request_write_finished_at_utc datetime2(7) NULL,
    response_headers_at_utc datetime2(7) NULL,
    first_content_at_utc datetime2(7) NULL,
    body_finished_at_utc datetime2(7) NULL,
    client_elapsed_ms decimal(18,3) NULL,
    connect_write_ms decimal(18,3) NULL,
    headers_wait_ms decimal(18,3) NULL,
    body_read_ms decimal(18,3) NULL,
    status varchar(32) NOT NULL,
    error_class varchar(80) NULL,
    error_detail nvarchar(max) NULL,
    CONSTRAINT fk_agent_requests_turn FOREIGN KEY(turn_id) REFERENCES agent.turns(turn_id),
    CONSTRAINT ck_agent_requests_bytes CHECK (request_bytes >= 0),
    CONSTRAINT ck_agent_requests_retry CHECK (retry_ordinal BETWEEN 0 AND 20)
  );
  CREATE INDEX ix_agent_requests_turn ON agent.model_requests(turn_id, retry_ordinal, model_request_id);
END;

IF OBJECT_ID(N'agent.model_responses', N'U') IS NULL
BEGIN
  CREATE TABLE agent.model_responses
  (
    model_response_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_agent_model_responses PRIMARY KEY,
    model_request_id bigint NOT NULL CONSTRAINT uq_agent_model_responses_request UNIQUE,
    http_status int NULL,
    service_request_id nvarchar(300) NULL,
    response_body varbinary(max) NOT NULL,
    response_body_sha256 char(64) NOT NULL,
    response_bytes int NOT NULL,
    finish_reason varchar(80) NULL,
    prompt_tokens bigint NULL,
    completion_tokens bigint NULL,
    total_tokens bigint NULL,
    token_count_provenance varchar(32) NULL,
    server_queue_ms decimal(18,3) NULL,
    server_service_ms decimal(18,3) NULL,
    server_ttft_ms decimal(18,3) NULL,
    server_decode_ms decimal(18,3) NULL,
    parse_started_at_utc datetime2(7) NULL,
    parse_finished_at_utc datetime2(7) NULL,
    parse_ms decimal(18,3) NULL,
    parse_status varchar(32) NOT NULL,
    repair_kind varchar(40) NOT NULL,
    structured_json nvarchar(max) NULL,
    error_class varchar(80) NULL,
    error_detail nvarchar(max) NULL,
    CONSTRAINT fk_agent_responses_request FOREIGN KEY(model_request_id) REFERENCES agent.model_requests(model_request_id),
    CONSTRAINT ck_agent_responses_bytes CHECK (response_bytes >= 0),
    CONSTRAINT ck_agent_responses_token_provenance CHECK (token_count_provenance IS NULL OR token_count_provenance IN ('server_reported','tokenizer_counted','estimated')),
    CONSTRAINT ck_agent_responses_repair CHECK (repair_kind IN ('none','strip_code_fence','strip_leading_text','rejected')),
    CONSTRAINT ck_agent_responses_json CHECK (structured_json IS NULL OR ISJSON(structured_json) = 1)
  );
END;

IF OBJECT_ID(N'agent.agent_steps', N'U') IS NULL
BEGIN
  CREATE TABLE agent.agent_steps
  (
    agent_step_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_agent_steps PRIMARY KEY,
    agent_run_id bigint NOT NULL,
    turn_id bigint NULL,
    step_ordinal int NOT NULL,
    step_kind varchar(40) NOT NULL,
    span_id char(16) NOT NULL,
    parent_span_id char(16) NULL,
    status varchar(32) NOT NULL,
    started_at_utc datetime2(7) NOT NULL,
    finished_at_utc datetime2(7) NULL,
    duration_ms decimal(18,3) NULL,
    attributes_json nvarchar(max) NOT NULL,
    error_class varchar(80) NULL,
    error_detail nvarchar(max) NULL,
    CONSTRAINT uq_agent_steps_ordinal UNIQUE(agent_run_id, step_ordinal),
    CONSTRAINT uq_agent_steps_span UNIQUE(agent_run_id, span_id),
    CONSTRAINT fk_agent_steps_run FOREIGN KEY(agent_run_id) REFERENCES agent.agent_runs(agent_run_id),
    CONSTRAINT fk_agent_steps_turn FOREIGN KEY(turn_id) REFERENCES agent.turns(turn_id),
    CONSTRAINT ck_agent_steps_json CHECK (ISJSON(attributes_json) = 1)
  );
  CREATE INDEX ix_agent_steps_turn ON agent.agent_steps(turn_id, agent_step_id);
END;

IF OBJECT_ID(N'agent.tool_invocations', N'U') IS NULL
BEGIN
  CREATE TABLE agent.tool_invocations
  (
    tool_invocation_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_agent_tool_invocations PRIMARY KEY,
    agent_step_id bigint NOT NULL,
    agent_run_id bigint NOT NULL,
    turn_id bigint NOT NULL,
    tool_call_id varchar(160) NOT NULL,
    tool_name varchar(80) NOT NULL,
    canonical_args_json nvarchar(max) NOT NULL,
    canonical_args_sha256 char(64) NOT NULL,
    policy_status varchar(32) NOT NULL,
    policy_reason nvarchar(1000) NOT NULL,
    execution_mode varchar(24) NOT NULL,
    cache_hit bit NOT NULL,
    snapshot_miss bit NOT NULL,
    started_at_utc datetime2(7) NOT NULL,
    finished_at_utc datetime2(7) NULL,
    latency_ms decimal(18,3) NULL,
    result_json nvarchar(max) NULL,
    result_sha256 char(64) NULL,
    result_bytes int NULL,
    row_count int NULL,
    truncated bit NOT NULL,
    status varchar(32) NOT NULL,
    error_class varchar(80) NULL,
    error_detail nvarchar(max) NULL,
    CONSTRAINT uq_agent_tool_call UNIQUE(agent_run_id, tool_call_id),
    CONSTRAINT fk_agent_tools_step FOREIGN KEY(agent_step_id) REFERENCES agent.agent_steps(agent_step_id),
    CONSTRAINT fk_agent_tools_run FOREIGN KEY(agent_run_id) REFERENCES agent.agent_runs(agent_run_id),
    CONSTRAINT fk_agent_tools_turn FOREIGN KEY(turn_id) REFERENCES agent.turns(turn_id),
    CONSTRAINT ck_agent_tools_args_json CHECK (ISJSON(canonical_args_json) = 1),
    CONSTRAINT ck_agent_tools_result_json CHECK (result_json IS NULL OR ISJSON(result_json) = 1),
    CONSTRAINT ck_agent_tools_mode CHECK (execution_mode IN ('replay','live','cache')),
    CONSTRAINT ck_agent_tools_bytes CHECK (result_bytes IS NULL OR result_bytes >= 0)
  );
  CREATE INDEX ix_agent_tools_run ON agent.tool_invocations(agent_run_id, turn_id, tool_invocation_id);
END;

IF OBJECT_ID(N'agent.validation_events', N'U') IS NULL
BEGIN
  CREATE TABLE agent.validation_events
  (
    validation_event_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_agent_validation_events PRIMARY KEY,
    agent_run_id bigint NOT NULL,
    turn_id bigint NULL,
    model_response_id bigint NULL,
    tool_invocation_id bigint NULL,
    layer varchar(40) NOT NULL,
    rule_id varchar(120) NOT NULL,
    outcome varchar(24) NOT NULL,
    repair_kind varchar(40) NULL,
    detail_json nvarchar(max) NOT NULL,
    at_utc datetime2(7) NOT NULL CONSTRAINT df_agent_validation_at DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_agent_validation_run FOREIGN KEY(agent_run_id) REFERENCES agent.agent_runs(agent_run_id),
    CONSTRAINT fk_agent_validation_turn FOREIGN KEY(turn_id) REFERENCES agent.turns(turn_id),
    CONSTRAINT fk_agent_validation_response FOREIGN KEY(model_response_id) REFERENCES agent.model_responses(model_response_id),
    CONSTRAINT fk_agent_validation_tool FOREIGN KEY(tool_invocation_id) REFERENCES agent.tool_invocations(tool_invocation_id),
    CONSTRAINT ck_agent_validation_outcome CHECK (outcome IN ('pass','fail','warn','repaired')),
    CONSTRAINT ck_agent_validation_json CHECK (ISJSON(detail_json) = 1)
  );
END;

IF OBJECT_ID(N'agent.decisions', N'U') IS NULL
BEGIN
  CREATE TABLE agent.decisions
  (
    decision_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_agent_decisions PRIMARY KEY,
    agent_run_id bigint NOT NULL CONSTRAINT uq_agent_decisions_run UNIQUE,
    turn_id bigint NOT NULL,
    incident_id bigint NULL,
    episode_id varchar(120) NOT NULL,
    incident_class varchar(80) NOT NULL,
    severity varchar(24) NOT NULL,
    action varchar(100) NOT NULL,
    confidence decimal(9,6) NOT NULL,
    abstained bit NOT NULL,
    summary nvarchar(2000) NOT NULL,
    correlation_key varchar(160) NULL,
    decision_json nvarchar(max) NOT NULL,
    decision_sha256 char(64) NOT NULL,
    contract_valid bit NOT NULL,
    policy_valid bit NOT NULL,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_agent_decisions_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_agent_decisions_run FOREIGN KEY(agent_run_id) REFERENCES agent.agent_runs(agent_run_id),
    CONSTRAINT fk_agent_decisions_turn FOREIGN KEY(turn_id) REFERENCES agent.turns(turn_id),
    CONSTRAINT fk_agent_decisions_incident FOREIGN KEY(incident_id) REFERENCES ops.incidents(incident_id),
    CONSTRAINT ck_agent_decisions_confidence CHECK (confidence BETWEEN 0 AND 1),
    CONSTRAINT ck_agent_decisions_json CHECK (ISJSON(decision_json) = 1)
  );
  CREATE INDEX ix_agent_decisions_episode ON agent.decisions(episode_id, decision_id);
END;

IF OBJECT_ID(N'agent.decision_citations', N'U') IS NULL
BEGIN
  CREATE TABLE agent.decision_citations
  (
    decision_id bigint NOT NULL,
    citation_ordinal int NOT NULL,
    chunk_id varchar(120) NOT NULL,
    retrieval_run_id bigint NULL,
    returned_to_agent bit NOT NULL,
    CONSTRAINT pk_agent_decision_citations PRIMARY KEY(decision_id, citation_ordinal),
    CONSTRAINT uq_agent_decision_chunk UNIQUE(decision_id, chunk_id),
    CONSTRAINT fk_agent_citations_decision FOREIGN KEY(decision_id) REFERENCES agent.decisions(decision_id),
    CONSTRAINT fk_agent_citations_chunk FOREIGN KEY(chunk_id) REFERENCES kb.runbook_chunks(chunk_id),
    CONSTRAINT fk_agent_citations_retrieval FOREIGN KEY(retrieval_run_id) REFERENCES kb.retrieval_runs(retrieval_run_id)
  );
END;

IF OBJECT_ID(N'agent.policy_events', N'U') IS NULL
BEGIN
  CREATE TABLE agent.policy_events
  (
    policy_event_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_agent_policy_events PRIMARY KEY,
    agent_run_id bigint NOT NULL,
    turn_id bigint NULL,
    decision_id bigint NULL,
    policy_version varchar(40) NOT NULL,
    policy_point varchar(80) NOT NULL,
    outcome varchar(32) NOT NULL,
    reason_code varchar(80) NOT NULL,
    input_sha256 char(64) NOT NULL,
    detail_json nvarchar(max) NOT NULL,
    at_utc datetime2(7) NOT NULL CONSTRAINT df_agent_policy_at DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_agent_policy_run FOREIGN KEY(agent_run_id) REFERENCES agent.agent_runs(agent_run_id),
    CONSTRAINT fk_agent_policy_turn FOREIGN KEY(turn_id) REFERENCES agent.turns(turn_id),
    CONSTRAINT fk_agent_policy_decision FOREIGN KEY(decision_id) REFERENCES agent.decisions(decision_id),
    CONSTRAINT ck_agent_policy_json CHECK (ISJSON(detail_json) = 1)
  );
END;

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'fk_ops_action_proposals_decision')
  ALTER TABLE ops.action_proposals WITH CHECK ADD CONSTRAINT fk_ops_action_proposals_decision
    FOREIGN KEY(decision_id) REFERENCES agent.decisions(decision_id);
