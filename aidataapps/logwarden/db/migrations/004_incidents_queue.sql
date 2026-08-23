SET XACT_ABORT ON;

IF OBJECT_ID(N'ops.incidents', N'U') IS NULL
BEGIN
  CREATE TABLE ops.incidents
  (
    incident_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_ops_incidents PRIMARY KEY,
    correlation_key varchar(160) NOT NULL CONSTRAINT uq_ops_incidents_correlation UNIQUE,
    source_mode varchar(24) NOT NULL,
    first_event_at_utc datetime2(7) NOT NULL,
    last_event_at_utc datetime2(7) NOT NULL,
    current_class varchar(80) NOT NULL,
    current_severity varchar(24) NOT NULL,
    status varchar(24) NOT NULL,
    event_count int NOT NULL,
    decision_count int NOT NULL,
    active_work_item_id bigint NULL,
    row_version rowversion NOT NULL,
    CONSTRAINT ck_ops_incidents_mode CHECK (source_mode IN ('replay','live')),
    CONSTRAINT ck_ops_incidents_counts CHECK (event_count >= 0 AND decision_count >= 0)
  );
END;

IF OBJECT_ID(N'ops.incident_events', N'U') IS NULL
BEGIN
  CREATE TABLE ops.incident_events
  (
    incident_id bigint NOT NULL,
    canonical_event_id bigint NOT NULL,
    linked_at_utc datetime2(7) NOT NULL CONSTRAINT df_ops_incident_events_linked DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_ops_incident_events PRIMARY KEY(incident_id, canonical_event_id),
    CONSTRAINT fk_ops_incident_events_incident FOREIGN KEY(incident_id) REFERENCES ops.incidents(incident_id),
    CONSTRAINT fk_ops_incident_events_event FOREIGN KEY(canonical_event_id) REFERENCES ingest.canonical_events(canonical_event_id)
  );
END;

IF OBJECT_ID(N'ops.work_items', N'U') IS NULL
BEGIN
  CREATE TABLE ops.work_items
  (
    work_item_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_ops_work_items PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    job_id bigint NULL,
    episode_id varchar(120) NOT NULL,
    incident_id bigint NULL,
    status varchar(24) NOT NULL CONSTRAINT df_ops_work_items_status DEFAULT('pending'),
    lease_owner varchar(120) NULL,
    lease_token uniqueidentifier NULL,
    leased_until_utc datetime2(7) NULL,
    attempt_count int NOT NULL CONSTRAINT df_ops_work_items_attempt DEFAULT(0),
    next_attempt_at_utc datetime2(7) NOT NULL CONSTRAINT df_ops_work_items_next DEFAULT SYSUTCDATETIME(),
    priority int NOT NULL CONSTRAINT df_ops_work_items_priority DEFAULT(0),
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_ops_work_items_created DEFAULT SYSUTCDATETIME(),
    completed_at_utc datetime2(7) NULL,
    row_version rowversion NOT NULL,
    CONSTRAINT uq_ops_work_items_job UNIQUE(job_id),
    CONSTRAINT fk_ops_work_items_job FOREIGN KEY(job_id) REFERENCES control.jobs(job_id),
    CONSTRAINT fk_ops_work_items_incident FOREIGN KEY(incident_id) REFERENCES ops.incidents(incident_id),
    CONSTRAINT ck_ops_work_items_status CHECK (status IN ('pending','leased','packet_loaded','model_requested','tool_requested','tool_completed','decision_received','validated','persisted','proposed_action_recorded','complete','retryable_failure','contract_rejected','policy_rejected','model_timeout','tool_timeout','lease_expired','stopped'))
  );
  CREATE INDEX ix_work_items_claim ON ops.work_items(status, next_attempt_at_utc, priority DESC, work_item_id)
    INCLUDE(lease_token, leased_until_utc);
END;

IF OBJECT_ID(N'ops.transitions', N'U') IS NULL
BEGIN
  CREATE TABLE ops.transitions
  (
    transition_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_ops_transitions PRIMARY KEY,
    entity_kind varchar(24) NOT NULL,
    entity_id bigint NOT NULL,
    from_state varchar(40) NULL,
    to_state varchar(40) NOT NULL,
    actor varchar(120) NOT NULL,
    reason nvarchar(1000) NOT NULL,
    at_utc datetime2(7) NOT NULL CONSTRAINT df_ops_transitions_at DEFAULT SYSUTCDATETIME(),
    CONSTRAINT ck_ops_transitions_kind CHECK (entity_kind IN ('incident','work_item','work_tracking_item'))
  );
  CREATE INDEX ix_ops_transitions_entity ON ops.transitions(entity_kind, entity_id, at_utc, transition_id);
END;

IF OBJECT_ID(N'ops.action_proposals', N'U') IS NULL
BEGIN
  CREATE TABLE ops.action_proposals
  (
    action_proposal_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_ops_action_proposals PRIMARY KEY,
    decision_id bigint NOT NULL,
    incident_id bigint NULL,
    action_type varchar(80) NOT NULL,
    action_arguments_json nvarchar(max) NOT NULL,
    policy_status varchar(24) NOT NULL,
    policy_reason nvarchar(1000) NOT NULL,
    caller_opted_in bit NOT NULL,
    execution_status varchar(24) NOT NULL,
    idempotency_key char(64) NOT NULL CONSTRAINT uq_ops_action_proposals_key UNIQUE,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_ops_action_proposals_created DEFAULT SYSUTCDATETIME(),
    executed_at_utc datetime2(7) NULL,
    CONSTRAINT fk_ops_action_proposals_incident FOREIGN KEY(incident_id) REFERENCES ops.incidents(incident_id),
    CONSTRAINT ck_ops_action_proposals_json CHECK (ISJSON(action_arguments_json) = 1)
  );
END;

IF OBJECT_ID(N'ops.work_tracking_items', N'U') IS NULL
BEGIN
  CREATE TABLE ops.work_tracking_items
  (
    work_tracking_item_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_ops_work_tracking_items PRIMARY KEY,
    action_proposal_id bigint NOT NULL,
    incident_id bigint NULL,
    title nvarchar(300) NOT NULL,
    description nvarchar(2000) NOT NULL,
    priority varchar(24) NOT NULL,
    status varchar(24) NOT NULL,
    idempotency_key char(64) NOT NULL CONSTRAINT uq_ops_work_tracking_key UNIQUE,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_ops_work_tracking_created DEFAULT SYSUTCDATETIME(),
    updated_at_utc datetime2(7) NOT NULL CONSTRAINT df_ops_work_tracking_updated DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_ops_tracking_proposal FOREIGN KEY(action_proposal_id) REFERENCES ops.action_proposals(action_proposal_id),
    CONSTRAINT fk_ops_tracking_incident FOREIGN KEY(incident_id) REFERENCES ops.incidents(incident_id)
  );
END;
GO

CREATE OR ALTER PROCEDURE ops.usp_claim_work_item
  @worker_id varchar(120),
  @lease_token uniqueidentifier,
  @lease_seconds int = 240
AS
BEGIN
  SET NOCOUNT ON;
  SET XACT_ABORT ON;
  IF @lease_seconds NOT BETWEEN 30 AND 1800 THROW 51000, 'Invalid lease duration', 1;
  BEGIN TRANSACTION;
  ;WITH next_item AS
  (
    SELECT TOP (1) *
    FROM ops.work_items WITH (UPDLOCK, READPAST, ROWLOCK, INDEX(ix_work_items_claim))
    WHERE (status = 'pending' OR (status = 'leased' AND leased_until_utc < SYSUTCDATETIME()))
      AND next_attempt_at_utc <= SYSUTCDATETIME()
    ORDER BY priority DESC, work_item_id ASC
  )
  UPDATE next_item
  SET status = 'leased',
      lease_owner = @worker_id,
      lease_token = @lease_token,
      leased_until_utc = DATEADD(SECOND, @lease_seconds, SYSUTCDATETIME()),
      attempt_count = attempt_count + 1
  OUTPUT INSERTED.*;
  COMMIT;
END;
GO

CREATE OR ALTER PROCEDURE ops.usp_heartbeat_work_item
  @work_item_id bigint,
  @lease_token uniqueidentifier,
  @lease_seconds int = 240
AS
BEGIN
  SET NOCOUNT ON;
  UPDATE ops.work_items
  SET leased_until_utc = DATEADD(SECOND, @lease_seconds, SYSUTCDATETIME())
  WHERE work_item_id = @work_item_id AND status = 'leased' AND lease_token = @lease_token;
  IF @@ROWCOUNT <> 1 THROW 51001, 'Stale or missing lease token', 1;
END;
GO

CREATE OR ALTER PROCEDURE ops.usp_complete_work_item
  @work_item_id bigint,
  @lease_token uniqueidentifier
AS
BEGIN
  SET NOCOUNT ON;
  SET XACT_ABORT ON;
  BEGIN TRANSACTION;
  UPDATE ops.work_items
  SET status = 'complete', completed_at_utc = SYSUTCDATETIME(), leased_until_utc = NULL
  WHERE work_item_id = @work_item_id AND lease_token = @lease_token AND status IN ('leased','persisted','proposed_action_recorded');
  IF @@ROWCOUNT <> 1
  BEGIN
    ROLLBACK;
    THROW 51002, 'Stale token or invalid terminal transition', 1;
  END;
  INSERT ops.transitions(entity_kind, entity_id, from_state, to_state, actor, reason)
  VALUES('work_item', @work_item_id, 'leased', 'complete', ORIGINAL_LOGIN(), 'lease-token completion');
  COMMIT;
END;
GO
