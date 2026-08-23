SET XACT_ABORT ON;

IF USER_ID(N'lw_lab') IS NULL CREATE USER [lw_lab] FOR LOGIN [lw_lab];
IF USER_ID(N'lw_agent') IS NULL CREATE USER [lw_agent] FOR LOGIN [lw_agent];
IF DATABASE_PRINCIPAL_ID(N'lw_lab_role') IS NULL CREATE ROLE [lw_lab_role] AUTHORIZATION [dbo];
IF DATABASE_PRINCIPAL_ID(N'lw_agent_role') IS NULL CREATE ROLE [lw_agent_role] AUTHORIZATION [dbo];

IF IS_ROLEMEMBER(N'lw_lab_role', N'lw_lab') <> 1 ALTER ROLE [lw_lab_role] ADD MEMBER [lw_lab];
IF IS_ROLEMEMBER(N'lw_agent_role', N'lw_agent') <> 1 ALTER ROLE [lw_agent_role] ADD MEMBER [lw_agent];

-- The Tier 1 operator principal owns migrations, injection, ingestion,
-- evaluation, and reporting. It is intentionally distinct from the runtime
-- agent principal and is not a server administrator.
GRANT CONTROL TO [lw_lab_role];

-- The runtime agent can only claim governed work and call reviewed
-- procedures. Ownership chaining keeps underlying packet/snapshot and
-- knowledge tables unreadable through arbitrary queries.
GRANT EXECUTE ON OBJECT::ops.usp_claim_work_item TO [lw_agent_role];
GRANT EXECUTE ON OBJECT::ops.usp_heartbeat_work_item TO [lw_agent_role];
GRANT EXECUTE ON OBJECT::ops.usp_complete_work_item TO [lw_agent_role];
GRANT EXECUTE ON OBJECT::kb.search_runbook_exact TO [lw_agent_role];
GRANT SELECT ON OBJECT::ops.work_items TO [lw_agent_role];
GRANT SELECT ON OBJECT::ops.incidents TO [lw_agent_role];

DENY SELECT ON SCHEMA::eval TO [lw_agent_role];
DENY SELECT ON SCHEMA::workload TO [lw_agent_role];
DENY SELECT ON OBJECT::ingest.context_snapshots TO [lw_agent_role];
DENY SELECT ON OBJECT::ingest.incident_packets TO [lw_agent_role];
DENY SELECT ON OBJECT::control.jobs TO [lw_agent_role];
DENY INSERT, UPDATE, DELETE ON SCHEMA::control TO [lw_agent_role];
DENY INSERT, UPDATE, DELETE ON SCHEMA::ingest TO [lw_agent_role];
DENY INSERT, UPDATE, DELETE ON SCHEMA::kb TO [lw_agent_role];
DENY INSERT, UPDATE, DELETE ON SCHEMA::eval TO [lw_agent_role];
DENY ALTER ANY SCHEMA TO [lw_agent_role];
DENY CREATE TABLE TO [lw_agent_role];
DENY CREATE PROCEDURE TO [lw_agent_role];
DENY CREATE FUNCTION TO [lw_agent_role];
DENY BACKUP DATABASE TO [lw_agent_role];
DENY BACKUP LOG TO [lw_agent_role];

-- Immutable evidence is append-only for both application identities. Schema
-- owners retain emergency recovery authority, which is audited separately.
DENY UPDATE, DELETE ON OBJECT::ops.transitions TO [lw_agent_role];
DENY UPDATE, DELETE ON OBJECT::agent.model_requests TO [lw_agent_role];
DENY UPDATE, DELETE ON OBJECT::agent.model_responses TO [lw_agent_role];
DENY UPDATE, DELETE ON OBJECT::agent.tool_invocations TO [lw_agent_role];
DENY UPDATE, DELETE ON OBJECT::agent.validation_events TO [lw_agent_role];
DENY UPDATE, DELETE ON OBJECT::agent.decisions TO [lw_agent_role];
DENY UPDATE, DELETE ON OBJECT::agent.policy_events TO [lw_agent_role];
DENY UPDATE, DELETE ON OBJECT::telemetry.journal_events TO [lw_agent_role];
DENY UPDATE, DELETE ON OBJECT::telemetry.metric_samples TO [lw_agent_role];
GO

CREATE OR ALTER PROCEDURE ops.usp_transition_work_item
  @work_item_id bigint,
  @lease_token uniqueidentifier,
  @to_state varchar(40),
  @actor varchar(120),
  @reason nvarchar(1000)
AS
BEGIN
  SET NOCOUNT ON;
  SET XACT_ABORT ON;
  DECLARE @from_state varchar(40);
  BEGIN TRANSACTION;
  SELECT @from_state = status
  FROM ops.work_items WITH (UPDLOCK, ROWLOCK)
  WHERE work_item_id = @work_item_id
    AND lease_token = @lease_token
    AND leased_until_utc >= SYSUTCDATETIME();
  IF @from_state IS NULL
  BEGIN
    ROLLBACK;
    THROW 51200, 'Stale, expired, or missing work-item lease', 1;
  END;
  IF NOT EXISTS
  (
    SELECT 1
    FROM (VALUES
      ('leased','packet_loaded'),
      ('packet_loaded','model_requested'),
      ('model_requested','tool_requested'),
      ('model_requested','decision_received'),
      ('tool_requested','tool_completed'),
      ('tool_completed','model_requested'),
      ('decision_received','validated'),
      ('validated','persisted'),
      ('persisted','proposed_action_recorded'),
      ('persisted','complete'),
      ('proposed_action_recorded','complete'),
      ('model_requested','model_timeout'),
      ('tool_requested','tool_timeout'),
      ('model_requested','contract_rejected'),
      ('decision_received','policy_rejected'),
      ('leased','retryable_failure'),
      ('packet_loaded','retryable_failure'),
      ('model_requested','retryable_failure'),
      ('tool_requested','retryable_failure')
    ) AS allowed(from_state, to_state)
    WHERE allowed.from_state = @from_state AND allowed.to_state = @to_state
  )
  BEGIN
    ROLLBACK;
    THROW 51201, 'Invalid work-item state transition', 1;
  END;
  UPDATE ops.work_items
  SET status = @to_state,
      completed_at_utc = CASE WHEN @to_state = 'complete' THEN SYSUTCDATETIME() ELSE completed_at_utc END
  WHERE work_item_id = @work_item_id AND lease_token = @lease_token;
  INSERT ops.transitions(entity_kind, entity_id, from_state, to_state, actor, reason)
  VALUES('work_item', @work_item_id, @from_state, @to_state, @actor, @reason);
  COMMIT;
END;
GO

CREATE OR ALTER PROCEDURE ops.usp_open_work_tracking_item
  @action_proposal_id bigint,
  @incident_id bigint = NULL,
  @title nvarchar(300),
  @description nvarchar(2000),
  @priority varchar(24),
  @idempotency_key char(64)
AS
BEGIN
  SET NOCOUNT ON;
  IF NOT EXISTS (SELECT 1 FROM ops.action_proposals WHERE action_proposal_id = @action_proposal_id AND policy_status = 'allowed')
    THROW 51202, 'Action proposal is missing or not policy-allowed', 1;
  INSERT ops.work_tracking_items(action_proposal_id, incident_id, title, description, priority, status, idempotency_key)
  SELECT @action_proposal_id, @incident_id, @title, @description, @priority, 'open', @idempotency_key
  WHERE NOT EXISTS (SELECT 1 FROM ops.work_tracking_items WHERE idempotency_key = @idempotency_key);
  SELECT * FROM ops.work_tracking_items WHERE idempotency_key = @idempotency_key;
END;
GO

GRANT EXECUTE ON OBJECT::ops.usp_transition_work_item TO [lw_agent_role];
GRANT EXECUTE ON OBJECT::ops.usp_open_work_tracking_item TO [lw_agent_role];
