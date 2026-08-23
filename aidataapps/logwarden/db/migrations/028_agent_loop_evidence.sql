SET XACT_ABORT ON;

IF COL_LENGTH(N'agent.agent_runs', N'loop_budget_json') IS NULL
  ALTER TABLE agent.agent_runs ADD
    loop_budget_json nvarchar(max) NULL,
    model_turn_count int NULL,
    tool_call_count int NULL,
    tool_result_chars int NULL;

IF COL_LENGTH(N'agent.tool_invocations', N'raw_result_path') IS NULL
  ALTER TABLE agent.tool_invocations ADD
    raw_result_path nvarchar(1000) NULL,
    raw_result_sha256 char(64) NULL,
    raw_result_bytes int NULL,
    result_chars int NULL,
    retrieval_run_id bigint NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name=N'ck_agent_runs_loop_budget_json')
  ALTER TABLE agent.agent_runs WITH CHECK ADD CONSTRAINT ck_agent_runs_loop_budget_json
    CHECK (loop_budget_json IS NULL OR ISJSON(loop_budget_json)=1);

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name=N'ck_agent_runs_loop_counts')
  ALTER TABLE agent.agent_runs WITH CHECK ADD CONSTRAINT ck_agent_runs_loop_counts
    CHECK
    (
      (model_turn_count IS NULL OR model_turn_count>=0)
      AND (tool_call_count IS NULL OR tool_call_count>=0)
      AND (tool_result_chars IS NULL OR tool_result_chars>=0)
    );

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name=N'ck_agent_tools_raw_result')
  ALTER TABLE agent.tool_invocations WITH CHECK ADD CONSTRAINT ck_agent_tools_raw_result
    CHECK
    (
      (raw_result_path IS NULL AND raw_result_sha256 IS NULL AND raw_result_bytes IS NULL)
      OR
      (raw_result_path IS NOT NULL AND raw_result_sha256 IS NOT NULL AND raw_result_bytes>=0)
    );

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name=N'ck_agent_tools_result_chars')
  ALTER TABLE agent.tool_invocations WITH CHECK ADD CONSTRAINT ck_agent_tools_result_chars
    CHECK (result_chars IS NULL OR result_chars>=0);

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name=N'fk_agent_tools_retrieval')
  ALTER TABLE agent.tool_invocations WITH CHECK ADD CONSTRAINT fk_agent_tools_retrieval
    FOREIGN KEY(retrieval_run_id) REFERENCES kb.retrieval_runs(retrieval_run_id);
GO

CREATE OR ALTER PROCEDURE ops.usp_transition_work_item
  @work_item_id bigint,
  @lease_token uniqueidentifier,
  @to_state varchar(40),
  @actor varchar(120),
  @reason nvarchar(1000),
  @retry_after_seconds int = 5
AS
BEGIN
  SET NOCOUNT ON;
  SET XACT_ABORT ON;
  IF @retry_after_seconds NOT BETWEEN 0 AND 3600 THROW 51203, 'Invalid retry delay', 1;
  DECLARE @from_state varchar(40);
  DECLARE @is_terminal bit = CASE WHEN @to_state IN
    ('complete','retryable_failure','contract_rejected','policy_rejected','model_timeout','tool_timeout','stopped')
    THEN 1 ELSE 0 END;
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
      ('tool_completed','tool_timeout'),
      ('model_requested','contract_rejected'),
      ('decision_received','contract_rejected'),
      ('decision_received','policy_rejected'),
      ('model_requested','policy_rejected'),
      ('tool_requested','policy_rejected'),
      ('tool_completed','policy_rejected'),
      ('leased','retryable_failure'),
      ('packet_loaded','retryable_failure'),
      ('model_requested','retryable_failure'),
      ('tool_requested','retryable_failure'),
      ('tool_completed','retryable_failure'),
      ('decision_received','retryable_failure'),
      ('validated','retryable_failure'),
      ('leased','stopped'),
      ('packet_loaded','stopped'),
      ('model_requested','stopped'),
      ('tool_requested','stopped'),
      ('tool_completed','stopped'),
      ('decision_received','stopped'),
      ('validated','stopped')
    ) AS allowed(from_state, to_state)
    WHERE allowed.from_state = @from_state AND allowed.to_state = @to_state
  )
  BEGIN
    ROLLBACK;
    THROW 51201, 'Invalid work-item state transition', 1;
  END;
  UPDATE ops.work_items
  SET status = @to_state,
      completed_at_utc = CASE WHEN @to_state = 'complete' THEN SYSUTCDATETIME() ELSE completed_at_utc END,
      next_attempt_at_utc = CASE WHEN @to_state = 'retryable_failure'
                                 THEN DATEADD(SECOND, @retry_after_seconds, SYSUTCDATETIME())
                                 ELSE next_attempt_at_utc END,
      lease_owner = CASE WHEN @is_terminal = 1 THEN NULL ELSE lease_owner END,
      lease_token = CASE WHEN @is_terminal = 1 THEN NULL ELSE lease_token END,
      leased_until_utc = CASE WHEN @is_terminal = 1 THEN NULL ELSE leased_until_utc END
  WHERE work_item_id = @work_item_id AND lease_token = @lease_token;
  INSERT ops.transitions(entity_kind, entity_id, from_state, to_state, actor, reason)
  VALUES('work_item', @work_item_id, @from_state, @to_state, @actor, @reason);
  COMMIT;
END;
GO
