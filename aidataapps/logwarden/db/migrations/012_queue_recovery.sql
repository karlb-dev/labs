SET XACT_ABORT ON;
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

  DECLARE @claimed TABLE(work_item_id bigint NOT NULL, from_state varchar(24) NOT NULL);
  BEGIN TRANSACTION;
  ;WITH next_item AS
  (
    SELECT TOP (1) *
    FROM ops.work_items WITH (UPDLOCK, READPAST, ROWLOCK, INDEX(ix_work_items_claim))
    WHERE next_attempt_at_utc <= SYSUTCDATETIME()
      AND
      (
        status IN ('pending','retryable_failure','lease_expired')
        OR
        (
          status IN ('leased','packet_loaded','model_requested','tool_requested','tool_completed',
                     'decision_received','validated','persisted','proposed_action_recorded')
          AND leased_until_utc < SYSUTCDATETIME()
        )
      )
    ORDER BY priority DESC, work_item_id ASC
  )
  UPDATE next_item
  SET status = 'leased',
      lease_owner = @worker_id,
      lease_token = @lease_token,
      leased_until_utc = DATEADD(SECOND, @lease_seconds, SYSUTCDATETIME()),
      attempt_count = attempt_count + 1
  OUTPUT INSERTED.work_item_id, DELETED.status INTO @claimed(work_item_id, from_state);

  INSERT ops.transitions(entity_kind, entity_id, from_state, to_state, actor, reason)
  SELECT 'work_item', work_item_id, from_state, 'leased', @worker_id,
         CASE WHEN from_state = 'pending' THEN 'initial queue claim' ELSE 'retry or expired-lease recovery' END
  FROM @claimed;

  SELECT item.*
  FROM ops.work_items AS item
  INNER JOIN @claimed AS claimed ON claimed.work_item_id = item.work_item_id;
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
  IF @lease_seconds NOT BETWEEN 30 AND 1800 THROW 51000, 'Invalid lease duration', 1;
  UPDATE ops.work_items
  SET leased_until_utc = DATEADD(SECOND, @lease_seconds, SYSUTCDATETIME())
  WHERE work_item_id = @work_item_id
    AND status IN ('leased','packet_loaded','model_requested','tool_requested','tool_completed',
                   'decision_received','validated','persisted','proposed_action_recorded')
    AND lease_token = @lease_token
    AND leased_until_utc >= SYSUTCDATETIME();
  IF @@ROWCOUNT <> 1 THROW 51001, 'Stale, expired, or missing lease token', 1;
END;
GO

CREATE OR ALTER PROCEDURE ops.usp_complete_work_item
  @work_item_id bigint,
  @lease_token uniqueidentifier
AS
BEGIN
  SET NOCOUNT ON;
  SET XACT_ABORT ON;
  DECLARE @from_state varchar(24);
  BEGIN TRANSACTION;
  SELECT @from_state = status
  FROM ops.work_items WITH (UPDLOCK, ROWLOCK)
  WHERE work_item_id = @work_item_id
    AND lease_token = @lease_token
    AND leased_until_utc >= SYSUTCDATETIME();
  IF @from_state IS NULL OR @from_state NOT IN ('leased','persisted','proposed_action_recorded')
  BEGIN
    ROLLBACK;
    THROW 51002, 'Stale token or invalid terminal transition', 1;
  END;
  UPDATE ops.work_items
  SET status = 'complete', completed_at_utc = SYSUTCDATETIME(),
      lease_owner = NULL, lease_token = NULL, leased_until_utc = NULL
  WHERE work_item_id = @work_item_id;
  INSERT ops.transitions(entity_kind, entity_id, from_state, to_state, actor, reason)
  VALUES('work_item', @work_item_id, @from_state, 'complete', ORIGINAL_LOGIN(), 'lease-token completion');
  COMMIT;
END;
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
      ('model_requested','contract_rejected'),
      ('decision_received','contract_rejected'),
      ('decision_received','policy_rejected'),
      ('leased','retryable_failure'),
      ('packet_loaded','retryable_failure'),
      ('model_requested','retryable_failure'),
      ('tool_requested','retryable_failure'),
      ('tool_completed','retryable_failure'),
      ('decision_received','retryable_failure'),
      ('validated','retryable_failure')
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
