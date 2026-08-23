SET XACT_ABORT ON;
GO

-- A pooled connection retains its prior SET TRANSACTION ISOLATION LEVEL after
-- a transaction commits. Normalize the queue claim itself so evidence writers
-- that use SERIALIZABLE cannot make a later READPAST claim invalid.
CREATE OR ALTER PROCEDURE ops.usp_claim_work_item
  @worker_id varchar(120),
  @lease_token uniqueidentifier,
  @lease_seconds int = 240
AS
BEGIN
  SET NOCOUNT ON;
  SET XACT_ABORT ON;
  SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
  IF @lease_seconds NOT BETWEEN 30 AND 1800 THROW 51000, 'Invalid lease duration', 1;

  DECLARE @claimed TABLE(work_item_id bigint NOT NULL, from_state varchar(24) NOT NULL);
  BEGIN TRANSACTION;
  ;WITH next_item AS
  (
    SELECT TOP (1) *
    FROM ops.work_items WITH (UPDLOCK, READPAST, READCOMMITTEDLOCK, ROWLOCK, INDEX(ix_work_items_claim))
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
