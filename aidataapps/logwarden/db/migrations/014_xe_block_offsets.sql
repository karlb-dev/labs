SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE ingest.usp_ingest_xe
  @run_id varchar(120),
  @worker_id varchar(120),
  @target_pattern nvarchar(500) = N'/var/opt/mssql/log/logwarden_capture*.xel'
AS
BEGIN
  SET NOCOUNT ON;
  SET XACT_ABORT ON;
  DECLARE @source_id varchar(80) = 'xe-logwarden-capture';
  DECLARE @initial_file_name nvarchar(500), @initial_offset bigint;
  DECLARE @batch_id bigint, @rows_read int, @rows_inserted int;
  DECLARE @last_file_name nvarchar(500), @last_offset bigint, @last_time datetime2(7);

  BEGIN TRANSACTION;
  DECLARE @lock_result int;
  EXEC @lock_result = sys.sp_getapplock
    @Resource = N'LogWarden:ingest:xe-logwarden-capture',
    @LockMode = N'Exclusive', @LockOwner = N'Transaction', @LockTimeout = 0;
  IF @lock_result < 0
  BEGIN
    ROLLBACK;
    THROW 51400, 'XE ingester is already active', 1;
  END;
  IF NOT EXISTS (SELECT 1 FROM ingest.sources WHERE source_id = @source_id)
  BEGIN
    ROLLBACK;
    THROW 51401, 'XE source metadata is missing; run db:setup', 1;
  END;

  SELECT @initial_file_name = file_name, @initial_offset = file_offset
  FROM ingest.source_cursors WITH (UPDLOCK, HOLDLOCK)
  WHERE source_id = @source_id;

  INSERT ingest.ingestion_batches(source_id, worker_id, start_cursor_json, status)
  VALUES(@source_id, @worker_id,
    (SELECT @initial_file_name AS file_name, @initial_offset AS file_offset FOR JSON PATH, WITHOUT_ARRAY_WRAPPER, INCLUDE_NULL_VALUES),
    'running');
  SET @batch_id = SCOPE_IDENTITY();

  CREATE TABLE #xe
  (
    file_name nvarchar(500) NOT NULL,
    file_offset bigint NOT NULL,
    block_ordinal int NOT NULL,
    source_timestamp_utc datetime2(7) NOT NULL,
    event_data xml NOT NULL,
    source_position_key char(64) NOT NULL PRIMARY KEY
  );

  ;WITH source_rows AS
  (
    SELECT file_name, file_offset, timestamp_utc, CONVERT(xml, event_data) AS event_data,
           ROW_NUMBER() OVER
           (
             PARTITION BY file_name, file_offset
             ORDER BY timestamp_utc, CONVERT(nvarchar(max), event_data)
           ) AS block_ordinal
    FROM sys.fn_xe_file_target_read_file(@target_pattern, NULL, @initial_file_name, @initial_offset)
  )
  INSERT #xe(file_name, file_offset, block_ordinal, source_timestamp_utc, event_data, source_position_key)
  SELECT file_name, file_offset, block_ordinal, timestamp_utc, event_data,
         LOWER(CONVERT(char(64), HASHBYTES('SHA2_256',
           CONCAT(file_name, N':', file_offset, N':', block_ordinal)), 2))
  FROM source_rows;
  SET @rows_read = @@ROWCOUNT;

  DECLARE @inserted TABLE(raw_event_id bigint NOT NULL, source_position_key char(64) NOT NULL PRIMARY KEY);
  INSERT ingest.raw_events
  (
    source_id, ingestion_batch_id, source_position_key, source_file_name,
    source_file_offset, source_timestamp_utc, captured_at_utc, raw_event_name,
    raw_payload_xml, raw_sha256, parse_status
  )
  OUTPUT INSERTED.raw_event_id, INSERTED.source_position_key
    INTO @inserted(raw_event_id, source_position_key)
  SELECT @source_id, @batch_id, x.source_position_key, x.file_name,
         x.file_offset, x.source_timestamp_utc, SYSUTCDATETIME(),
         x.event_data.value('(/event/@name)[1]', 'varchar(120)'), x.event_data,
         LOWER(CONVERT(char(64), HASHBYTES('SHA2_256', CONVERT(nvarchar(max), x.event_data)), 2)),
         'parsed'
  FROM #xe AS x
  WHERE NOT EXISTS
    (SELECT 1 FROM ingest.raw_events AS prior WHERE prior.source_position_key = x.source_position_key);
  SET @rows_inserted = @@ROWCOUNT;

  INSERT ingest.canonical_events
  (
    raw_event_id, source_kind, source_event_name, occurred_at_utc,
    captured_at_utc, database_name, session_id, error_number, severity,
    state, message_raw, message_normalized, sql_text_sha256,
    client_app_name, login_name, duration_ms, cpu_ms, logical_reads, writes,
    activity_id, parser_version, normalization_version, canonical_json,
    normalized_sha256
  )
  SELECT i.raw_event_id, 'xe', p.event_name, x.source_timestamp_utc,
         SYSUTCDATETIME(), p.database_name, p.session_id, p.error_number,
         p.severity, p.error_state, p.message_raw,
         LOWER(LTRIM(RTRIM(COALESCE(p.message_raw, p.event_name)))),
         CASE WHEN p.sql_text IS NULL THEN NULL ELSE LOWER(CONVERT(char(64), HASHBYTES('SHA2_256', p.sql_text), 2)) END,
         p.client_app_name, p.login_name,
         CASE WHEN p.duration_microseconds IS NULL THEN NULL ELSE CONVERT(decimal(18,3), p.duration_microseconds / 1000.0) END,
         CASE WHEN p.cpu_microseconds IS NULL THEN NULL ELSE CONVERT(decimal(18,3), p.cpu_microseconds / 1000.0) END,
         p.logical_reads, p.writes, p.activity_id,
         'xe-xml-v1', 'minimal-v1',
         (SELECT 1 AS schemaVersion, p.event_name AS eventName,
                 x.source_timestamp_utc AS occurredAtUtc, x.file_name AS sourceFileName,
                 x.file_offset AS sourceFileOffset, x.block_ordinal AS blockOrdinal,
                 p.database_name AS databaseName, p.session_id AS sessionId,
                 p.error_number AS errorNumber, p.severity AS severity,
                 p.error_state AS state, p.message_raw AS [message],
                 p.client_app_name AS clientAppName, p.login_name AS loginName,
                 p.activity_id AS activityId
          FOR JSON PATH, WITHOUT_ARRAY_WRAPPER, INCLUDE_NULL_VALUES),
         LOWER(CONVERT(char(64),
           HASHBYTES('SHA2_256', CONCAT_WS(N'|', p.event_name,
             p.error_number, p.severity, p.error_state,
             LOWER(LTRIM(RTRIM(COALESCE(p.message_raw, N'')))))),
           2))
  FROM @inserted AS i
  INNER JOIN #xe AS x ON x.source_position_key = i.source_position_key
  CROSS APPLY
  (
    SELECT
      x.event_data.value('(/event/@name)[1]', 'varchar(120)') AS event_name,
      x.event_data.value('(/event/data[@name="error_number"]/value/text())[1]', 'int') AS error_number,
      x.event_data.value('(/event/data[@name="severity"]/value/text())[1]', 'int') AS severity,
      x.event_data.value('(/event/data[@name="state"]/value/text())[1]', 'int') AS error_state,
      x.event_data.value('(/event/data[@name="message"]/value/text())[1]', 'nvarchar(4000)') AS message_raw,
      x.event_data.value('(/event/data[@name="duration"]/value/text())[1]', 'bigint') AS duration_microseconds,
      x.event_data.value('(/event/data[@name="cpu_time"]/value/text())[1]', 'bigint') AS cpu_microseconds,
      x.event_data.value('(/event/data[@name="logical_reads"]/value/text())[1]', 'bigint') AS logical_reads,
      x.event_data.value('(/event/data[@name="writes"]/value/text())[1]', 'bigint') AS writes,
      x.event_data.value('(/event/action[@name="database_name"]/value/text())[1]', 'nvarchar(128)') AS database_name,
      x.event_data.value('(/event/action[@name="session_id"]/value/text())[1]', 'int') AS session_id,
      x.event_data.value('(/event/action[@name="client_app_name"]/value/text())[1]', 'nvarchar(256)') AS client_app_name,
      x.event_data.value('(/event/action[@name="username"]/value/text())[1]', 'nvarchar(256)') AS login_name,
      x.event_data.value('(/event/action[@name="attach_activity_id"]/value/text())[1]', 'varchar(100)') AS activity_id,
      x.event_data.value('(/event/action[@name="sql_text"]/value/text())[1]', 'nvarchar(4000)') AS sql_text
  ) AS p;

  INSERT ingest.event_fingerprints
    (canonical_event_id, fingerprint_version, event_fingerprint,
     message_template_hash, identifier_masked_text, error_number_masked_text)
  SELECT c.canonical_event_id, 'fingerprint-v1',
         LOWER(CONVERT(char(64), HASHBYTES('SHA2_256', CONCAT_WS(N'|', c.source_event_name, c.error_number, c.message_normalized)), 2)),
         LOWER(CONVERT(char(64), HASHBYTES('SHA2_256', COALESCE(c.message_normalized, N'')), 2)),
         COALESCE(c.message_normalized, N''), COALESCE(c.message_normalized, N'')
  FROM ingest.canonical_events AS c
  INNER JOIN @inserted AS i ON i.raw_event_id = c.raw_event_id;

  SELECT TOP (1) @last_file_name = file_name, @last_offset = file_offset,
                 @last_time = source_timestamp_utc
  FROM #xe ORDER BY source_timestamp_utc DESC, file_name DESC, file_offset DESC, block_ordinal DESC;

  IF @last_file_name IS NOT NULL
  BEGIN
    UPDATE ingest.source_cursors
    SET file_name = @last_file_name, file_offset = @last_offset,
        watermark_utc = @last_time,
        cursor_json = (SELECT @last_file_name AS file_name, @last_offset AS file_offset,
                              @last_time AS watermark_utc
                       FOR JSON PATH, WITHOUT_ARRAY_WRAPPER),
        updated_at_utc = SYSUTCDATETIME()
    WHERE source_id = @source_id;
  END;

  UPDATE ingest.ingestion_batches
  SET end_cursor_json = COALESCE((SELECT cursor_json FROM ingest.source_cursors WHERE source_id=@source_id), N'{}'),
      rows_read = @rows_read, rows_inserted = @rows_inserted,
      rows_duplicate = @rows_read - @rows_inserted, rows_failed = 0,
      committed_at_utc = SYSUTCDATETIME(), status = 'complete'
  WHERE ingestion_batch_id = @batch_id;
  COMMIT;

  SELECT @batch_id AS ingestion_batch_id, @rows_read AS rows_read,
         @rows_inserted AS rows_inserted, @rows_read - @rows_inserted AS rows_duplicate,
         @last_file_name AS file_name, @last_offset AS file_offset, @last_time AS watermark_utc;
END;
