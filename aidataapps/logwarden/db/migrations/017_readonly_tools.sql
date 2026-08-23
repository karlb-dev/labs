SET XACT_ABORT ON;

IF CERT_ID(N'LogWardenToolCertificate') IS NULL
  CREATE CERTIFICATE [LogWardenToolCertificate]
    WITH SUBJECT = N'LogWarden narrowly scoped read-only server diagnostics';
GO

CREATE OR ALTER PROCEDURE ops.usp_get_recent_incident_counts
  @incident_class varchar(80) = NULL,
  @window_minutes int = 60
AS
BEGIN
  SET NOCOUNT ON;
  IF @window_minutes NOT BETWEEN 1 AND 10080 THROW 51700, 'window_minutes must be between 1 and 10080', 1;
  SELECT
    COALESCE(@incident_class,'all') AS requested_class,
    @window_minutes AS window_minutes,
    COUNT_BIG(*) AS incident_count,
    COALESCE(SUM(CONVERT(bigint,event_count)),0) AS event_count,
    COUNT_BIG(DISTINCT correlation_key) AS correlation_count,
    MAX(last_event_at_utc) AS newest_incident_at_utc
  FROM ops.incidents
  WHERE last_event_at_utc >= DATEADD(MINUTE,-@window_minutes,SYSUTCDATETIME())
    AND (@incident_class IS NULL OR current_class=@incident_class);
END;
GO

CREATE OR ALTER PROCEDURE kb.usp_search_runbooks
  @query nvarchar(1000),
  @top_k int = 5,
  @corpus_id varchar(80) = 'primary-v1'
AS
BEGIN
  SET NOCOUNT ON;
  IF NULLIF(LTRIM(RTRIM(@query)),N'') IS NULL THROW 51701, 'query must not be empty', 1;
  IF @top_k NOT BETWEEN 1 AND 20 THROW 51702, 'top_k must be between 1 and 20', 1;
  IF NOT EXISTS (SELECT 1 FROM kb.search_corpora WHERE corpus_id=@corpus_id AND corpus_kind='primary')
    THROW 51703, 'unknown or non-primary corpus', 1;
  DECLARE @candidate_k int = @top_k * 4;
  SELECT TOP (@top_k)
    chunk.chunk_id,chunk.runbook_id,chunk.heading_path,chunk.content,
    hit.[RANK] AS lexical_score,
    ROW_NUMBER() OVER (ORDER BY hit.[RANK] DESC,chunk.chunk_id) AS rank_ordinal
  FROM FREETEXTTABLE(kb.runbook_chunks,content,@query,LANGUAGE 1033,@candidate_k) AS hit
  INNER JOIN kb.runbook_chunks AS chunk ON chunk.chunk_id=hit.[KEY]
  INNER JOIN kb.runbooks AS book ON book.runbook_id=chunk.runbook_id
  WHERE chunk.corpus_id=@corpus_id AND book.enabled=1
  ORDER BY hit.[RANK] DESC,chunk.chunk_id;
END;
GO

CREATE OR ALTER PROCEDURE agent.usp_tool_get_blocking_snapshot
  @database_name sysname,
  @max_rows int = 20
AS
BEGIN
  SET NOCOUNT ON;
  IF @max_rows NOT BETWEEN 1 AND 100 THROW 51704, 'max_rows must be between 1 and 100', 1;
  DECLARE @database_id int=DB_ID(@database_name);
  IF @database_id IS NULL THROW 51705, 'database does not exist', 1;
  IF @database_name<>N'LogWardenWorkload' AND NOT EXISTS
    (SELECT 1 FROM workload.disposable_databases WHERE database_name=@database_name AND dropped_at_utc IS NULL)
    THROW 51706, 'database is outside the LogWarden allowlist', 1;
  SELECT TOP (@max_rows)
    request.session_id,request.blocking_session_id,DB_NAME(request.database_id) AS database_name,
    request.status,request.command,request.wait_type,request.wait_time AS wait_time_ms,
    request.wait_resource,request.open_transaction_count,
    CONVERT(varchar(34),request.start_time,127) AS request_started_at_utc
  FROM sys.dm_exec_requests AS request
  WHERE request.blocking_session_id<>0 AND request.database_id=@database_id
  ORDER BY request.wait_time DESC,request.session_id;
END;
GO

CREATE OR ALTER PROCEDURE agent.usp_tool_get_log_space
  @database_name sysname
AS
BEGIN
  SET NOCOUNT ON;
  DECLARE @database_id int=DB_ID(@database_name);
  IF @database_id IS NULL THROW 51707, 'database does not exist', 1;
  IF @database_name<>N'LogWardenWorkload' AND NOT EXISTS
    (SELECT 1 FROM workload.disposable_databases WHERE database_name=@database_name AND dropped_at_utc IS NULL)
    THROW 51708, 'database is outside the LogWarden allowlist', 1;
  SELECT @database_name AS database_name,
    stats.total_log_size_mb,stats.active_log_size_mb,
    CONVERT(decimal(9,4),CASE WHEN stats.total_log_size_mb=0 THEN NULL ELSE 100.0*stats.active_log_size_mb/stats.total_log_size_mb END) AS active_log_percent,
    stats.log_truncation_holdup_reason,stats.log_backup_time,
    stats.log_since_last_log_backup_mb,stats.log_checkpoint_lsn
  FROM sys.dm_db_log_stats(@database_id) AS stats;
END;
GO

CREATE OR ALTER PROCEDURE agent.usp_tool_get_active_transactions
  @database_name sysname,
  @max_rows int = 20
AS
BEGIN
  SET NOCOUNT ON;
  IF @max_rows NOT BETWEEN 1 AND 100 THROW 51709, 'max_rows must be between 1 and 100', 1;
  DECLARE @database_id int=DB_ID(@database_name);
  IF @database_id IS NULL THROW 51710, 'database does not exist', 1;
  IF @database_name<>N'LogWardenWorkload' AND NOT EXISTS
    (SELECT 1 FROM workload.disposable_databases WHERE database_name=@database_name AND dropped_at_utc IS NULL)
    THROW 51711, 'database is outside the LogWarden allowlist', 1;
  SELECT TOP (@max_rows)
    txn.transaction_id,session_transaction.session_id,@database_name AS database_name,
    txn.database_transaction_type,txn.database_transaction_state,
    txn.database_transaction_log_record_count,
    txn.database_transaction_log_bytes_used,
    txn.database_transaction_log_bytes_reserved
  FROM sys.dm_tran_database_transactions AS txn
  LEFT JOIN sys.dm_tran_session_transactions AS session_transaction
    ON session_transaction.transaction_id=txn.transaction_id
  WHERE txn.database_id=@database_id
  ORDER BY txn.database_transaction_log_bytes_used DESC,txn.transaction_id;
END;
GO

CREATE OR ALTER PROCEDURE agent.usp_tool_get_backup_history
  @database_name sysname,
  @max_rows int = 20
AS
BEGIN
  SET NOCOUNT ON;
  IF @max_rows NOT BETWEEN 1 AND 100 THROW 51712, 'max_rows must be between 1 and 100', 1;
  IF @database_name<>N'LogWardenWorkload' AND NOT EXISTS
    (SELECT 1 FROM workload.disposable_databases WHERE database_name=@database_name)
    THROW 51713, 'database is outside the LogWarden allowlist', 1;
  ;WITH history AS
  (
    SELECT N'msdb' AS source_kind,backup_start_date AS started_at_utc,
      backup_finish_date AS finished_at_utc,CONVERT(bit,1) AS succeeded,
      type AS backup_type,backup_size,compressed_backup_size,
      CONVERT(int,NULL) AS error_number
    FROM msdb.dbo.backupset WHERE database_name=@database_name
    UNION ALL
    SELECT N'lab_attempt',started_at_utc,finished_at_utc,succeeded,
      'D',CONVERT(numeric(20,0),NULL),CONVERT(numeric(20,0),NULL),error_number
    FROM workload.backup_attempts WHERE database_name=@database_name
  )
  SELECT TOP (@max_rows) * FROM history
  ORDER BY started_at_utc DESC,source_kind;
END;
GO

CREATE OR ALTER PROCEDURE agent.usp_tool_get_deadlock_graph
  @max_rows int = 5
AS
BEGIN
  SET NOCOUNT ON;
  IF @max_rows NOT BETWEEN 1 AND 20 THROW 51714, 'max_rows must be between 1 and 20', 1;
  ;WITH source AS
  (
    SELECT timestamp_utc,CONVERT(xml,event_data) AS event_xml
    FROM sys.fn_xe_file_target_read_file(N'/var/opt/mssql/log/logwarden_capture*.xel',NULL,NULL,NULL)
    WHERE object_name=N'xml_deadlock_report'
  )
  SELECT TOP (@max_rows)
    timestamp_utc AS occurred_at_utc,
    event_xml.value('count((/event/data/value/deadlock/victim-list/victimProcess))','int') AS victim_count,
    event_xml.value('count((/event/data/value/deadlock/process-list/process))','int') AS process_count,
    LOWER(CONVERT(char(64),HASHBYTES('SHA2_256',CONVERT(nvarchar(max),event_xml)),2)) AS graph_sha256
  FROM source ORDER BY timestamp_utc DESC,graph_sha256;
END;
GO

CREATE OR ALTER PROCEDURE agent.usp_tool_resolve_context_snapshot
  @episode_id varchar(120),
  @tool_id varchar(80),
  @canonical_args_sha256 char(64)
AS
BEGIN
  SET NOCOUNT ON;
  IF LEN(@canonical_args_sha256)<>64 OR @canonical_args_sha256 COLLATE Latin1_General_100_BIN2 LIKE '%[^0-9a-f]%'
    THROW 51715, 'canonical_args_sha256 must be lowercase SHA-256 hex', 1;
  IF EXISTS
    (SELECT 1 FROM ingest.context_snapshots WHERE episode_id=@episode_id AND tool_id=@tool_id AND canonical_args_sha256=@canonical_args_sha256)
  BEGIN
    SELECT 'ok' AS status,CONVERT(nvarchar(200),NULL) AS reason,CONVERT(bit,0) AS snapshot_miss,
      result_json,result_sha256,row_count,procedure_version,captured_at_utc
    FROM ingest.context_snapshots
    WHERE episode_id=@episode_id AND tool_id=@tool_id AND canonical_args_sha256=@canonical_args_sha256;
    RETURN;
  END;
  DECLARE @miss nvarchar(max)=N'{"reason":"no snapshot for arguments","status":"no_data"}';
  SELECT 'no_data' AS status,N'no snapshot for arguments' AS reason,CONVERT(bit,1) AS snapshot_miss,
    @miss AS result_json,
    LOWER(CONVERT(char(64),HASHBYTES('SHA2_256',CONVERT(varbinary(max),@miss)),2)) AS result_sha256,
    0 AS row_count,'snapshot-resolver-v1' AS procedure_version,CONVERT(datetime2(7),NULL) AS captured_at_utc;
END;
GO

GRANT EXECUTE ON OBJECT::ops.usp_get_recent_incident_counts TO [lw_agent_role];
GRANT EXECUTE ON OBJECT::kb.usp_search_runbooks TO [lw_agent_role];
GRANT EXECUTE ON OBJECT::agent.usp_tool_get_blocking_snapshot TO [lw_agent_role];
GRANT EXECUTE ON OBJECT::agent.usp_tool_get_log_space TO [lw_agent_role];
GRANT EXECUTE ON OBJECT::agent.usp_tool_get_active_transactions TO [lw_agent_role];
GRANT EXECUTE ON OBJECT::agent.usp_tool_get_backup_history TO [lw_agent_role];
GRANT EXECUTE ON OBJECT::agent.usp_tool_get_deadlock_graph TO [lw_agent_role];
GRANT EXECUTE ON OBJECT::agent.usp_tool_resolve_context_snapshot TO [lw_agent_role];
GO

IF NOT EXISTS
(
  SELECT 1 FROM sys.crypt_properties
  WHERE class_desc='OBJECT_OR_COLUMN' AND major_id=OBJECT_ID(N'agent.usp_tool_get_blocking_snapshot')
    AND thumbprint=(SELECT thumbprint FROM sys.certificates WHERE name=N'LogWardenToolCertificate')
)
  ADD SIGNATURE TO OBJECT::agent.usp_tool_get_blocking_snapshot BY CERTIFICATE [LogWardenToolCertificate];
IF NOT EXISTS
(
  SELECT 1 FROM sys.crypt_properties
  WHERE class_desc='OBJECT_OR_COLUMN' AND major_id=OBJECT_ID(N'agent.usp_tool_get_log_space')
    AND thumbprint=(SELECT thumbprint FROM sys.certificates WHERE name=N'LogWardenToolCertificate')
)
  ADD SIGNATURE TO OBJECT::agent.usp_tool_get_log_space BY CERTIFICATE [LogWardenToolCertificate];
IF NOT EXISTS
(
  SELECT 1 FROM sys.crypt_properties
  WHERE class_desc='OBJECT_OR_COLUMN' AND major_id=OBJECT_ID(N'agent.usp_tool_get_active_transactions')
    AND thumbprint=(SELECT thumbprint FROM sys.certificates WHERE name=N'LogWardenToolCertificate')
)
  ADD SIGNATURE TO OBJECT::agent.usp_tool_get_active_transactions BY CERTIFICATE [LogWardenToolCertificate];
IF NOT EXISTS
(
  SELECT 1 FROM sys.crypt_properties
  WHERE class_desc='OBJECT_OR_COLUMN' AND major_id=OBJECT_ID(N'agent.usp_tool_get_backup_history')
    AND thumbprint=(SELECT thumbprint FROM sys.certificates WHERE name=N'LogWardenToolCertificate')
)
  ADD SIGNATURE TO OBJECT::agent.usp_tool_get_backup_history BY CERTIFICATE [LogWardenToolCertificate];
IF NOT EXISTS
(
  SELECT 1 FROM sys.crypt_properties
  WHERE class_desc='OBJECT_OR_COLUMN' AND major_id=OBJECT_ID(N'agent.usp_tool_get_deadlock_graph')
    AND thumbprint=(SELECT thumbprint FROM sys.certificates WHERE name=N'LogWardenToolCertificate')
)
  ADD SIGNATURE TO OBJECT::agent.usp_tool_get_deadlock_graph BY CERTIFICATE [LogWardenToolCertificate];
