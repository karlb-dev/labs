SET XACT_ABORT ON;
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
  CREATE TABLE #msdb_history
  (
    source_kind nvarchar(20) NOT NULL,
    started_at_utc datetime NULL,
    finished_at_utc datetime NULL,
    succeeded bit NOT NULL,
    backup_type char(1) NULL,
    backup_size numeric(20,0) NULL,
    compressed_backup_size numeric(20,0) NULL,
    error_number int NULL
  );
  INSERT #msdb_history
  EXEC sys.sp_executesql
    N'EXEC msdb.dbo.usp_logwarden_backup_history @database_name,@max_rows',
    N'@database_name sysname,@max_rows int',@database_name,@max_rows;
  ;WITH history AS
  (
    SELECT * FROM #msdb_history
    UNION ALL
    SELECT N'lab_attempt',started_at_utc,finished_at_utc,succeeded,
      'D',CONVERT(numeric(20,0),NULL),CONVERT(numeric(20,0),NULL),error_number
    FROM workload.backup_attempts WHERE database_name=@database_name
  )
  SELECT TOP (@max_rows) * FROM history
  ORDER BY started_at_utc DESC,source_kind;
END;
