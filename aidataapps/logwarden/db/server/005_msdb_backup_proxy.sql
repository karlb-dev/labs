SET XACT_ABORT ON;
USE [msdb];

EXEC(N'
CREATE OR ALTER PROCEDURE dbo.usp_logwarden_backup_history
  @database_name sysname,
  @max_rows int = 20
WITH EXECUTE AS OWNER
AS
BEGIN
  SET NOCOUNT ON;
  IF @max_rows NOT BETWEEN 1 AND 100 THROW 51723, ''max_rows must be between 1 and 100'', 1;
  IF @database_name<>N''LogWardenWorkload'' AND @database_name NOT LIKE N''LW[_]%''
    THROW 51724, ''database is outside the LogWarden allowlist'', 1;
  SELECT TOP (@max_rows)
    N''msdb'' AS source_kind,backup_start_date AS started_at_utc,
    backup_finish_date AS finished_at_utc,CONVERT(bit,1) AS succeeded,
    type AS backup_type,CONVERT(numeric(20,0),backup_size) AS backup_size,
    CONVERT(numeric(20,0),compressed_backup_size) AS compressed_backup_size,
    CONVERT(int,NULL) AS error_number
  FROM dbo.backupset
  WHERE database_name=@database_name
  ORDER BY backup_start_date DESC,backup_set_id DESC;
END;
');

GRANT EXECUTE ON OBJECT::dbo.usp_logwarden_backup_history TO [lw_agent];
DENY SELECT ON OBJECT::dbo.backupset TO [lw_agent];
DENY SELECT ON OBJECT::dbo.backupmediafamily TO [lw_agent];
REVOKE SELECT ON OBJECT::dbo.backupset FROM [LogWardenToolCertificateUser];
REVOKE SELECT ON OBJECT::dbo.backupmediafamily FROM [LogWardenToolCertificateUser];
