SET XACT_ABORT ON;

IF NOT EXISTS (SELECT 1 FROM sys.server_event_sessions WHERE name = N'logwarden_capture')
BEGIN
  EXEC(N'
    CREATE EVENT SESSION [logwarden_capture] ON SERVER
    ADD EVENT sqlserver.error_reported
    (
      ACTION
      (
        sqlserver.client_app_name,
        sqlserver.client_hostname,
        sqlserver.database_name,
        sqlserver.session_id,
        sqlserver.sql_text,
        sqlserver.username
      )
      WHERE ([sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden%''))
    ),
    ADD EVENT sqlserver.rpc_completed
    (
      ACTION
      (
        sqlserver.client_app_name,
        sqlserver.client_hostname,
        sqlserver.database_name,
        sqlserver.session_id,
        sqlserver.sql_text,
        sqlserver.username
      )
      WHERE ([sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden%''))
    ),
    ADD EVENT sqlserver.sql_batch_completed
    (
      ACTION
      (
        sqlserver.client_app_name,
        sqlserver.client_hostname,
        sqlserver.database_name,
        sqlserver.session_id,
        sqlserver.sql_text,
        sqlserver.username
      )
      WHERE ([sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden%''))
    ),
    ADD EVENT sqlserver.xml_deadlock_report,
    ADD EVENT sqlserver.blocked_process_report
    ADD TARGET package0.event_file
    (
      SET filename = N''/var/opt/mssql/log/logwarden_capture.xel'',
          max_file_size = (100),
          max_rollover_files = (8)
    )
    WITH
    (
      MAX_MEMORY = 8192 KB,
      EVENT_RETENTION_MODE = ALLOW_SINGLE_EVENT_LOSS,
      MAX_DISPATCH_LATENCY = 2 SECONDS,
      MAX_EVENT_SIZE = 0 KB,
      MEMORY_PARTITION_MODE = NONE,
      TRACK_CAUSALITY = ON,
      STARTUP_STATE = ON
    );
  ');
END;

IF NOT EXISTS (SELECT 1 FROM sys.dm_xe_sessions WHERE name = N'logwarden_capture')
  ALTER EVENT SESSION [logwarden_capture] ON SERVER STATE = START;
