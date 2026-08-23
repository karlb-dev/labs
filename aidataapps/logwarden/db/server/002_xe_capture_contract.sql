SET XACT_ABORT ON;

-- Migration 001 established the foundation session. Replace that pre-data
-- definition once with the capture-specific predicate contract. Subsequent
-- setup runs are no-ops because the attention event proves v2 is installed.
IF EXISTS (SELECT 1 FROM sys.server_event_sessions WHERE name = N'logwarden_capture')
   AND NOT EXISTS
   (
     SELECT 1 FROM sys.server_event_session_events AS e
     INNER JOIN sys.server_event_sessions AS s ON s.event_session_id = e.event_session_id
     WHERE s.name = N'logwarden_capture' AND e.name = N'attention'
   )
BEGIN
  IF EXISTS (SELECT 1 FROM sys.dm_xe_sessions WHERE name = N'logwarden_capture')
    ALTER EVENT SESSION [logwarden_capture] ON SERVER STATE = STOP;
  DROP EVENT SESSION [logwarden_capture] ON SERVER;
END;

IF NOT EXISTS (SELECT 1 FROM sys.server_event_sessions WHERE name = N'logwarden_capture')
BEGIN
  EXEC(N'
    CREATE EVENT SESSION [logwarden_capture] ON SERVER
    ADD EVENT sqlserver.error_reported
    (
      ACTION
      (
        sqlserver.client_app_name, sqlserver.client_hostname,
        sqlserver.database_id, sqlserver.database_name, sqlserver.session_id,
        sqlserver.sql_text, sqlserver.username
      )
      WHERE
      (
        [sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden-Inject%'')
        OR [sqlserver].[like_i_sql_unicode_string]([sqlserver].[database_name],N''LW[_]%'')
      )
    ),
    ADD EVENT sqlserver.attention
    (
      ACTION
      (
        sqlserver.client_app_name, sqlserver.client_hostname,
        sqlserver.database_id, sqlserver.database_name, sqlserver.session_id,
        sqlserver.sql_text, sqlserver.username
      )
      WHERE ([sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden-Inject%''))
    ),
    ADD EVENT sqlserver.database_file_size_change
    (
      ACTION
      (
        sqlserver.client_app_name, sqlserver.client_hostname,
        sqlserver.database_id, sqlserver.database_name, sqlserver.session_id,
        sqlserver.sql_text, sqlserver.username
      )
      WHERE
      (
        [sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden-Inject%'')
        OR [sqlserver].[like_i_sql_unicode_string]([sqlserver].[database_name],N''LW[_]%'')
      )
    ),
    ADD EVENT sqlserver.rpc_completed
    (
      ACTION
      (
        sqlserver.client_app_name, sqlserver.client_hostname,
        sqlserver.database_id, sqlserver.database_name, sqlserver.session_id,
        sqlserver.sql_text, sqlserver.username
      )
      WHERE ([sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden-Inject%''))
    ),
    ADD EVENT sqlserver.sql_batch_completed
    (
      ACTION
      (
        sqlserver.client_app_name, sqlserver.client_hostname,
        sqlserver.database_id, sqlserver.database_name, sqlserver.session_id,
        sqlserver.sql_text, sqlserver.username
      )
      WHERE ([sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden-Inject%''))
    ),
    ADD EVENT sqlserver.xml_deadlock_report,
    ADD EVENT sqlserver.blocked_process_report
    ADD TARGET package0.event_file
    (
      SET filename = N''/var/opt/mssql/log/logwarden_capture.xel'',
          max_file_size = (100), max_rollover_files = (8)
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
