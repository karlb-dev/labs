SET XACT_ABORT ON;

-- Governing addendum S-1 contract. Recreate only when one of the binding
-- retention/dispatch/target settings is absent or different.
IF EXISTS (SELECT 1 FROM sys.server_event_sessions WHERE name=N'logwarden_capture')
   AND NOT EXISTS
   (
     SELECT 1
     FROM sys.server_event_sessions AS session
     WHERE session.name=N'logwarden_capture'
       AND session.event_retention_mode_desc=N'ALLOW_SINGLE_EVENT_LOSS'
       AND session.max_dispatch_latency=1000
       AND session.startup_state=1
       AND EXISTS
       (
         SELECT 1 FROM sys.server_event_session_targets AS target
         INNER JOIN sys.server_event_session_fields AS size
           ON size.event_session_id=target.event_session_id
          AND size.object_id=target.target_id
          AND size.name=N'max_file_size' AND TRY_CONVERT(int,size.value)=16
         INNER JOIN sys.server_event_session_fields AS rollover
           ON rollover.event_session_id=target.event_session_id
          AND rollover.object_id=target.target_id
          AND rollover.name=N'max_rollover_files' AND TRY_CONVERT(int,rollover.value)=20
         WHERE target.event_session_id=session.event_session_id AND target.name=N'event_file'
       )
   )
BEGIN
  IF EXISTS (SELECT 1 FROM sys.dm_xe_sessions WHERE name=N'logwarden_capture')
    ALTER EVENT SESSION [logwarden_capture] ON SERVER STATE=STOP;
  DROP EVENT SESSION [logwarden_capture] ON SERVER;
END;

IF NOT EXISTS (SELECT 1 FROM sys.server_event_sessions WHERE name=N'logwarden_capture')
BEGIN
  EXEC(N'
    CREATE EVENT SESSION [logwarden_capture] ON SERVER
    ADD EVENT sqlserver.error_reported
    (
      ACTION
      (
        sqlserver.client_app_name,sqlserver.client_hostname,
        sqlserver.database_id,sqlserver.database_name,sqlserver.session_id,
        sqlserver.sql_text,sqlserver.username
      )
      WHERE
      (
        (
          [sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden-Inject%'')
          OR [sqlserver].[like_i_sql_unicode_string]([sqlserver].[database_name],N''LW[_]%'')
        )
        AND
        (
          [severity]>=(11)
          OR [error_number]=(1205) OR [error_number]=(18456) OR [error_number]=(9002)
          OR [error_number]=(245) OR [error_number]=(208) OR [error_number]=(2601)
          OR [error_number]=(2627) OR [error_number]=(2628) OR [error_number]=(8152)
          OR [error_number]=(8134) OR [error_number]=(3201) OR [error_number]=(3013)
          OR [error_number]=(4064) OR [error_number]=(50000)
        )
      )
    ),
    ADD EVENT sqlserver.attention
    (
      ACTION
      (
        sqlserver.client_app_name,sqlserver.client_hostname,
        sqlserver.database_id,sqlserver.database_name,sqlserver.session_id,
        sqlserver.sql_text,sqlserver.username
      )
      WHERE ([sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden-Inject%''))
    ),
    ADD EVENT sqlserver.database_file_size_change
    (
      ACTION
      (
        sqlserver.client_app_name,sqlserver.client_hostname,
        sqlserver.database_id,sqlserver.database_name,sqlserver.session_id,
        sqlserver.sql_text,sqlserver.username
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
        sqlserver.client_app_name,sqlserver.client_hostname,
        sqlserver.database_id,sqlserver.database_name,sqlserver.session_id,
        sqlserver.sql_text,sqlserver.username
      )
      WHERE ([sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden-Inject%''))
    ),
    ADD EVENT sqlserver.sql_batch_completed
    (
      ACTION
      (
        sqlserver.client_app_name,sqlserver.client_hostname,
        sqlserver.database_id,sqlserver.database_name,sqlserver.session_id,
        sqlserver.sql_text,sqlserver.username
      )
      WHERE ([sqlserver].[like_i_sql_unicode_string]([sqlserver].[client_app_name],N''LogWarden-Inject%''))
    ),
    ADD EVENT sqlserver.xml_deadlock_report,
    ADD EVENT sqlserver.blocked_process_report
    ADD TARGET package0.event_file
    (
      SET filename=N''/var/opt/mssql/log/logwarden_capture.xel'',
          max_file_size=(16),max_rollover_files=(20)
    )
    WITH
    (
      MAX_MEMORY=8192 KB,
      -- SQL Server 17.0.4075.5 rejects error_reported in a NO_EVENT_LOSS
      -- session (25643); this is the strongest supported retention mode.
      EVENT_RETENTION_MODE=ALLOW_SINGLE_EVENT_LOSS,
      MAX_DISPATCH_LATENCY=1 SECONDS,
      MAX_EVENT_SIZE=0 KB,
      MEMORY_PARTITION_MODE=NONE,
      TRACK_CAUSALITY=ON,
      STARTUP_STATE=ON
    );
  ');
END;

IF NOT EXISTS (SELECT 1 FROM sys.dm_xe_sessions WHERE name=N'logwarden_capture')
  ALTER EVENT SESSION [logwarden_capture] ON SERVER STATE=START;
