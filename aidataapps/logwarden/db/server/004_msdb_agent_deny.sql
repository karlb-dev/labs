SET XACT_ABORT ON;
USE [msdb];

IF USER_ID(N'lw_agent') IS NULL CREATE USER [lw_agent] FOR LOGIN [lw_agent];
DENY SELECT ON OBJECT::dbo.backupset TO [lw_agent];
DENY SELECT ON OBJECT::dbo.backupmediafamily TO [lw_agent];
