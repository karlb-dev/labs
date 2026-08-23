SET XACT_ABORT ON;
USE [msdb];

-- Only the certificate token added by the registry-listed control procedure
-- may invoke the internal proxy. The runtime login has no direct entry point.
GRANT EXECUTE ON OBJECT::dbo.usp_logwarden_backup_history TO [LogWardenToolCertificateUser];
REVOKE EXECUTE ON OBJECT::dbo.usp_logwarden_backup_history FROM [lw_agent];
