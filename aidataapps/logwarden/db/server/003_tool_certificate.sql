SET XACT_ABORT ON;

DECLARE @certificate varbinary(max),@statement nvarchar(max);
USE [LogWardenControl];
SELECT @certificate=CERTENCODED(CERT_ID(N'LogWardenToolCertificate'));
IF @certificate IS NULL THROW 51720, 'LogWarden tool certificate is missing from control database', 1;

USE [master];
IF CERT_ID(N'LogWardenToolCertificate') IS NULL
BEGIN
  SET @statement=N'CREATE CERTIFICATE [LogWardenToolCertificate] FROM BINARY='+CONVERT(nvarchar(max),@certificate,1)+N';';
  EXEC(@statement);
END;
IF CERTENCODED(CERT_ID(N'LogWardenToolCertificate'))<>@certificate
  THROW 51721, 'Master tool certificate does not match control certificate', 1;
IF SUSER_ID(N'LogWardenToolCertificateLogin') IS NULL
  CREATE LOGIN [LogWardenToolCertificateLogin] FROM CERTIFICATE [LogWardenToolCertificate];
GRANT VIEW SERVER STATE TO [LogWardenToolCertificateLogin];
GRANT VIEW SERVER PERFORMANCE STATE TO [LogWardenToolCertificateLogin];
GRANT VIEW ANY DATABASE TO [LogWardenToolCertificateLogin];

USE [msdb];
IF CERT_ID(N'LogWardenToolCertificate') IS NULL
BEGIN
  SET @statement=N'CREATE CERTIFICATE [LogWardenToolCertificate] FROM BINARY='+CONVERT(nvarchar(max),@certificate,1)+N';';
  EXEC(@statement);
END;
IF CERTENCODED(CERT_ID(N'LogWardenToolCertificate'))<>@certificate
  THROW 51722, 'msdb tool certificate does not match control certificate', 1;
IF USER_ID(N'LogWardenToolCertificateUser') IS NULL
  CREATE USER [LogWardenToolCertificateUser] FOR CERTIFICATE [LogWardenToolCertificate];
GRANT SELECT ON OBJECT::dbo.backupset TO [LogWardenToolCertificateUser];
GRANT SELECT ON OBJECT::dbo.backupmediafamily TO [LogWardenToolCertificateUser];
