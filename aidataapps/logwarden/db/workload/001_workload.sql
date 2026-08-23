SET XACT_ABORT ON;

IF SCHEMA_ID(N'lab') IS NULL EXEC(N'CREATE SCHEMA lab AUTHORIZATION dbo');

IF OBJECT_ID(N'lab.execution_markers', N'U') IS NULL
BEGIN
  CREATE TABLE lab.execution_markers
  (
    execution_marker_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_lab_execution_markers PRIMARY KEY,
    episode_id varchar(120) NOT NULL,
    injector_id varchar(100) NOT NULL,
    opaque_token uniqueidentifier NOT NULL,
    phase varchar(32) NOT NULL,
    detail_json nvarchar(max) NOT NULL,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_lab_execution_markers_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_lab_execution_marker UNIQUE(episode_id, injector_id, phase),
    CONSTRAINT ck_lab_execution_markers_json CHECK (ISJSON(detail_json) = 1)
  );
END;

IF OBJECT_ID(N'lab.accounts', N'U') IS NULL
BEGIN
  CREATE TABLE lab.accounts
  (
    account_id int NOT NULL CONSTRAINT pk_lab_accounts PRIMARY KEY,
    opaque_code varchar(40) NOT NULL CONSTRAINT uq_lab_accounts_code UNIQUE,
    amount decimal(12,2) NOT NULL,
    short_value varchar(12) NULL,
    updated_at_utc datetime2(7) NOT NULL CONSTRAINT df_lab_accounts_updated DEFAULT SYSUTCDATETIME()
  );
  INSERT lab.accounts(account_id, opaque_code, amount, short_value)
  VALUES(1, 'lw_t_0001', 100.00, 'seed-a'), (2, 'lw_t_0002', 200.00, 'seed-b');
END;

IF OBJECT_ID(N'lab.lock_a', N'U') IS NULL
BEGIN
  CREATE TABLE lab.lock_a(id int NOT NULL CONSTRAINT pk_lab_lock_a PRIMARY KEY, value int NOT NULL);
  INSERT lab.lock_a VALUES(1, 0), (2, 0);
END;

IF OBJECT_ID(N'lab.lock_b', N'U') IS NULL
BEGIN
  CREATE TABLE lab.lock_b(id int NOT NULL CONSTRAINT pk_lab_lock_b PRIMARY KEY, value int NOT NULL);
  INSERT lab.lock_b VALUES(1, 0), (2, 0);
END;

IF OBJECT_ID(N'lab.noise_rows', N'U') IS NULL
BEGIN
  CREATE TABLE lab.noise_rows
  (
    noise_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_lab_noise_rows PRIMARY KEY,
    opaque_value nvarchar(400) NOT NULL,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_lab_noise_created DEFAULT SYSUTCDATETIME()
  );
END;
GO

CREATE OR ALTER PROCEDURE lab.usp_reset_seed
AS
BEGIN
  SET NOCOUNT ON;
  SET XACT_ABORT ON;
  BEGIN TRANSACTION;
  DELETE FROM lab.noise_rows;
  DELETE FROM lab.accounts;
  INSERT lab.accounts(account_id, opaque_code, amount, short_value)
  VALUES(1, 'lw_t_0001', 100.00, 'seed-a'), (2, 'lw_t_0002', 200.00, 'seed-b');
  UPDATE lab.lock_a SET value = 0;
  UPDATE lab.lock_b SET value = 0;
  COMMIT;
END;
GO

CREATE OR ALTER PROCEDURE lab.usp_raise_controlled_signal
  @message nvarchar(1000),
  @severity int,
  @state int
AS
BEGIN
  SET NOCOUNT ON;
  IF @severity NOT BETWEEN 11 AND 19 THROW 51300, 'Controlled signal severity must be 11 through 19', 1;
  IF @state NOT BETWEEN 1 AND 127 THROW 51301, 'Controlled signal state must be 1 through 127', 1;
  RAISERROR(@message, @severity, @state) WITH LOG;
END;
