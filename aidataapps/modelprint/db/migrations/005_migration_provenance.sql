IF OBJECT_ID(N'dbo.schema_migration_reconciliations', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.schema_migration_reconciliations
  (
    migration_id VARCHAR(80) NOT NULL,
    recorded_sha256 CHAR(64) NOT NULL,
    source_sha256 CHAR(64) NOT NULL,
    rationale NVARCHAR(1000) NOT NULL,
    reconciled_at DATETIME2(3) NOT NULL CONSTRAINT df_schema_migration_reconciliations_at DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_schema_migration_reconciliations PRIMARY KEY (migration_id, recorded_sha256, source_sha256)
  );
END;
GO

-- The full experiment database was initialized before the initial Lab 02
-- source milestone. During that construction window, 001 and 003 gained the
-- final idempotent schema definitions while the database retained the hashes
-- of their earlier working copies. Preserve both hashes instead of rewriting
-- history. A fresh database records the final source hashes directly and does
-- not need either reconciliation row.
IF EXISTS
(
  SELECT 1 FROM dbo.schema_migrations
  WHERE migration_id='001_core.sql'
    AND migration_sha256='160432d7933eaf9579fcca2ff0eabdedf7cb2203afd05c005062e9305ecc0f54'
)
AND NOT EXISTS
(
  SELECT 1 FROM dbo.schema_migration_reconciliations
  WHERE migration_id='001_core.sql'
    AND recorded_sha256='160432d7933eaf9579fcca2ff0eabdedf7cb2203afd05c005062e9305ecc0f54'
    AND source_sha256='82d55d746f19a8a1b441e19d9d35a32eb82d19797eb8f3d190f7fa98be4b6c71'
)
  INSERT dbo.schema_migration_reconciliations(migration_id,recorded_sha256,source_sha256,rationale)
  VALUES('001_core.sql','160432d7933eaf9579fcca2ff0eabdedf7cb2203afd05c005062e9305ecc0f54',
    '82d55d746f19a8a1b441e19d9d35a32eb82d19797eb8f3d190f7fa98be4b6c71',
    N'Initial full-run database predates the first committed Lab 02 source milestone; final idempotent core DDL is schema-parity audited by the CPU handoff workflow.');
GO

IF EXISTS
(
  SELECT 1 FROM dbo.schema_migrations
  WHERE migration_id='003_search_and_eval.sql'
    AND migration_sha256='ea3b59eef141d77e4b50368c80abc677cfa7eb53f022e6c9b5f1233bcd13b9d4'
)
AND NOT EXISTS
(
  SELECT 1 FROM dbo.schema_migration_reconciliations
  WHERE migration_id='003_search_and_eval.sql'
    AND recorded_sha256='ea3b59eef141d77e4b50368c80abc677cfa7eb53f022e6c9b5f1233bcd13b9d4'
    AND source_sha256='c87e9ac32bf76302eecfff008868a8bacbc36106442040a4a046666a864eb495'
)
  INSERT dbo.schema_migration_reconciliations(migration_id,recorded_sha256,source_sha256,rationale)
  VALUES('003_search_and_eval.sql','ea3b59eef141d77e4b50368c80abc677cfa7eb53f022e6c9b5f1233bcd13b9d4',
    'c87e9ac32bf76302eecfff008868a8bacbc36106442040a4a046666a864eb495',
    N'Initial full-run database predates the first committed Lab 02 source milestone; final idempotent search/evaluation DDL is schema-parity audited by the CPU handoff workflow.');
GO
