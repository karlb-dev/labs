-- Catalog snapshots (GT-2-lite): the package catalog text files, stored raw
-- (authoritative for prompting) plus a tolerant structured parse for the
-- grammar-aware B2 baseline and identifier-binding validators. Lines the
-- parser does not model land in catalog.extras — nothing is silently dropped.
IF SCHEMA_ID(N'catalog') IS NULL
  EXEC (N'CREATE SCHEMA catalog');
GO

IF OBJECT_ID(N'catalog.snapshots', N'U') IS NULL
  CREATE TABLE catalog.snapshots
  (
    snapshot_id varchar(120) NOT NULL CONSTRAINT pk_catalog_snapshots PRIMARY KEY,
    catalog_name varchar(80) NOT NULL,
    source_file varchar(200) NOT NULL,
    header_json nvarchar(max) NOT NULL,
    raw_text nvarchar(max) NOT NULL,
    raw_sha256 char(64) NOT NULL,
    line_count int NOT NULL,
    object_count int NOT NULL,
    extra_line_count int NOT NULL,
    loaded_at_utc datetime2(3) NOT NULL CONSTRAINT df_catalog_loaded DEFAULT SYSUTCDATETIME()
  );
GO

IF OBJECT_ID(N'catalog.objects', N'U') IS NULL
  CREATE TABLE catalog.objects
  (
    object_pk int IDENTITY(1,1) NOT NULL CONSTRAINT pk_catalog_objects PRIMARY KEY,
    snapshot_id varchar(120) NOT NULL
      CONSTRAINT fk_catobj_snapshot REFERENCES catalog.snapshots(snapshot_id),
    object_kind varchar(30) NOT NULL,      -- table|view|procedure|function|synonym|system
    schema_name nvarchar(128) NULL,        -- NULL for system objects like sys.tables
    object_name nvarchar(256) NOT NULL,    -- unbracketed
    names_only bit NOT NULL CONSTRAINT df_catobj_namesonly DEFAULT 0,
    unavailable bit NOT NULL CONSTRAINT df_catobj_unavail DEFAULT 0,
    is_system_versioned bit NOT NULL CONSTRAINT df_catobj_sysver DEFAULT 0,
    synonym_target nvarchar(400) NULL,
    annotations_json nvarchar(max) NULL,   -- PERIOD FOR / HISTORY_TABLE / CONSTRAINT entries
    source_line int NOT NULL,
    raw_def nvarchar(max) NOT NULL
  );
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'ix_catobj_lookup')
  CREATE INDEX ix_catobj_lookup ON catalog.objects(snapshot_id, object_kind, object_name);
GO

IF OBJECT_ID(N'catalog.columns', N'U') IS NULL
  CREATE TABLE catalog.columns
  (
    column_pk int IDENTITY(1,1) NOT NULL CONSTRAINT pk_catalog_columns PRIMARY KEY,
    object_pk int NOT NULL CONSTRAINT fk_catcol_object REFERENCES catalog.objects(object_pk),
    ordinal int NOT NULL,
    column_name nvarchar(256) NOT NULL,    -- unbracketed; parameters keep their @
    is_parameter bit NOT NULL CONSTRAINT df_catcol_param DEFAULT 0,
    type_name nvarchar(200) NULL,
    nullability varchar(10) NULL,          -- 'NOT NULL' | 'NULL' | NULL (undeclared)
    is_pk bit NOT NULL CONSTRAINT df_catcol_pk DEFAULT 0,
    fk_target nvarchar(400) NULL,
    is_generated bit NOT NULL CONSTRAINT df_catcol_gen DEFAULT 0,
    raw_entry nvarchar(max) NOT NULL
  );
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'ix_catcol_object')
  CREATE INDEX ix_catcol_object ON catalog.columns(object_pk, ordinal);
GO

IF OBJECT_ID(N'catalog.extras', N'U') IS NULL
  CREATE TABLE catalog.extras
  (
    extra_pk int IDENTITY(1,1) NOT NULL CONSTRAINT pk_catalog_extras PRIMARY KEY,
    snapshot_id varchar(120) NOT NULL
      CONSTRAINT fk_catextra_snapshot REFERENCES catalog.snapshots(snapshot_id),
    kind varchar(40) NOT NULL,             -- fulltext_index|change_tracking|index|unparsed
    source_line int NOT NULL,
    raw_line nvarchar(max) NOT NULL
  );
GO
