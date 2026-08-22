IF OBJECT_ID(N'dbo.assets', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.assets
  (
    asset_tag VARCHAR(32) NOT NULL CONSTRAINT pk_assets PRIMARY KEY,
    model NVARCHAR(80) NOT NULL,
    depot NVARCHAR(80) NOT NULL,
    status VARCHAR(24) NOT NULL,
    odometer_km INT NOT NULL,
    last_service_at DATETIME2(0) NOT NULL,
    notes NVARCHAR(500) NULL,
    CONSTRAINT ck_assets_status CHECK (status IN ('active', 'inspection_due', 'out_of_service'))
  );
END;
GO

IF OBJECT_ID(N'dbo.knowledge_documents', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.knowledge_documents
  (
    document_id VARCHAR(80) NOT NULL CONSTRAINT pk_knowledge_documents PRIMARY KEY,
    title NVARCHAR(200) NOT NULL,
    source_uri NVARCHAR(300) NOT NULL,
    revision VARCHAR(32) NOT NULL,
    effective_at DATE NOT NULL
  );
END;
GO

IF OBJECT_ID(N'dbo.knowledge_chunks', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.knowledge_chunks
  (
    chunk_id VARCHAR(100) NOT NULL CONSTRAINT pk_knowledge_chunks PRIMARY KEY,
    document_id VARCHAR(80) NOT NULL,
    document_title NVARCHAR(200) NOT NULL,
    heading NVARCHAR(200) NOT NULL,
    content NVARCHAR(MAX) NOT NULL,
    content_sha256 CHAR(64) NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    CONSTRAINT fk_knowledge_chunks_document
      FOREIGN KEY (document_id) REFERENCES dbo.knowledge_documents(document_id)
  );
  CREATE INDEX ix_knowledge_chunks_document ON dbo.knowledge_chunks(document_id);
END;
GO

IF OBJECT_ID(N'dbo.work_orders', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.work_orders
  (
    work_order_id INT IDENTITY(1, 1) NOT NULL CONSTRAINT pk_work_orders PRIMARY KEY,
    asset_tag VARCHAR(32) NOT NULL,
    title NVARCHAR(160) NOT NULL,
    description NVARCHAR(MAX) NOT NULL,
    priority VARCHAR(16) NOT NULL,
    status VARCHAR(16) NOT NULL CONSTRAINT df_work_orders_status DEFAULT ('open'),
    created_at DATETIME2(3) NOT NULL CONSTRAINT df_work_orders_created_at DEFAULT SYSUTCDATETIME(),
    closed_at DATETIME2(3) NULL,
    CONSTRAINT fk_work_orders_asset FOREIGN KEY (asset_tag) REFERENCES dbo.assets(asset_tag),
    CONSTRAINT ck_work_orders_priority CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    CONSTRAINT ck_work_orders_status CHECK (status IN ('open', 'in_progress', 'closed', 'cancelled'))
  );
  CREATE INDEX ix_work_orders_asset_status ON dbo.work_orders(asset_tag, status);
END;
GO
