IF COL_LENGTH(N'dbo.search_semantic_train', N'text_view_id') IS NULL
  ALTER TABLE dbo.search_semantic_train ADD text_view_id VARCHAR(80) NOT NULL CONSTRAINT df_search_semantic_view DEFAULT 'raw-final-v1';
GO
IF COL_LENGTH(N'dbo.search_style_train', N'text_view_id') IS NULL
  ALTER TABLE dbo.search_style_train ADD text_view_id VARCHAR(80) NOT NULL CONSTRAINT df_search_style_view DEFAULT 'raw-final-v1';
GO

IF OBJECT_ID(N'dbo.metric_results', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.metric_results
  (
    metric_result_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_metric_results PRIMARY KEY,
    run_id VARCHAR(120) NOT NULL,
    evidence_tag VARCHAR(24) NOT NULL,
    representation_id VARCHAR(80) NOT NULL,
    method VARCHAR(80) NOT NULL,
    suite VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NULL,
    ci_low FLOAT NULL,
    ci_high FLOAT NULL,
    null_mean FLOAT NULL,
    null_p95 FLOAT NULL,
    rows_count INT NULL,
    groups_count INT NULL,
    detail_json NVARCHAR(MAX) NOT NULL,
    created_at DATETIME2(3) NOT NULL CONSTRAINT df_metric_results_at DEFAULT SYSUTCDATETIME(),
    CONSTRAINT ck_metric_results_json CHECK (ISJSON(detail_json)=1)
  );
  CREATE INDEX ix_metric_results_run ON dbo.metric_results(run_id,evidence_tag,representation_id,method,suite);
END;
GO

IF OBJECT_ID(N'dbo.claims', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.claims
  (
    claim_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_claims PRIMARY KEY,
    run_id VARCHAR(120) NOT NULL,
    research_question VARCHAR(40) NULL,
    evidence_tag VARCHAR(24) NOT NULL,
    representation_id VARCHAR(80) NOT NULL,
    method VARCHAR(80) NOT NULL,
    suite VARCHAR(100) NOT NULL,
    taxonomy VARCHAR(80) NOT NULL,
    supported BIT NOT NULL,
    rationale NVARCHAR(2000) NOT NULL,
    metric_result_id BIGINT NULL,
    created_at DATETIME2(3) NOT NULL CONSTRAINT df_claims_at DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_claims_metric FOREIGN KEY(metric_result_id) REFERENCES dbo.metric_results(metric_result_id)
  );
END;
GO

IF OBJECT_ID(N'dbo.evaluation_items', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.evaluation_items
  (
    evaluation_item_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_evaluation_items PRIMARY KEY,
    run_id VARCHAR(120) NOT NULL,
    source_id VARCHAR(100) NOT NULL,
    source_row_id VARCHAR(180) NOT NULL,
    prompt_group_id VARCHAR(120) NULL,
    split_role VARCHAR(40) NOT NULL,
    item_text NVARCHAR(MAX) NOT NULL,
    text_sha256 CHAR(64) NOT NULL,
    metadata_json NVARCHAR(MAX) NOT NULL,
    created_at DATETIME2(3) NOT NULL CONSTRAINT df_evaluation_items_at DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_evaluation_items UNIQUE(run_id,source_id,source_row_id),
    CONSTRAINT ck_evaluation_items_json CHECK(ISJSON(metadata_json)=1)
  );
END;
GO

IF OBJECT_ID(N'dbo.evaluation_vectors', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.evaluation_vectors
  (
    evaluation_item_id BIGINT NOT NULL,
    representation_id VARCHAR(80) NOT NULL,
    dimensions INT NOT NULL,
    vector_json NVARCHAR(MAX) NOT NULL,
    vector_sha256 CHAR(64) NOT NULL,
    created_by_run VARCHAR(120) NOT NULL,
    CONSTRAINT pk_evaluation_vectors PRIMARY KEY(evaluation_item_id,representation_id),
    CONSTRAINT fk_evaluation_vectors_item FOREIGN KEY(evaluation_item_id) REFERENCES dbo.evaluation_items(evaluation_item_id),
    CONSTRAINT ck_evaluation_vectors_json CHECK(ISJSON(vector_json)=1)
  );
END;
GO

IF OBJECT_ID(N'dbo.geometry_pairs', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.geometry_pairs
  (
    geometry_pair_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_geometry_pairs PRIMARY KEY,
    run_id VARCHAR(120) NOT NULL,
    representation_id VARCHAR(80) NOT NULL,
    left_generation_id BIGINT NOT NULL,
    right_generation_id BIGINT NOT NULL,
    distance FLOAT NOT NULL,
    same_model BIT NOT NULL,
    same_prompt_group BIT NOT NULL,
    same_family BIT NOT NULL,
    same_carrier BIT NOT NULL,
    same_decode BIT NOT NULL,
    length_difference INT NULL
  );
  CREATE INDEX ix_geometry_pairs_run ON dbo.geometry_pairs(run_id,representation_id);
END;
GO

IF OBJECT_ID(N'dbo.run_artifacts', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.run_artifacts
  (
    run_id VARCHAR(120) NOT NULL,
    relative_path NVARCHAR(500) NOT NULL,
    artifact_kind VARCHAR(80) NOT NULL,
    byte_count BIGINT NOT NULL,
    sha256 CHAR(64) NOT NULL,
    created_at DATETIME2(3) NOT NULL CONSTRAINT df_run_artifacts_at DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_run_artifacts PRIMARY KEY(run_id,relative_path)
  );
END;
GO
