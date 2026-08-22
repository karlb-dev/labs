IF OBJECT_ID(N'dbo.search_semantic_train', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.search_semantic_train
  (
    vector_id BIGINT NOT NULL CONSTRAINT pk_search_semantic_train PRIMARY KEY CLUSTERED,
    embedding_profile_id VARCHAR(80) NOT NULL,
    model_profile_id VARCHAR(80) NOT NULL,
    prompt_group_id VARCHAR(120) NOT NULL,
    text_artifact_id BIGINT NOT NULL,
    split VARCHAR(40) NOT NULL,
    length_band VARCHAR(16) NOT NULL,
    stratum VARCHAR(80) NOT NULL,
    corpus_manifest_hash CHAR(64) NOT NULL,
    frozen_at DATETIME2(3) NOT NULL,
    embedding VECTOR(1024) NOT NULL
  );
END;
GO

IF OBJECT_ID(N'dbo.search_segment_train', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.search_segment_train
  (
    vector_id BIGINT NOT NULL CONSTRAINT pk_search_segment_train PRIMARY KEY CLUSTERED,
    embedding_profile_id VARCHAR(80) NOT NULL,
    model_profile_id VARCHAR(80) NOT NULL,
    prompt_group_id VARCHAR(120) NOT NULL,
    text_artifact_id BIGINT NOT NULL,
    segment_id BIGINT NOT NULL,
    split VARCHAR(40) NOT NULL,
    length_band VARCHAR(16) NOT NULL,
    stratum VARCHAR(80) NOT NULL,
    corpus_manifest_hash CHAR(64) NOT NULL,
    frozen_at DATETIME2(3) NOT NULL,
    embedding VECTOR(1024) NOT NULL
  );
END;
GO

IF OBJECT_ID(N'dbo.search_residual_train', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.search_residual_train
  (
    vector_id BIGINT NOT NULL CONSTRAINT pk_search_residual_train PRIMARY KEY CLUSTERED,
    representation_id VARCHAR(80) NOT NULL,
    model_profile_id VARCHAR(80) NOT NULL,
    prompt_group_id VARCHAR(120) NOT NULL,
    text_artifact_id BIGINT NOT NULL,
    split VARCHAR(40) NOT NULL,
    length_band VARCHAR(16) NOT NULL,
    stratum VARCHAR(80) NOT NULL,
    corpus_manifest_hash CHAR(64) NOT NULL,
    frozen_at DATETIME2(3) NOT NULL,
    embedding VECTOR(1024) NOT NULL
  );
END;
GO

IF OBJECT_ID(N'dbo.search_style_train', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.search_style_train
  (
    vector_id BIGINT NOT NULL CONSTRAINT pk_search_style_train PRIMARY KEY CLUSTERED,
    model_profile_id VARCHAR(80) NOT NULL,
    prompt_group_id VARCHAR(120) NOT NULL,
    text_artifact_id BIGINT NOT NULL,
    split VARCHAR(40) NOT NULL,
    length_band VARCHAR(16) NOT NULL,
    stratum VARCHAR(80) NOT NULL,
    corpus_manifest_hash CHAR(64) NOT NULL,
    frozen_at DATETIME2(3) NOT NULL,
    embedding VECTOR(512) NOT NULL
  );
END;
GO

IF OBJECT_ID(N'dbo.search_fingerprint_train', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.search_fingerprint_train
  (
    vector_id BIGINT NOT NULL CONSTRAINT pk_search_fingerprint_train PRIMARY KEY CLUSTERED,
    model_profile_id VARCHAR(80) NOT NULL,
    prompt_group_id VARCHAR(120) NOT NULL,
    text_artifact_id BIGINT NOT NULL,
    split VARCHAR(40) NOT NULL,
    length_band VARCHAR(16) NOT NULL,
    stratum VARCHAR(80) NOT NULL,
    corpus_manifest_hash CHAR(64) NOT NULL,
    frozen_at DATETIME2(3) NOT NULL,
    embedding VECTOR(64) NOT NULL
  );
END;
GO

IF OBJECT_ID(N'dbo.search_likelihood_train', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.search_likelihood_train
  (
    vector_id BIGINT NOT NULL CONSTRAINT pk_search_likelihood_train PRIMARY KEY CLUSTERED,
    model_profile_id VARCHAR(80) NOT NULL,
    prompt_group_id VARCHAR(120) NOT NULL,
    text_artifact_id BIGINT NOT NULL,
    split VARCHAR(40) NOT NULL,
    length_band VARCHAR(16) NOT NULL,
    stratum VARCHAR(80) NOT NULL,
    corpus_manifest_hash CHAR(64) NOT NULL,
    frozen_at DATETIME2(3) NOT NULL,
    embedding VECTOR(8) NOT NULL
  );
END;
GO

IF OBJECT_ID(N'dbo.capability_snapshots', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.capability_snapshots
  (
    capability_snapshot_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_capability_snapshots PRIMARY KEY,
    run_id VARCHAR(120) NOT NULL,
    snapshot_sha256 CHAR(64) NOT NULL,
    snapshot_json NVARCHAR(MAX) NOT NULL,
    created_at DATETIME2(3) NOT NULL CONSTRAINT df_capability_snapshots_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_capability_snapshots_hash UNIQUE (snapshot_sha256),
    CONSTRAINT ck_capability_snapshots_json CHECK (ISJSON(snapshot_json) = 1)
  );
END;
GO

IF OBJECT_ID(N'dbo.search_runs', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.search_runs
  (
    search_run_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_search_runs PRIMARY KEY,
    run_id VARCHAR(120) NOT NULL,
    representation_id VARCHAR(80) NOT NULL,
    requested_search_mode VARCHAR(24) NOT NULL,
    actual_search_mode VARCHAR(32) NOT NULL,
    index_name SYSNAME NULL,
    index_version VARCHAR(20) NULL,
    metric VARCHAR(20) NOT NULL,
    candidate_k INT NOT NULL,
    returned_k INT NOT NULL,
    fallback_reason NVARCHAR(500) NULL,
    query_plan_hash CHAR(64) NULL,
    started_at DATETIME2(3) NOT NULL,
    finished_at DATETIME2(3) NULL,
    CONSTRAINT ck_search_runs_requested CHECK (requested_search_mode IN ('exact','ann_legacy','ann_v3','auto')),
    CONSTRAINT ck_search_runs_actual CHECK (actual_search_mode IN ('exact','ann_legacy','ann_v3','exact_fallback','execution_mode_unknown'))
  );
END;
GO

IF OBJECT_ID(N'dbo.neighbor_results', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.neighbor_results
  (
    search_run_id BIGINT NOT NULL,
    query_id BIGINT NOT NULL,
    rank INT NOT NULL,
    neighbor_id BIGINT NOT NULL,
    distance FLOAT NOT NULL,
    neighbor_model_profile_id VARCHAR(80) NOT NULL,
    neighbor_prompt_group_id VARCHAR(120) NOT NULL,
    CONSTRAINT pk_neighbor_results PRIMARY KEY (search_run_id, query_id, rank),
    CONSTRAINT fk_neighbor_results_run FOREIGN KEY (search_run_id) REFERENCES dbo.search_runs(search_run_id)
  );
END;
GO

IF OBJECT_ID(N'dbo.attribution_models', N'U') IS NULL CREATE TABLE dbo.attribution_models (attribution_model_id BIGINT IDENTITY PRIMARY KEY, run_id VARCHAR(120) NOT NULL, model_kind VARCHAR(80) NOT NULL, training_manifest_hash CHAR(64) NOT NULL, artifact_json NVARCHAR(MAX) NOT NULL, created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME());
GO
IF OBJECT_ID(N'dbo.calibration_models', N'U') IS NULL CREATE TABLE dbo.calibration_models (calibration_model_id BIGINT IDENTITY PRIMARY KEY, attribution_model_id BIGINT NOT NULL, method VARCHAR(80) NOT NULL, calibration_manifest_hash CHAR(64) NOT NULL, artifact_json NVARCHAR(MAX) NOT NULL, created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME());
GO
IF OBJECT_ID(N'dbo.prediction_runs', N'U') IS NULL CREATE TABLE dbo.prediction_runs (prediction_run_id BIGINT IDENTITY PRIMARY KEY, run_id VARCHAR(120) NOT NULL, suite VARCHAR(100) NOT NULL, representation_id VARCHAR(80) NOT NULL, method VARCHAR(80) NOT NULL, config_json NVARCHAR(MAX) NOT NULL, created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME());
GO
IF OBJECT_ID(N'dbo.predictions', N'U') IS NULL CREATE TABLE dbo.predictions (prediction_run_id BIGINT NOT NULL, generation_id BIGINT NOT NULL, true_model_profile_id VARCHAR(80) NULL, predicted_model_profile_id VARCHAR(80) NULL, decision VARCHAR(60) NOT NULL, calibrated_probability FLOAT NULL, candidate_set_json NVARCHAR(MAX) NOT NULL, novelty_score FLOAT NULL, PRIMARY KEY (prediction_run_id, generation_id));
GO
IF OBJECT_ID(N'dbo.prediction_candidates', N'U') IS NULL CREATE TABLE dbo.prediction_candidates (prediction_run_id BIGINT NOT NULL, generation_id BIGINT NOT NULL, model_profile_id VARCHAR(80) NOT NULL, probability FLOAT NOT NULL, features_json NVARCHAR(MAX) NOT NULL, PRIMARY KEY (prediction_run_id, generation_id, model_profile_id));
GO
IF OBJECT_ID(N'dbo.pair_runs', N'U') IS NULL CREATE TABLE dbo.pair_runs (pair_run_id BIGINT IDENTITY PRIMARY KEY, run_id VARCHAR(120) NOT NULL, config_json NVARCHAR(MAX) NOT NULL, created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME());
GO
IF OBJECT_ID(N'dbo.pair_predictions', N'U') IS NULL CREATE TABLE dbo.pair_predictions (pair_run_id BIGINT NOT NULL, left_generation_id BIGINT NOT NULL, right_generation_id BIGINT NOT NULL, pair_cell VARCHAR(40) NOT NULL, same_source_label BIT NOT NULL, probability FLOAT NOT NULL, predicted_same_source BIT NOT NULL, features_json NVARCHAR(MAX) NOT NULL, PRIMARY KEY (pair_run_id, left_generation_id, right_generation_id));
GO
IF OBJECT_ID(N'dbo.group_runs', N'U') IS NULL CREATE TABLE dbo.group_runs (group_run_id BIGINT IDENTITY PRIMARY KEY, run_id VARCHAR(120) NOT NULL, config_json NVARCHAR(MAX) NOT NULL, created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME());
GO
IF OBJECT_ID(N'dbo.group_assignments', N'U') IS NULL CREATE TABLE dbo.group_assignments (group_run_id BIGINT NOT NULL, generation_id BIGINT NOT NULL, source_group INT NULL, ambiguous BIT NOT NULL, diagnostics_json NVARCHAR(MAX) NOT NULL, PRIMARY KEY (group_run_id, generation_id));
GO
IF OBJECT_ID(N'dbo.cluster_runs', N'U') IS NULL CREATE TABLE dbo.cluster_runs (cluster_run_id BIGINT IDENTITY PRIMARY KEY, run_id VARCHAR(120) NOT NULL, representation_id VARCHAR(80) NOT NULL, algorithm VARCHAR(80) NOT NULL, labels_hidden BIT NOT NULL, config_json NVARCHAR(MAX) NOT NULL, created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME());
GO
IF OBJECT_ID(N'dbo.cluster_assignments', N'U') IS NULL CREATE TABLE dbo.cluster_assignments (cluster_run_id BIGINT NOT NULL, generation_id BIGINT NOT NULL, cluster_label INT NOT NULL, PRIMARY KEY (cluster_run_id, generation_id));
GO
IF OBJECT_ID(N'dbo.projection_coordinates', N'U') IS NULL CREATE TABLE dbo.projection_coordinates (cluster_run_id BIGINT NOT NULL, generation_id BIGINT NOT NULL, x FLOAT NOT NULL, y FLOAT NOT NULL, method VARCHAR(40) NOT NULL, PRIMARY KEY (cluster_run_id, generation_id, method));
GO
IF OBJECT_ID(N'dbo.ann_benchmark_runs', N'U') IS NULL CREATE TABLE dbo.ann_benchmark_runs (ann_benchmark_run_id BIGINT IDENTITY PRIMARY KEY, run_id VARCHAR(120) NOT NULL, representation_id VARCHAR(80) NOT NULL, corpus_size INT NOT NULL, index_name SYSNAME NULL, index_version VARCHAR(20) NULL, plan_evidence BIT NOT NULL, config_json NVARCHAR(MAX) NOT NULL, created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME());
GO
IF OBJECT_ID(N'dbo.ann_benchmark_rows', N'U') IS NULL CREATE TABLE dbo.ann_benchmark_rows (ann_benchmark_run_id BIGINT NOT NULL, query_id BIGINT NOT NULL, k INT NOT NULL, recall FLOAT NULL, latency_ms FLOAT NOT NULL, vote_agreement BIT NULL, decision_agreement BIT NULL, candidate_loss INT NOT NULL, fallback_reason NVARCHAR(500) NULL, execution_mode VARCHAR(40) NOT NULL, PRIMARY KEY (ann_benchmark_run_id, query_id, k));
GO
