IF OBJECT_ID(N'dbo.output_scalar_features', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.output_scalar_features
  (
    text_artifact_id BIGINT NOT NULL,
    feature_schema_hash CHAR(64) NOT NULL,
    features_json NVARCHAR(MAX) NOT NULL,
    CONSTRAINT pk_output_scalar_features PRIMARY KEY (text_artifact_id, feature_schema_hash),
    CONSTRAINT fk_output_scalar_features_artifact FOREIGN KEY (text_artifact_id) REFERENCES dbo.text_artifacts(text_artifact_id),
    CONSTRAINT ck_output_scalar_features_json CHECK (ISJSON(features_json) = 1)
  );
END;
GO

IF OBJECT_ID(N'dbo.phrase_dictionary', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.phrase_dictionary
  (
    phrase_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_phrase_dictionary PRIMARY KEY,
    phrase_kind VARCHAR(40) NOT NULL,
    phrase NVARCHAR(600) NOT NULL,
    phrase_sha256 CHAR(64) NOT NULL,
    CONSTRAINT uq_phrase_dictionary UNIQUE (phrase_kind, phrase_sha256)
  );
END;
GO

IF OBJECT_ID(N'dbo.phrase_occurrences', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.phrase_occurrences
  (
    phrase_id BIGINT NOT NULL,
    generation_id BIGINT NOT NULL,
    CONSTRAINT pk_phrase_occurrences PRIMARY KEY (phrase_id, generation_id),
    CONSTRAINT fk_phrase_occurrences_phrase FOREIGN KEY (phrase_id) REFERENCES dbo.phrase_dictionary(phrase_id),
    CONSTRAINT fk_phrase_occurrences_generation FOREIGN KEY (generation_id) REFERENCES dbo.generations(generation_id)
  );
END;
GO

IF OBJECT_ID(N'dbo.model_phrase_stats', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.model_phrase_stats
  (
    phrase_id BIGINT NOT NULL,
    model_profile_id VARCHAR(80) NOT NULL,
    document_frequency_model INT NOT NULL,
    document_frequency_other INT NOT NULL,
    log_odds FLOAT NOT NULL,
    z_score FLOAT NOT NULL,
    training_manifest_hash CHAR(64) NOT NULL,
    CONSTRAINT pk_model_phrase_stats PRIMARY KEY (phrase_id, model_profile_id, training_manifest_hash),
    CONSTRAINT fk_model_phrase_stats_phrase FOREIGN KEY (phrase_id) REFERENCES dbo.phrase_dictionary(phrase_id),
    CONSTRAINT fk_model_phrase_stats_model FOREIGN KEY (model_profile_id) REFERENCES dbo.model_profiles(model_profile_id)
  );
END;
GO

IF OBJECT_ID(N'dbo.prompt_embeddings', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.prompt_embeddings
  (
    prompt_variant_id VARCHAR(80) NOT NULL,
    embedding_profile_id VARCHAR(80) NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    embedding_sha256 CHAR(64) NOT NULL,
    CONSTRAINT pk_prompt_embeddings PRIMARY KEY (prompt_variant_id, embedding_profile_id),
    CONSTRAINT fk_prompt_embeddings_variant FOREIGN KEY (prompt_variant_id) REFERENCES dbo.prompt_variants(prompt_variant_id),
    CONSTRAINT fk_prompt_embeddings_profile FOREIGN KEY (embedding_profile_id) REFERENCES dbo.embedding_profiles(embedding_profile_id)
  );
END;
GO

IF OBJECT_ID(N'dbo.semantic_vectors', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.semantic_vectors
  (
    semantic_vector_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_semantic_vectors PRIMARY KEY,
    text_artifact_id BIGINT NULL,
    segment_id BIGINT NULL,
    embedding_profile_id VARCHAR(80) NOT NULL,
    representation_id VARCHAR(80) NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    embedding_sha256 CHAR(64) NOT NULL,
    created_by_run VARCHAR(120) NOT NULL,
    created_at DATETIME2(3) NOT NULL CONSTRAINT df_semantic_vectors_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_semantic_vectors_artifact FOREIGN KEY (text_artifact_id) REFERENCES dbo.text_artifacts(text_artifact_id),
    CONSTRAINT fk_semantic_vectors_segment FOREIGN KEY (segment_id) REFERENCES dbo.output_segments(segment_id),
    CONSTRAINT fk_semantic_vectors_profile FOREIGN KEY (embedding_profile_id) REFERENCES dbo.embedding_profiles(embedding_profile_id),
    CONSTRAINT ck_semantic_vector_owner CHECK (
      (text_artifact_id IS NOT NULL AND segment_id IS NULL)
      OR (text_artifact_id IS NULL AND segment_id IS NOT NULL)
    ),
    CONSTRAINT uq_semantic_vectors UNIQUE (text_artifact_id, segment_id, embedding_profile_id, representation_id)
  );
END;
GO

IF OBJECT_ID(N'dbo.residual_vectors', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.residual_vectors
  (
    residual_vector_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_residual_vectors PRIMARY KEY,
    generation_id BIGINT NOT NULL,
    embedding_profile_id VARCHAR(80) NOT NULL,
    representation_id VARCHAR(80) NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    training_manifest_hash CHAR(64) NULL,
    created_by_run VARCHAR(120) NOT NULL,
    CONSTRAINT uq_residual_vectors UNIQUE (generation_id, embedding_profile_id, representation_id),
    CONSTRAINT fk_residual_vectors_generation FOREIGN KEY (generation_id) REFERENCES dbo.generations(generation_id),
    CONSTRAINT fk_residual_vectors_profile FOREIGN KEY (embedding_profile_id) REFERENCES dbo.embedding_profiles(embedding_profile_id)
  );
END;
GO

IF OBJECT_ID(N'dbo.style_vectors', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.style_vectors
  (
    style_vector_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_style_vectors PRIMARY KEY,
    text_artifact_id BIGINT NOT NULL,
    representation_id VARCHAR(80) NOT NULL,
    feature_schema_hash CHAR(64) NOT NULL,
    embedding VECTOR(512) NOT NULL,
    created_by_run VARCHAR(120) NOT NULL,
    CONSTRAINT uq_style_vectors UNIQUE (text_artifact_id, representation_id),
    CONSTRAINT fk_style_vectors_artifact FOREIGN KEY (text_artifact_id) REFERENCES dbo.text_artifacts(text_artifact_id)
  );
END;
GO

IF OBJECT_ID(N'dbo.fingerprint_vectors', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.fingerprint_vectors
  (
    fingerprint_vector_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_fingerprint_vectors PRIMARY KEY,
    generation_id BIGINT NOT NULL,
    representation_id VARCHAR(80) NOT NULL,
    projection_manifest_hash CHAR(64) NOT NULL,
    embedding VECTOR(64) NOT NULL,
    created_by_run VARCHAR(120) NOT NULL,
    CONSTRAINT uq_fingerprint_vectors UNIQUE (generation_id, representation_id),
    CONSTRAINT fk_fingerprint_vectors_generation FOREIGN KEY (generation_id) REFERENCES dbo.generations(generation_id)
  );
END;
GO

IF OBJECT_ID(N'dbo.likelihood_scores', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.likelihood_scores
  (
    likelihood_score_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_likelihood_scores PRIMARY KEY,
    generation_id BIGINT NOT NULL,
    scoring_model_profile_id VARCHAR(80) NOT NULL,
    prompted BIT NOT NULL,
    ll_sum_nats FLOAT NOT NULL,
    tokens_m INT NOT NULL,
    chars INT NOT NULL,
    bits_per_char FLOAT NOT NULL,
    bits_per_ref_token FLOAT NULL,
    scored_at DATETIME2(3) NOT NULL CONSTRAINT df_likelihood_scores_at DEFAULT SYSUTCDATETIME(),
    scoring_run_id VARCHAR(120) NOT NULL,
    CONSTRAINT uq_likelihood_scores UNIQUE (generation_id, scoring_model_profile_id, prompted),
    CONSTRAINT fk_likelihood_scores_generation FOREIGN KEY (generation_id) REFERENCES dbo.generations(generation_id),
    CONSTRAINT fk_likelihood_scores_model FOREIGN KEY (scoring_model_profile_id) REFERENCES dbo.model_profiles(model_profile_id)
  );
END;
GO

IF OBJECT_ID(N'dbo.likelihood_profile_vectors', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.likelihood_profile_vectors
  (
    likelihood_profile_vector_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_likelihood_profile_vectors PRIMARY KEY,
    generation_id BIGINT NOT NULL,
    representation_id VARCHAR(80) NOT NULL,
    embedding VECTOR(8) NOT NULL,
    training_manifest_hash CHAR(64) NOT NULL,
    CONSTRAINT uq_likelihood_profile_vectors UNIQUE (generation_id, representation_id),
    CONSTRAINT fk_likelihood_profile_generation FOREIGN KEY (generation_id) REFERENCES dbo.generations(generation_id)
  );
END;
GO
