IF OBJECT_ID(N'dbo.schema_migrations', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.schema_migrations
  (
    migration_id VARCHAR(80) NOT NULL CONSTRAINT pk_schema_migrations PRIMARY KEY,
    migration_sha256 CHAR(64) NOT NULL,
    applied_at DATETIME2(3) NOT NULL CONSTRAINT df_schema_migrations_applied DEFAULT SYSUTCDATETIME()
  );
END;
GO

IF OBJECT_ID(N'dbo.campaigns', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.campaigns
  (
    campaign_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_campaigns PRIMARY KEY,
    campaign_name VARCHAR(100) NOT NULL,
    tier VARCHAR(20) NOT NULL,
    campaign_hash CHAR(64) NOT NULL,
    governing_spec_hash CHAR(64) NOT NULL,
    governing_addendum_hash CHAR(64) NOT NULL,
    status VARCHAR(24) NOT NULL CONSTRAINT df_campaigns_status DEFAULT 'building',
    config_json NVARCHAR(MAX) NOT NULL,
    created_at DATETIME2(3) NOT NULL CONSTRAINT df_campaigns_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_campaigns_hash UNIQUE (campaign_hash),
    CONSTRAINT ck_campaigns_tier CHECK (tier IN ('smoke','dev','standard','full')),
    CONSTRAINT ck_campaigns_status CHECK (status IN ('building','frozen','running','complete','stopped')),
    CONSTRAINT ck_campaigns_config_json CHECK (ISJSON(config_json) = 1)
  );
END;
GO

IF OBJECT_ID(N'dbo.campaign_freezes', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.campaign_freezes
  (
    freeze_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_campaign_freezes PRIMARY KEY,
    campaign_id BIGINT NOT NULL,
    freeze_hash CHAR(64) NOT NULL,
    manifests_json NVARCHAR(MAX) NOT NULL,
    frozen_at DATETIME2(3) NOT NULL CONSTRAINT df_campaign_freezes_at DEFAULT SYSUTCDATETIME(),
    git_commit CHAR(40) NULL,
    CONSTRAINT uq_campaign_freezes_hash UNIQUE (freeze_hash),
    CONSTRAINT fk_campaign_freezes_campaign FOREIGN KEY (campaign_id) REFERENCES dbo.campaigns(campaign_id),
    CONSTRAINT ck_campaign_freezes_json CHECK (ISJSON(manifests_json) = 1)
  );
END;
GO

IF OBJECT_ID(N'dbo.model_profiles', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.model_profiles
  (
    model_profile_id VARCHAR(80) NOT NULL CONSTRAINT pk_model_profiles PRIMARY KEY,
    family VARCHAR(40) NOT NULL,
    model_repository NVARCHAR(240) NOT NULL,
    model_revision CHAR(40) NOT NULL,
    image_reference NVARCHAR(300) NOT NULL,
    image_digest CHAR(64) NOT NULL,
    chat_template_sha256 CHAR(64) NULL,
    profile_hash CHAR(64) NOT NULL,
    profile_json NVARCHAR(MAX) NOT NULL,
    is_target BIT NOT NULL,
    frozen_at DATETIME2(3) NOT NULL CONSTRAINT df_model_profiles_at DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_model_profiles_hash UNIQUE (profile_hash),
    CONSTRAINT ck_model_profiles_json CHECK (ISJSON(profile_json) = 1)
  );
END;
GO

IF OBJECT_ID(N'dbo.embedding_profiles', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.embedding_profiles
  (
    embedding_profile_id VARCHAR(80) NOT NULL CONSTRAINT pk_embedding_profiles PRIMARY KEY,
    model_repository NVARCHAR(240) NOT NULL,
    model_revision CHAR(40) NOT NULL,
    dimensions INT NOT NULL,
    profile_hash CHAR(64) NOT NULL,
    profile_json NVARCHAR(MAX) NOT NULL,
    CONSTRAINT uq_embedding_profiles_hash UNIQUE (profile_hash),
    CONSTRAINT ck_embedding_profiles_dimensions CHECK (dimensions = 1024),
    CONSTRAINT ck_embedding_profiles_json CHECK (ISJSON(profile_json) = 1)
  );
END;
GO

IF OBJECT_ID(N'dbo.decode_configs', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.decode_configs
  (
    decode_config_id VARCHAR(40) NOT NULL CONSTRAINT pk_decode_configs PRIMARY KEY,
    config_hash CHAR(64) NOT NULL,
    config_json NVARCHAR(MAX) NOT NULL,
    CONSTRAINT uq_decode_configs_hash UNIQUE (config_hash),
    CONSTRAINT ck_decode_configs_json CHECK (ISJSON(config_json) = 1)
  );
END;
GO

IF OBJECT_ID(N'dbo.prompt_sources', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.prompt_sources
  (
    prompt_source_id VARCHAR(80) NOT NULL CONSTRAINT pk_prompt_sources PRIMARY KEY,
    source_kind VARCHAR(32) NOT NULL,
    source_uri NVARCHAR(500) NOT NULL,
    source_revision VARCHAR(80) NOT NULL,
    source_sha256 CHAR(64) NOT NULL,
    license_id VARCHAR(80) NULL,
    manifest_json NVARCHAR(MAX) NOT NULL,
    CONSTRAINT uq_prompt_sources_identity UNIQUE (source_uri, source_revision, source_sha256),
    CONSTRAINT ck_prompt_sources_kind CHECK (source_kind IN ('repository','external','rag','synthetic')),
    CONSTRAINT ck_prompt_sources_json CHECK (ISJSON(manifest_json) = 1)
  );
END;
GO

IF OBJECT_ID(N'dbo.prompt_groups', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.prompt_groups
  (
    prompt_group_id VARCHAR(120) NOT NULL CONSTRAINT pk_prompt_groups PRIMARY KEY,
    prompt_source_id VARCHAR(80) NOT NULL,
    source_row_id VARCHAR(160) NOT NULL,
    family VARCHAR(100) NOT NULL,
    domain VARCHAR(100) NOT NULL,
    stratum VARCHAR(80) NOT NULL,
    split VARCHAR(40) NOT NULL,
    canonical_text NVARCHAR(MAX) NOT NULL,
    canonical_sha256 CHAR(64) NOT NULL,
    metadata_json NVARCHAR(MAX) NOT NULL,
    CONSTRAINT uq_prompt_groups_source_row UNIQUE (prompt_source_id, source_row_id),
    CONSTRAINT fk_prompt_groups_source FOREIGN KEY (prompt_source_id) REFERENCES dbo.prompt_sources(prompt_source_id),
    CONSTRAINT ck_prompt_groups_split CHECK (split IN ('train','calibration','test_id','test_source_holdout')),
    CONSTRAINT ck_prompt_groups_json CHECK (ISJSON(metadata_json) = 1)
  );
END;
GO

IF OBJECT_ID(N'dbo.prompt_variants', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.prompt_variants
  (
    prompt_variant_id VARCHAR(80) NOT NULL CONSTRAINT pk_prompt_variants PRIMARY KEY,
    prompt_group_id VARCHAR(120) NOT NULL,
    carrier_id VARCHAR(80) NOT NULL,
    rendered_text NVARCHAR(MAX) NOT NULL,
    render_sha256 CHAR(64) NOT NULL,
    max_tokens INT NOT NULL,
    evaluation_only BIT NOT NULL CONSTRAINT df_prompt_variants_eval DEFAULT 0,
    tier_membership_json NVARCHAR(200) NOT NULL,
    metadata_json NVARCHAR(MAX) NOT NULL,
    CONSTRAINT uq_prompt_variants_render UNIQUE (prompt_group_id, carrier_id, render_sha256),
    CONSTRAINT fk_prompt_variants_group FOREIGN KEY (prompt_group_id) REFERENCES dbo.prompt_groups(prompt_group_id),
    CONSTRAINT ck_prompt_variants_tokens CHECK (max_tokens BETWEEN 1 AND 2000),
    CONSTRAINT ck_prompt_variants_tier_json CHECK (ISJSON(tier_membership_json) = 1),
    CONSTRAINT ck_prompt_variants_metadata_json CHECK (ISJSON(metadata_json) = 1)
  );
END;
GO

IF OBJECT_ID(N'dbo.generation_jobs', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.generation_jobs
  (
    generation_job_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_generation_jobs PRIMARY KEY,
    job_key CHAR(64) NOT NULL,
    campaign_id BIGINT NOT NULL,
    model_profile_id VARCHAR(80) NOT NULL,
    prompt_variant_id VARCHAR(80) NOT NULL,
    decode_config_id VARCHAR(40) NOT NULL,
    sample_index INT NOT NULL CONSTRAINT df_generation_jobs_sample DEFAULT 0,
    status VARCHAR(24) NOT NULL CONSTRAINT df_generation_jobs_status DEFAULT 'pending',
    created_at DATETIME2(3) NOT NULL CONSTRAINT df_generation_jobs_created DEFAULT SYSUTCDATETIME(),
    completed_at DATETIME2(3) NULL,
    CONSTRAINT uq_generation_jobs_key UNIQUE (job_key),
    CONSTRAINT fk_generation_jobs_campaign FOREIGN KEY (campaign_id) REFERENCES dbo.campaigns(campaign_id),
    CONSTRAINT fk_generation_jobs_model FOREIGN KEY (model_profile_id) REFERENCES dbo.model_profiles(model_profile_id),
    CONSTRAINT fk_generation_jobs_prompt FOREIGN KEY (prompt_variant_id) REFERENCES dbo.prompt_variants(prompt_variant_id),
    CONSTRAINT fk_generation_jobs_decode FOREIGN KEY (decode_config_id) REFERENCES dbo.decode_configs(decode_config_id),
    CONSTRAINT ck_generation_jobs_status CHECK (status IN ('pending','running','complete','failed','stopped'))
  );
  CREATE INDEX ix_generation_jobs_resume ON dbo.generation_jobs(campaign_id, model_profile_id, status, generation_job_id);
END;
GO

IF OBJECT_ID(N'dbo.generation_attempts', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.generation_attempts
  (
    generation_attempt_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_generation_attempts PRIMARY KEY,
    generation_job_id BIGINT NOT NULL,
    attempt_number INT NOT NULL,
    request_json NVARCHAR(MAX) NOT NULL,
    raw_response_json NVARCHAR(MAX) NULL,
    http_status INT NULL,
    latency_ms FLOAT NULL,
    error_class VARCHAR(100) NULL,
    error_detail NVARCHAR(MAX) NULL,
    started_at DATETIME2(3) NOT NULL,
    finished_at DATETIME2(3) NULL,
    CONSTRAINT uq_generation_attempt UNIQUE (generation_job_id, attempt_number),
    CONSTRAINT fk_generation_attempts_job FOREIGN KEY (generation_job_id) REFERENCES dbo.generation_jobs(generation_job_id),
    CONSTRAINT ck_generation_attempts_request CHECK (ISJSON(request_json) = 1),
    CONSTRAINT ck_generation_attempts_response CHECK (raw_response_json IS NULL OR ISJSON(raw_response_json) = 1)
  );
END;
GO

IF OBJECT_ID(N'dbo.generations', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.generations
  (
    generation_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_generations PRIMARY KEY,
    generation_job_id BIGINT NOT NULL,
    campaign_id BIGINT NOT NULL,
    model_profile_id VARCHAR(80) NOT NULL,
    prompt_variant_id VARCHAR(80) NOT NULL,
    decode_config_id VARCHAR(40) NOT NULL,
    split VARCHAR(40) NOT NULL,
    sample_index INT NOT NULL,
    request_json NVARCHAR(MAX) NOT NULL,
    raw_response_json NVARCHAR(MAX) NOT NULL,
    final_text NVARCHAR(MAX) NOT NULL,
    reasoning_text NVARCHAR(MAX) NULL,
    output_sha256 CHAR(64) NOT NULL,
    normalized_output_sha256 CHAR(64) NOT NULL,
    input_token_count INT NULL,
    output_token_count INT NULL,
    reference_token_count INT NULL,
    reasoning_token_count INT NULL,
    output_char_count INT NOT NULL,
    output_word_count INT NOT NULL,
    length_band VARCHAR(16) NULL,
    latency_ms FLOAT NOT NULL,
    finish_reason VARCHAR(80) NULL,
    http_status INT NOT NULL,
    attempt_count INT NOT NULL,
    truncated BIT NOT NULL,
    template_residue_found BIT NOT NULL,
    self_name_found BIT NOT NULL,
    started_at DATETIME2(3) NOT NULL,
    finished_at DATETIME2(3) NOT NULL,
    error_class VARCHAR(100) NULL,
    error_detail NVARCHAR(MAX) NULL,
    CONSTRAINT uq_generations_job UNIQUE (generation_job_id),
    CONSTRAINT fk_generations_job FOREIGN KEY (generation_job_id) REFERENCES dbo.generation_jobs(generation_job_id),
    CONSTRAINT fk_generations_campaign FOREIGN KEY (campaign_id) REFERENCES dbo.campaigns(campaign_id),
    CONSTRAINT fk_generations_model FOREIGN KEY (model_profile_id) REFERENCES dbo.model_profiles(model_profile_id),
    CONSTRAINT fk_generations_prompt FOREIGN KEY (prompt_variant_id) REFERENCES dbo.prompt_variants(prompt_variant_id),
    CONSTRAINT fk_generations_decode FOREIGN KEY (decode_config_id) REFERENCES dbo.decode_configs(decode_config_id),
    CONSTRAINT ck_generations_split CHECK (split IN ('train','calibration','test_id','test_source_holdout')),
    CONSTRAINT ck_generations_length_band CHECK (length_band IS NULL OR length_band IN ('micro','short','medium','long')),
    CONSTRAINT ck_generations_request CHECK (ISJSON(request_json) = 1),
    CONSTRAINT ck_generations_response CHECK (ISJSON(raw_response_json) = 1)
  );
  CREATE INDEX ix_generations_eval ON dbo.generations(campaign_id, split, model_profile_id, generation_id);
END;
GO

IF OBJECT_ID(N'dbo.text_artifacts', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.text_artifacts
  (
    text_artifact_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_text_artifacts PRIMARY KEY,
    text_view_id VARCHAR(80) NOT NULL,
    normalized_sha256 CHAR(64) NOT NULL,
    artifact_text NVARCHAR(MAX) NOT NULL,
    multi_source_exact_duplicate BIT NOT NULL CONSTRAINT df_text_artifacts_duplicate DEFAULT 0,
    created_at DATETIME2(3) NOT NULL CONSTRAINT df_text_artifacts_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_text_artifacts_view_hash UNIQUE (text_view_id, normalized_sha256)
  );
END;
GO

IF OBJECT_ID(N'dbo.generation_text_artifacts', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.generation_text_artifacts
  (
    generation_id BIGINT NOT NULL,
    text_artifact_id BIGINT NOT NULL,
    CONSTRAINT pk_generation_text_artifacts PRIMARY KEY (generation_id, text_artifact_id),
    CONSTRAINT fk_generation_text_generation FOREIGN KEY (generation_id) REFERENCES dbo.generations(generation_id),
    CONSTRAINT fk_generation_text_artifact FOREIGN KEY (text_artifact_id) REFERENCES dbo.text_artifacts(text_artifact_id)
  );
END;
GO

IF OBJECT_ID(N'dbo.output_segments', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.output_segments
  (
    segment_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_output_segments PRIMARY KEY,
    text_artifact_id BIGINT NOT NULL,
    segmenter_id VARCHAR(80) NOT NULL,
    ordinal INT NOT NULL,
    char_start INT NOT NULL,
    char_end INT NOT NULL,
    token_start INT NULL,
    token_end INT NULL,
    segment_text NVARCHAR(MAX) NOT NULL,
    segment_sha256 CHAR(64) NOT NULL,
    is_primary_eligible BIT NOT NULL,
    CONSTRAINT uq_output_segments UNIQUE (text_artifact_id, segmenter_id, ordinal),
    CONSTRAINT fk_output_segments_artifact FOREIGN KEY (text_artifact_id) REFERENCES dbo.text_artifacts(text_artifact_id),
    CONSTRAINT ck_output_segments_bounds CHECK (char_start >= 0 AND char_end >= char_start)
  );
END;
GO

IF OBJECT_ID(N'dbo.evidence_events', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.evidence_events
  (
    evidence_event_id BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_evidence_events PRIMARY KEY,
    campaign_id BIGINT NULL,
    run_id VARCHAR(120) NOT NULL,
    stage VARCHAR(80) NOT NULL,
    event_kind VARCHAR(80) NOT NULL,
    disposition VARCHAR(80) NOT NULL,
    detail_json NVARCHAR(MAX) NOT NULL,
    created_at DATETIME2(3) NOT NULL CONSTRAINT df_evidence_events_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_evidence_events_campaign FOREIGN KEY (campaign_id) REFERENCES dbo.campaigns(campaign_id),
    CONSTRAINT ck_evidence_events_json CHECK (ISJSON(detail_json) = 1)
  );
END;
GO
