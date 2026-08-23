-- Immutable dataset rows: cases, gold contracts, split groups, oracle flags.
IF OBJECT_ID(N'dataset.dataset_versions', N'U') IS NULL
  CREATE TABLE dataset.dataset_versions
  (
    dataset_version varchar(40) NOT NULL CONSTRAINT pk_dataset_versions PRIMARY KEY,
    package_path nvarchar(400) NOT NULL,
    record_count int NOT NULL,
    legacy_count int NOT NULL,
    manifest_sha256 char(64) NOT NULL,
    imported_at_utc datetime2(3) NOT NULL CONSTRAINT df_dsv_imported DEFAULT SYSUTCDATETIME()
  );

IF OBJECT_ID(N'dataset.cases', N'U') IS NULL
  CREATE TABLE dataset.cases
  (
    case_id varchar(80) NOT NULL CONSTRAINT pk_dataset_cases PRIMARY KEY,
    dataset_version varchar(40) NOT NULL,
    legacy_id varchar(80) NULL,
    record_sha256 char(64) NOT NULL,
    data_role varchar(40) NOT NULL,
    split_group_id varchar(120) NOT NULL,
    template_family varchar(120) NOT NULL,
    near_duplicate_family varchar(120) NOT NULL,
    completion_class varchar(40) NOT NULL,
    completion_category varchar(40) NOT NULL,
    dialect varchar(20) NOT NULL,
    compatibility_level int NOT NULL,
    fixture_id varchar(80) NOT NULL,
    catalog_snapshot_id varchar(120) NOT NULL,
    permission_profile_id varchar(80) NOT NULL,
    cursor_offset_utf16 int NOT NULL,
    cursor_offset_utf8 int NOT NULL,
    selection_start_utf16 int NOT NULL,
    selection_end_utf16 int NOT NULL,
    doc_prefix nvarchar(max) NOT NULL,
    doc_suffix nvarchar(max) NOT NULL,
    line_prefix nvarchar(max) NOT NULL,
    line_suffix nvarchar(max) NOT NULL,
    canonical_insertion nvarchar(max) NOT NULL,
    expect_empty bit NOT NULL,
    known_ambiguity bit NOT NULL,
    abstention_reason nvarchar(400) NULL,
    difficulty varchar(20) NOT NULL,
    scenario varchar(40) NOT NULL,
    capability_requirements_json nvarchar(max) NOT NULL,
    must_reference_json nvarchar(max) NOT NULL,
    must_not_reference_json nvarchar(max) NOT NULL,
    oracle_json nvarchar(max) NOT NULL,
    eval_json nvarchar(max) NOT NULL,
    prompt_message nvarchar(max) NOT NULL,   -- frozen model-neutral single user message (packet prompt bytes)
    prompt_sha256 char(64) NOT NULL,
    source_json nvarchar(max) NOT NULL,
    environment_json nvarchar(max) NOT NULL
  );

IF OBJECT_ID(N'dataset.gold_candidates', N'U') IS NULL
  CREATE TABLE dataset.gold_candidates
  (
    case_id varchar(80) NOT NULL,
    ordinal int NOT NULL,
    insertion nvarchar(max) NOT NULL,
    is_canonical bit NOT NULL,
    CONSTRAINT pk_gold_candidates PRIMARY KEY (case_id, ordinal)
  );

IF OBJECT_ID(N'dataset.split_groups', N'U') IS NULL
  CREATE TABLE dataset.split_groups
  (
    split_group_id varchar(120) NOT NULL CONSTRAINT pk_split_groups PRIMARY KEY,
    data_role varchar(40) NOT NULL,
    case_count int NOT NULL
  );

IF OBJECT_ID(N'dataset.case_oracle_status', N'U') IS NULL
  CREATE TABLE dataset.case_oracle_status
  (
    case_id varchar(80) NOT NULL,
    oracle varchar(40) NOT NULL,          -- insertion|parse|binding|compile|execution|safety|parser_generation
    status varchar(30) NOT NULL,          -- pass|fail|not_applicable|unavailable|ineligible
    detail_json nvarchar(max) NULL,
    checked_at_utc datetime2(3) NOT NULL CONSTRAINT df_oracle_checked DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_case_oracle_status PRIMARY KEY (case_id, oracle)
  );
GO
