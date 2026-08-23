SET XACT_ABORT ON;

IF OBJECT_ID(N'kb.search_corpora', N'U') IS NULL
BEGIN
  CREATE TABLE kb.search_corpora
  (
    corpus_id varchar(80) NOT NULL CONSTRAINT pk_kb_search_corpora PRIMARY KEY,
    corpus_version varchar(40) NOT NULL,
    corpus_kind varchar(32) NOT NULL,
    source_manifest_json nvarchar(max) NOT NULL,
    source_manifest_sha256 char(64) NOT NULL,
    frozen_at_utc datetime2(7) NULL,
    CONSTRAINT uq_kb_search_corpora_manifest UNIQUE(source_manifest_sha256),
    CONSTRAINT ck_kb_search_corpora_kind CHECK (corpus_kind IN ('primary','shuffled','oracle')),
    CONSTRAINT ck_kb_search_corpora_json CHECK (ISJSON(source_manifest_json) = 1)
  );
END;

IF OBJECT_ID(N'kb.runbooks', N'U') IS NULL
BEGIN
  CREATE TABLE kb.runbooks
  (
    runbook_id varchar(100) NOT NULL CONSTRAINT pk_kb_runbooks PRIMARY KEY,
    corpus_id varchar(80) NOT NULL,
    title nvarchar(300) NOT NULL,
    incident_class varchar(80) NOT NULL,
    severity_floor varchar(24) NOT NULL,
    source_uri nvarchar(1000) NULL,
    source_kind varchar(40) NOT NULL,
    source_sha256 char(64) NOT NULL,
    body_markdown nvarchar(max) NOT NULL,
    body_sha256 char(64) NOT NULL,
    metadata_json nvarchar(max) NOT NULL,
    enabled bit NOT NULL CONSTRAINT df_kb_runbooks_enabled DEFAULT(1),
    CONSTRAINT fk_kb_runbooks_corpus FOREIGN KEY(corpus_id) REFERENCES kb.search_corpora(corpus_id),
    CONSTRAINT uq_kb_runbooks_source UNIQUE(corpus_id, source_sha256),
    CONSTRAINT ck_kb_runbooks_metadata_json CHECK (ISJSON(metadata_json) = 1)
  );
  CREATE INDEX ix_kb_runbooks_class ON kb.runbooks(corpus_id, incident_class, runbook_id);
END;

IF OBJECT_ID(N'kb.runbook_chunks', N'U') IS NULL
BEGIN
  CREATE TABLE kb.runbook_chunks
  (
    chunk_id varchar(120) NOT NULL CONSTRAINT pk_kb_runbook_chunks PRIMARY KEY,
    runbook_id varchar(100) NOT NULL,
    corpus_id varchar(80) NOT NULL,
    ordinal int NOT NULL,
    heading_path nvarchar(800) NOT NULL,
    content nvarchar(max) NOT NULL,
    token_count int NOT NULL,
    chunker_version varchar(40) NOT NULL,
    content_sha256 char(64) NOT NULL,
    metadata_json nvarchar(max) NOT NULL,
    CONSTRAINT fk_kb_chunks_runbook FOREIGN KEY(runbook_id) REFERENCES kb.runbooks(runbook_id),
    CONSTRAINT fk_kb_chunks_corpus FOREIGN KEY(corpus_id) REFERENCES kb.search_corpora(corpus_id),
    CONSTRAINT uq_kb_chunks_ordinal UNIQUE(runbook_id, ordinal),
    CONSTRAINT uq_kb_chunks_hash UNIQUE(corpus_id, content_sha256),
    CONSTRAINT ck_kb_chunks_token_count CHECK (token_count > 0),
    CONSTRAINT ck_kb_chunks_metadata_json CHECK (ISJSON(metadata_json) = 1)
  );
  CREATE INDEX ix_kb_chunks_corpus ON kb.runbook_chunks(corpus_id, runbook_id, ordinal);
END;

IF OBJECT_ID(N'kb.chunk_embeddings', N'U') IS NULL
BEGIN
  CREATE TABLE kb.chunk_embeddings
  (
    chunk_id varchar(120) NOT NULL,
    embedding_profile_id varchar(80) NOT NULL,
    embedding vector(1024) NOT NULL,
    embedding_sha256 char(64) NOT NULL,
    normalized bit NOT NULL,
    generated_at_utc datetime2(7) NOT NULL CONSTRAINT df_kb_embeddings_generated DEFAULT SYSUTCDATETIME(),
    latency_ms decimal(18,3) NULL,
    CONSTRAINT pk_kb_chunk_embeddings PRIMARY KEY(chunk_id, embedding_profile_id),
    CONSTRAINT fk_kb_embeddings_chunk FOREIGN KEY(chunk_id) REFERENCES kb.runbook_chunks(chunk_id),
    CONSTRAINT fk_kb_embeddings_profile FOREIGN KEY(embedding_profile_id) REFERENCES control.embedding_profiles(embedding_profile_id)
  );
END;

IF OBJECT_ID(N'kb.retrieval_runs', N'U') IS NULL
BEGIN
  CREATE TABLE kb.retrieval_runs
  (
    retrieval_run_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_kb_retrieval_runs PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    episode_id varchar(120) NULL,
    turn_id bigint NULL,
    corpus_id varchar(80) NOT NULL,
    embedding_profile_id varchar(80) NULL,
    retrieval_mode varchar(40) NOT NULL,
    query_text nvarchar(max) NOT NULL,
    query_sha256 char(64) NOT NULL,
    query_embedding vector(1024) NULL,
    requested_k int NOT NULL,
    candidate_count int NOT NULL,
    started_at_utc datetime2(7) NOT NULL,
    finished_at_utc datetime2(7) NULL,
    latency_ms decimal(18,3) NULL,
    status varchar(24) NOT NULL,
    error_detail nvarchar(max) NULL,
    CONSTRAINT fk_kb_retrieval_corpus FOREIGN KEY(corpus_id) REFERENCES kb.search_corpora(corpus_id),
    CONSTRAINT fk_kb_retrieval_profile FOREIGN KEY(embedding_profile_id) REFERENCES control.embedding_profiles(embedding_profile_id),
    CONSTRAINT ck_kb_retrieval_mode CHECK (retrieval_mode IN ('lexical_fulltext','lexical_bm25','vector_exact','hybrid_rrf','oracle_runbook','shuffled_runbook')),
    CONSTRAINT ck_kb_retrieval_k CHECK (requested_k BETWEEN 1 AND 100)
  );
  CREATE INDEX ix_kb_retrieval_episode ON kb.retrieval_runs(run_id, episode_id, retrieval_run_id);
END;

IF OBJECT_ID(N'kb.retrieval_results', N'U') IS NULL
BEGIN
  CREATE TABLE kb.retrieval_results
  (
    retrieval_run_id bigint NOT NULL,
    rank_ordinal int NOT NULL,
    chunk_id varchar(120) NOT NULL,
    runbook_id varchar(100) NOT NULL,
    lexical_rank int NULL,
    vector_rank int NULL,
    lexical_score float NULL,
    vector_distance float NULL,
    fused_score float NOT NULL,
    returned_to_agent bit NOT NULL,
    CONSTRAINT pk_kb_retrieval_results PRIMARY KEY(retrieval_run_id, rank_ordinal),
    CONSTRAINT uq_kb_retrieval_chunk UNIQUE(retrieval_run_id, chunk_id),
    CONSTRAINT fk_kb_results_run FOREIGN KEY(retrieval_run_id) REFERENCES kb.retrieval_runs(retrieval_run_id),
    CONSTRAINT fk_kb_results_chunk FOREIGN KEY(chunk_id) REFERENCES kb.runbook_chunks(chunk_id),
    CONSTRAINT fk_kb_results_runbook FOREIGN KEY(runbook_id) REFERENCES kb.runbooks(runbook_id),
    CONSTRAINT ck_kb_results_rank CHECK (rank_ordinal > 0)
  );
END;

GO

CREATE OR ALTER PROCEDURE kb.search_runbook_exact
  @corpus_id varchar(80),
  @embedding_profile_id varchar(80),
  @query_embedding vector(1024),
  @top_k int = 8
AS
BEGIN
  SET NOCOUNT ON;
  IF @top_k NOT BETWEEN 1 AND 100 THROW 51100, 'top_k must be between 1 and 100', 1;
  SELECT TOP (@top_k)
    c.chunk_id,
    c.runbook_id,
    c.heading_path,
    c.content,
    VECTOR_DISTANCE('cosine', e.embedding, @query_embedding) AS vector_distance
  FROM kb.chunk_embeddings AS e
  INNER JOIN kb.runbook_chunks AS c ON c.chunk_id = e.chunk_id
  WHERE c.corpus_id = @corpus_id
    AND e.embedding_profile_id = @embedding_profile_id
  ORDER BY VECTOR_DISTANCE('cosine', e.embedding, @query_embedding), c.chunk_id;
END;
