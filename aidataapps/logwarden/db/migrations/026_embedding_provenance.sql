SET XACT_ABORT ON;

IF COL_LENGTH(N'kb.chunk_embeddings', N'input_sha256') IS NULL
  ALTER TABLE kb.chunk_embeddings ADD input_sha256 char(64) NULL;
IF COL_LENGTH(N'kb.chunk_embeddings', N'input_format_version') IS NULL
  ALTER TABLE kb.chunk_embeddings ADD input_format_version varchar(40) NULL;
IF COL_LENGTH(N'kb.chunk_embeddings', N'embedding_operation_id') IS NULL
  ALTER TABLE kb.chunk_embeddings ADD embedding_operation_id uniqueidentifier NULL;
IF COL_LENGTH(N'kb.chunk_embeddings', N'request_body_sha256') IS NULL
  ALTER TABLE kb.chunk_embeddings ADD request_body_sha256 char(64) NULL;
IF COL_LENGTH(N'kb.chunk_embeddings', N'response_body_sha256') IS NULL
  ALTER TABLE kb.chunk_embeddings ADD response_body_sha256 char(64) NULL;
IF COL_LENGTH(N'kb.chunk_embeddings', N'batch_prompt_tokens') IS NULL
  ALTER TABLE kb.chunk_embeddings ADD batch_prompt_tokens int NULL;
IF COL_LENGTH(N'kb.chunk_embeddings', N'batch_input_count') IS NULL
  ALTER TABLE kb.chunk_embeddings ADD batch_input_count int NULL;
IF COL_LENGTH(N'kb.chunk_embeddings', N'batch_ordinal') IS NULL
  ALTER TABLE kb.chunk_embeddings ADD batch_ordinal int NULL;
IF COL_LENGTH(N'kb.chunk_embeddings', N'generated_by_run_id') IS NULL
  ALTER TABLE kb.chunk_embeddings ADD generated_by_run_id varchar(120) NULL;
IF COL_LENGTH(N'kb.chunk_embeddings', N'request_artifact_path') IS NULL
  ALTER TABLE kb.chunk_embeddings ADD request_artifact_path nvarchar(1000) NULL;
IF COL_LENGTH(N'kb.chunk_embeddings', N'response_artifact_path') IS NULL
  ALTER TABLE kb.chunk_embeddings ADD response_artifact_path nvarchar(1000) NULL;

GO

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name=N'ck_kb_embeddings_prompt_tokens')
  ALTER TABLE kb.chunk_embeddings ADD CONSTRAINT ck_kb_embeddings_prompt_tokens
    CHECK (batch_prompt_tokens IS NULL OR batch_prompt_tokens >= 0);
IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name=N'ck_kb_embeddings_batch')
  ALTER TABLE kb.chunk_embeddings ADD CONSTRAINT ck_kb_embeddings_batch
    CHECK (
      (batch_input_count IS NULL AND batch_ordinal IS NULL)
      OR
      (batch_input_count > 0 AND batch_ordinal >= 0 AND batch_ordinal < batch_input_count)
    );
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name=N'fk_kb_embeddings_run')
  ALTER TABLE kb.chunk_embeddings ADD CONSTRAINT fk_kb_embeddings_run
    FOREIGN KEY(generated_by_run_id) REFERENCES control.runs(run_id);
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE object_id=OBJECT_ID(N'kb.chunk_embeddings') AND name=N'ix_kb_embeddings_operation')
  CREATE INDEX ix_kb_embeddings_operation
    ON kb.chunk_embeddings(embedding_profile_id,embedding_operation_id,batch_ordinal)
    INCLUDE(input_sha256,embedding_sha256,generated_by_run_id);
