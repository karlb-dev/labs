SET XACT_ABORT ON;

IF COL_LENGTH(N'agent.model_responses', N'reasoning_content') IS NULL
  ALTER TABLE agent.model_responses ADD
    reasoning_content nvarchar(max) NULL,
    reasoning_sha256 char(64) NULL,
    reasoning_bytes int NULL;

IF COL_LENGTH(N'kb.retrieval_runs', N'evaluator_only') IS NULL
  ALTER TABLE kb.retrieval_runs ADD
    evaluator_only bit NOT NULL
      CONSTRAINT df_kb_retrieval_runs_evaluator_only DEFAULT(0) WITH VALUES;

IF COL_LENGTH(N'telemetry.model_service_samples', N'source_instance') IS NULL
  ALTER TABLE telemetry.model_service_samples ADD
    source_instance varchar(160) NOT NULL
      CONSTRAINT df_telemetry_model_service_source DEFAULT('legacy-unattributed') WITH VALUES,
    phase varchar(40) NOT NULL
      CONSTRAINT df_telemetry_model_service_phase DEFAULT('legacy') WITH VALUES;
GO

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name=N'ck_agent_responses_reasoning')
  ALTER TABLE agent.model_responses WITH CHECK ADD CONSTRAINT ck_agent_responses_reasoning
    CHECK
    (
      (reasoning_content IS NULL AND reasoning_sha256 IS NULL AND reasoning_bytes IS NULL)
      OR
      (reasoning_content IS NOT NULL AND reasoning_sha256 IS NOT NULL AND reasoning_bytes>=0)
    );

IF NOT EXISTS
(
  SELECT 1 FROM sys.indexes
  WHERE object_id=OBJECT_ID(N'telemetry.model_service_samples')
    AND name=N'ix_telemetry_model_service_instance'
)
  CREATE INDEX ix_telemetry_model_service_instance
    ON telemetry.model_service_samples(run_id,source_instance,phase,sampled_at_utc);

IF OBJECT_ID(N'eval.retrieval_benchmark_results', N'U') IS NULL
BEGIN
  CREATE TABLE eval.retrieval_benchmark_results
  (
    retrieval_benchmark_result_id bigint IDENTITY(1,1) NOT NULL
      CONSTRAINT pk_eval_retrieval_benchmark_results PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    episode_id varchar(120) NOT NULL,
    split_role varchar(40) NOT NULL,
    retrieval_mode varchar(40) NOT NULL,
    retrieval_run_id bigint NULL,
    requested_k int NOT NULL,
    query_sha256 char(64) NOT NULL,
    expected_runbooks_json nvarchar(max) NOT NULL,
    returned_runbooks_json nvarchar(max) NOT NULL,
    recall_at_k decimal(9,6) NOT NULL,
    reciprocal_rank decimal(9,6) NOT NULL,
    ndcg_at_k decimal(9,6) NOT NULL,
    no_answer_correct bit NULL,
    evaluator_only bit NOT NULL,
    result_sha256 char(64) NOT NULL,
    created_at_utc datetime2(7) NOT NULL
      CONSTRAINT df_eval_retrieval_benchmark_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_eval_retrieval_benchmark_cell
      UNIQUE(run_id,episode_id,retrieval_mode),
    CONSTRAINT fk_eval_retrieval_benchmark_truth
      FOREIGN KEY(episode_id) REFERENCES eval.ground_truth_episodes(episode_id),
    CONSTRAINT fk_eval_retrieval_benchmark_run
      FOREIGN KEY(retrieval_run_id) REFERENCES kb.retrieval_runs(retrieval_run_id),
    CONSTRAINT ck_eval_retrieval_benchmark_mode CHECK
      (retrieval_mode IN ('lexical_fulltext','vector_exact','hybrid_rrf','oracle_runbook','shuffled_runbook')),
    CONSTRAINT ck_eval_retrieval_benchmark_k CHECK (requested_k BETWEEN 1 AND 20),
    CONSTRAINT ck_eval_retrieval_benchmark_expected_json CHECK (ISJSON(expected_runbooks_json)=1),
    CONSTRAINT ck_eval_retrieval_benchmark_returned_json CHECK (ISJSON(returned_runbooks_json)=1),
    CONSTRAINT ck_eval_retrieval_benchmark_metrics CHECK
      (recall_at_k BETWEEN 0 AND 1 AND reciprocal_rank BETWEEN 0 AND 1 AND ndcg_at_k BETWEEN 0 AND 1),
    CONSTRAINT ck_eval_retrieval_benchmark_evaluator CHECK (evaluator_only=1)
  );
  CREATE INDEX ix_eval_retrieval_benchmark_scorecard
    ON eval.retrieval_benchmark_results(retrieval_mode,split_role,episode_id)
    INCLUDE(recall_at_k,reciprocal_rank,ndcg_at_k);
END;
