SET XACT_ABORT ON;

IF COL_LENGTH(N'eval.predictions', N'control_id') IS NULL
  ALTER TABLE eval.predictions ADD control_id varchar(80) NULL;

IF COL_LENGTH(N'eval.predictions', N'source_prediction_id') IS NULL
  ALTER TABLE eval.predictions ADD source_prediction_id bigint NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name=N'fk_eval_predictions_source_prediction')
  ALTER TABLE eval.predictions WITH CHECK ADD CONSTRAINT fk_eval_predictions_source_prediction
    FOREIGN KEY(source_prediction_id) REFERENCES eval.predictions(prediction_id);

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name=N'ck_eval_predictions_control_id')
  ALTER TABLE eval.predictions WITH CHECK ADD CONSTRAINT ck_eval_predictions_control_id
    CHECK(control_id IS NULL OR control_id IN ('error-number-mask-v1','shuffled-runbooks-v1','batching-sequential-v1'));

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name=N'ck_eval_predictions_control_source')
  ALTER TABLE eval.predictions WITH CHECK ADD CONSTRAINT ck_eval_predictions_control_source
    CHECK((control_id IS NULL AND source_prediction_id IS NULL) OR (control_id IS NOT NULL AND source_prediction_id IS NOT NULL));

IF NOT EXISTS
(
  SELECT 1 FROM sys.indexes
  WHERE object_id=OBJECT_ID(N'eval.predictions') AND name=N'ix_eval_predictions_control_cell'
)
  CREATE INDEX ix_eval_predictions_control_cell
    ON eval.predictions(control_id,model_profile_id,agent_arm_id,episode_id,prediction_id);

IF NOT EXISTS
(
  SELECT 1 FROM sys.indexes
  WHERE object_id=OBJECT_ID(N'eval.predictions') AND name=N'ux_eval_predictions_control_cell'
)
  CREATE UNIQUE INDEX ux_eval_predictions_control_cell
    ON eval.predictions(control_id,model_profile_id,agent_arm_id,episode_id)
    WHERE control_id IS NOT NULL;

IF OBJECT_ID(N'eval.control_assignments', N'U') IS NULL
BEGIN
  CREATE TABLE eval.control_assignments
  (
    run_id varchar(120) NOT NULL,
    control_id varchar(80) NOT NULL,
    episode_id varchar(120) NOT NULL,
    assignment_ordinal int NOT NULL,
    source_packet_sha256 char(64) NOT NULL,
    transformed_packet_sha256 char(64) NOT NULL,
    transform_manifest_json nvarchar(max) NOT NULL,
    transform_manifest_sha256 char(64) NOT NULL,
    created_at_utc datetime2(7) NOT NULL
      CONSTRAINT df_eval_control_assignments_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_eval_control_assignments PRIMARY KEY(run_id,control_id,episode_id),
    CONSTRAINT uq_eval_control_assignment_ordinal UNIQUE(run_id,control_id,assignment_ordinal),
    CONSTRAINT fk_eval_control_assignment_run FOREIGN KEY(run_id) REFERENCES control.runs(run_id),
    CONSTRAINT fk_eval_control_assignment_truth FOREIGN KEY(episode_id) REFERENCES eval.ground_truth_episodes(episode_id),
    CONSTRAINT ck_eval_control_assignment_json CHECK(ISJSON(transform_manifest_json)=1),
    CONSTRAINT ck_eval_control_assignment_ordinal CHECK(assignment_ordinal>=0)
  );
END;

IF OBJECT_ID(N'eval.control_comparisons', N'U') IS NULL
BEGIN
  CREATE TABLE eval.control_comparisons
  (
    control_prediction_id bigint NOT NULL CONSTRAINT pk_eval_control_comparisons PRIMARY KEY,
    primary_prediction_id bigint NOT NULL,
    raw_response_agreement bit NULL,
    decision_agreement bit NULL,
    ordered_tool_call_agreement bit NULL,
    action_score_delta decimal(9,6) NULL,
    confidence_delta decimal(9,6) NULL,
    detail_json nvarchar(max) NOT NULL,
    created_at_utc datetime2(7) NOT NULL
      CONSTRAINT df_eval_control_comparisons_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_eval_control_comparison_control FOREIGN KEY(control_prediction_id) REFERENCES eval.predictions(prediction_id),
    CONSTRAINT fk_eval_control_comparison_primary FOREIGN KEY(primary_prediction_id) REFERENCES eval.predictions(prediction_id),
    CONSTRAINT ck_eval_control_comparison_json CHECK(ISJSON(detail_json)=1)
  );
END;
GO

CREATE OR ALTER VIEW reporting.v_model_arm_scorecard
AS
SELECT
  p.model_profile_id,
  p.agent_arm_id,
  t.split_role,
  t.family,
  t.regime,
  COUNT_BIG(*) AS eligible_predictions,
  SUM(CASE WHEN p.complete_case_eligible = 1 THEN 1 ELSE 0 END) AS complete_case_predictions,
  SUM(CASE WHEN p.outcome = 'failure' THEN 1 ELSE 0 END) AS terminal_failures,
  AVG(CONVERT(float, s.end_to_end_success)) AS end_to_end_success,
  AVG(CONVERT(float, s.class_correct)) AS class_accuracy,
  AVG(CONVERT(float, s.severity_correct)) AS severity_accuracy,
  AVG(CONVERT(float, s.severity_ordinal_cost)) AS severity_ordinal_cost,
  AVG(CONVERT(float, s.action_correct)) AS action_accuracy,
  AVG(CONVERT(float, s.abstention_correct)) AS abstention_accuracy,
  AVG(CONVERT(float, s.contract_success)) AS contract_success,
  AVG(CONVERT(float, s.composite_score)) AS composite_score
FROM eval.predictions AS p
INNER JOIN eval.ground_truth_episodes AS t ON t.episode_id = p.episode_id
LEFT JOIN eval.decision_scores AS s ON s.prediction_id = p.prediction_id
WHERE p.eligible = 1 AND p.control_id IS NULL
GROUP BY p.model_profile_id, p.agent_arm_id, t.split_role, t.family, t.regime;
GO

CREATE OR ALTER VIEW reporting.v_failure_stage_funnel
AS
SELECT
  p.model_profile_id,
  p.agent_arm_id,
  t.split_role,
  COALESCE(p.failure_stage, CASE WHEN p.outcome = 'failure' THEN 'unknown_failure' ELSE 'success' END) AS failure_stage,
  COALESCE(p.failure_class, CASE WHEN p.outcome = 'failure' THEN 'unknown' ELSE 'none' END) AS failure_class,
  COUNT_BIG(*) AS episode_count
FROM eval.predictions AS p
INNER JOIN eval.ground_truth_episodes AS t ON t.episode_id = p.episode_id
WHERE p.eligible = 1 AND p.control_id IS NULL
GROUP BY p.model_profile_id, p.agent_arm_id, t.split_role,
  COALESCE(p.failure_stage, CASE WHEN p.outcome = 'failure' THEN 'unknown_failure' ELSE 'success' END),
  COALESCE(p.failure_class, CASE WHEN p.outcome = 'failure' THEN 'unknown' ELSE 'none' END);
GO

CREATE OR ALTER VIEW reporting.v_control_scorecard
AS
SELECT
  p.control_id,
  p.model_profile_id,
  p.agent_arm_id,
  t.split_role,
  t.family,
  t.regime,
  COUNT_BIG(*) AS eligible_predictions,
  SUM(CASE WHEN p.outcome='failure' THEN 1 ELSE 0 END) AS terminal_failures,
  AVG(CONVERT(float,score.end_to_end_success)) AS end_to_end_success,
  AVG(CONVERT(float,score.class_correct)) AS class_accuracy,
  AVG(CONVERT(float,score.action_correct)) AS action_accuracy,
  AVG(CONVERT(float,comparison.raw_response_agreement)) AS raw_response_agreement,
  AVG(CONVERT(float,comparison.decision_agreement)) AS decision_agreement,
  AVG(CONVERT(float,comparison.ordered_tool_call_agreement)) AS ordered_tool_call_agreement
FROM eval.predictions p
INNER JOIN eval.ground_truth_episodes t ON t.episode_id=p.episode_id
LEFT JOIN eval.decision_scores score ON score.prediction_id=p.prediction_id
LEFT JOIN eval.control_comparisons comparison ON comparison.control_prediction_id=p.prediction_id
WHERE p.eligible=1 AND p.control_id IS NOT NULL
GROUP BY p.control_id,p.model_profile_id,p.agent_arm_id,t.split_role,t.family,t.regime;
GO

CREATE OR ALTER VIEW reporting.v_tool_scorecard
AS
SELECT
  p.model_profile_id,p.agent_arm_id,t.split_role,s.tool_name,s.expectation,
  COUNT_BIG(*) AS episode_tool_rows,
  SUM(CONVERT(bigint,s.called_count)) AS calls,
  SUM(CONVERT(bigint,s.valid_call_count)) AS valid_calls,
  SUM(CONVERT(bigint,s.snapshot_miss_count)) AS snapshot_misses,
  SUM(CONVERT(bigint,s.tool_error_count)) AS tool_errors,
  AVG(CONVERT(float,s.correct_use)) AS correct_use_rate,
  AVG(CONVERT(float,s.score)) AS tool_score
FROM eval.tool_scores s
INNER JOIN eval.predictions p ON p.prediction_id=s.prediction_id
INNER JOIN eval.ground_truth_episodes t ON t.episode_id=p.episode_id
WHERE p.control_id IS NULL
GROUP BY p.model_profile_id,p.agent_arm_id,t.split_role,s.tool_name,s.expectation;
GO

CREATE OR ALTER VIEW reporting.v_retrieval_scorecard
AS
SELECT
  p.model_profile_id,p.agent_arm_id,t.split_role,COUNT_BIG(*) AS prediction_count,
  AVG(CONVERT(float,s.recall_at_k)) AS recall_at_k,
  AVG(CONVERT(float,s.reciprocal_rank)) AS mean_reciprocal_rank,
  AVG(CONVERT(float,s.ndcg_at_k)) AS ndcg_at_k,
  AVG(CONVERT(float,s.no_answer_correct)) AS no_answer_accuracy,
  AVG(CONVERT(float,s.citation_precision)) AS citation_precision,
  AVG(CONVERT(float,s.citation_recall)) AS citation_recall,
  AVG(CONVERT(float,s.grounded)) AS grounding_rate
FROM eval.retrieval_scores s
INNER JOIN eval.predictions p ON p.prediction_id=s.prediction_id
INNER JOIN eval.ground_truth_episodes t ON t.episode_id=p.episode_id
WHERE p.control_id IS NULL
GROUP BY p.model_profile_id,p.agent_arm_id,t.split_role;
GO

CREATE OR ALTER VIEW reporting.v_calibration_rows
AS
SELECT
  p.prediction_id,p.model_profile_id,p.agent_arm_id,t.split_role,t.family,t.regime,
  p.confidence,p.abstained,p.outcome,s.end_to_end_success,s.class_correct,s.action_correct,
  CONVERT(bit,CASE WHEN p.prediction_source='B1' THEN 1 ELSE 0 END) AS b1_resolved,
  CONVERT(bit,0) AS b1_agrees_with_model
FROM eval.predictions p
INNER JOIN eval.ground_truth_episodes t ON t.episode_id=p.episode_id
LEFT JOIN eval.decision_scores s ON s.prediction_id=p.prediction_id
WHERE p.eligible=1 AND p.control_id IS NULL;
