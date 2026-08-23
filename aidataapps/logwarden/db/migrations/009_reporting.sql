SET XACT_ABORT ON;

IF OBJECT_ID(N'reporting.report_snapshots', N'U') IS NULL
BEGIN
  CREATE TABLE reporting.report_snapshots
  (
    report_snapshot_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_reporting_report_snapshots PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    report_name varchar(160) NOT NULL,
    report_version varchar(40) NOT NULL,
    input_set_sha256 char(64) NOT NULL,
    content_sha256 char(64) NOT NULL,
    relative_path nvarchar(500) NOT NULL,
    row_counts_json nvarchar(max) NOT NULL,
    generated_at_utc datetime2(7) NOT NULL CONSTRAINT df_reporting_snapshots_generated DEFAULT SYSUTCDATETIME(),
    CONSTRAINT uq_reporting_snapshot UNIQUE(run_id, report_name, report_version),
    CONSTRAINT ck_reporting_snapshot_counts_json CHECK (ISJSON(row_counts_json) = 1)
  );
END;
GO

CREATE OR ALTER VIEW reporting.v_run_completeness
AS
SELECT
  r.run_id,
  r.run_kind,
  r.model_profile_id,
  r.agent_arm_id,
  r.packet_role,
  r.status AS run_status,
  COUNT_BIG(j.job_id) AS job_count,
  SUM(CASE WHEN j.status IN ('complete','failed','stopped','STOP_BUDGET','STOP_PORT','STOP_DATA','STOP_SAFETY') THEN 1 ELSE 0 END) AS terminal_job_count,
  SUM(CASE WHEN j.status = 'complete' THEN 1 ELSE 0 END) AS complete_job_count,
  SUM(CASE WHEN j.status NOT IN ('complete','failed','stopped','STOP_BUDGET','STOP_PORT','STOP_DATA','STOP_SAFETY') THEN 1 ELSE 0 END) AS nonterminal_job_count,
  SUM(CASE WHEN p.prediction_id IS NOT NULL THEN 1 ELSE 0 END) AS prediction_count,
  SUM(CASE WHEN p.outcome = 'failure' THEN 1 ELSE 0 END) AS failed_prediction_count,
  MIN(j.created_at_utc) AS first_job_created_at_utc,
  MAX(j.completed_at_utc) AS last_job_completed_at_utc
FROM control.runs AS r
LEFT JOIN control.jobs AS j
  ON j.model_profile_id = r.model_profile_id
 AND j.agent_arm_id = r.agent_arm_id
 AND (r.packet_role IS NULL OR j.episode_id IS NOT NULL)
LEFT JOIN eval.predictions AS p ON p.job_id = j.job_id
GROUP BY r.run_id, r.run_kind, r.model_profile_id, r.agent_arm_id, r.packet_role, r.status;
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
WHERE p.eligible = 1
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
WHERE p.eligible = 1
GROUP BY p.model_profile_id, p.agent_arm_id, t.split_role,
  COALESCE(p.failure_stage, CASE WHEN p.outcome = 'failure' THEN 'unknown_failure' ELSE 'success' END),
  COALESCE(p.failure_class, CASE WHEN p.outcome = 'failure' THEN 'unknown' ELSE 'none' END);
GO

CREATE OR ALTER VIEW reporting.v_tool_scorecard
AS
SELECT
  p.model_profile_id,
  p.agent_arm_id,
  t.split_role,
  s.tool_name,
  s.expectation,
  COUNT_BIG(*) AS episode_tool_rows,
  SUM(CONVERT(bigint, s.called_count)) AS calls,
  SUM(CONVERT(bigint, s.valid_call_count)) AS valid_calls,
  SUM(CONVERT(bigint, s.snapshot_miss_count)) AS snapshot_misses,
  SUM(CONVERT(bigint, s.tool_error_count)) AS tool_errors,
  AVG(CONVERT(float, s.correct_use)) AS correct_use_rate,
  AVG(CONVERT(float, s.score)) AS tool_score
FROM eval.tool_scores AS s
INNER JOIN eval.predictions AS p ON p.prediction_id = s.prediction_id
INNER JOIN eval.ground_truth_episodes AS t ON t.episode_id = p.episode_id
GROUP BY p.model_profile_id, p.agent_arm_id, t.split_role, s.tool_name, s.expectation;
GO

CREATE OR ALTER VIEW reporting.v_retrieval_scorecard
AS
SELECT
  p.model_profile_id,
  p.agent_arm_id,
  t.split_role,
  COUNT_BIG(*) AS prediction_count,
  AVG(CONVERT(float, s.recall_at_k)) AS recall_at_k,
  AVG(CONVERT(float, s.reciprocal_rank)) AS mean_reciprocal_rank,
  AVG(CONVERT(float, s.ndcg_at_k)) AS ndcg_at_k,
  AVG(CONVERT(float, s.no_answer_correct)) AS no_answer_accuracy,
  AVG(CONVERT(float, s.citation_precision)) AS citation_precision,
  AVG(CONVERT(float, s.citation_recall)) AS citation_recall,
  AVG(CONVERT(float, s.grounded)) AS grounding_rate
FROM eval.retrieval_scores AS s
INNER JOIN eval.predictions AS p ON p.prediction_id = s.prediction_id
INNER JOIN eval.ground_truth_episodes AS t ON t.episode_id = p.episode_id
GROUP BY p.model_profile_id, p.agent_arm_id, t.split_role;
GO

CREATE OR ALTER VIEW reporting.v_calibration_rows
AS
SELECT
  p.prediction_id,
  p.model_profile_id,
  p.agent_arm_id,
  t.split_role,
  t.family,
  t.regime,
  p.confidence,
  p.abstained,
  p.outcome,
  s.end_to_end_success,
  s.class_correct,
  s.action_correct,
  CONVERT(bit, CASE WHEN p.prediction_source = 'B1' THEN 1 ELSE 0 END) AS b1_resolved,
  CONVERT(bit, 0) AS b1_agrees_with_model
FROM eval.predictions AS p
INNER JOIN eval.ground_truth_episodes AS t ON t.episode_id = p.episode_id
LEFT JOIN eval.decision_scores AS s ON s.prediction_id = p.prediction_id
WHERE p.eligible = 1;
GO

CREATE OR ALTER VIEW reporting.v_safety_audit
AS
SELECT
  principal_name,
  check_name,
  observed_value,
  expected_value,
  CONVERT(bit, CASE WHEN observed_value = expected_value THEN 1 ELSE 0 END) AS passed
FROM
(
  SELECT
    CAST(N'lw_agent' AS sysname) AS principal_name,
    CAST(N'sysadmin_membership' AS varchar(80)) AS check_name,
    CAST(COALESCE(IS_SRVROLEMEMBER(N'sysadmin', N'lw_agent'), 0) AS varchar(20)) AS observed_value,
    CAST(N'0' AS varchar(20)) AS expected_value
  UNION ALL
  SELECT
    CAST(N'lw_agent' AS sysname),
    CAST(N'db_owner_membership' AS varchar(80)),
    CAST(CASE WHEN IS_ROLEMEMBER(N'db_owner', N'lw_agent') = 1 THEN 1 ELSE 0 END AS varchar(20)),
    CAST(N'0' AS varchar(20))
  UNION ALL
  SELECT
    CAST(N'lw_agent' AS sysname),
    CAST(N'direct_alter_any_schema_grant' AS varchar(80)),
    CAST(CASE WHEN EXISTS
      (
        SELECT 1
        FROM sys.database_permissions AS dp
        INNER JOIN sys.database_principals AS pr ON pr.principal_id = dp.grantee_principal_id
        WHERE pr.name = N'lw_agent'
          AND dp.permission_name = N'ALTER ANY SCHEMA'
          AND dp.state IN ('G','W')
      ) THEN 1 ELSE 0 END AS varchar(20)),
    CAST(N'0' AS varchar(20))
) AS checks;
GO

CREATE OR ALTER VIEW reporting.v_claim_support
AS
SELECT
  c.claim_id,
  c.run_id,
  c.evidence_tag,
  c.status,
  c.claim_text,
  c.claim_sha256,
  (SELECT COUNT_BIG(*) FROM eval.metric_results AS m WHERE m.run_id = c.run_id) AS metric_result_count,
  (SELECT COUNT_BIG(*) FROM eval.bootstrap_results AS b WHERE b.run_id = c.run_id) AS bootstrap_result_count,
  (SELECT COUNT_BIG(*) FROM eval.taxonomy_assignments AS t WHERE t.run_id = c.run_id) AS taxonomy_assignment_count,
  c.support_artifacts_json,
  c.limitations_json
FROM eval.claims AS c;
GO

CREATE OR ALTER VIEW reporting.v_agent_span_waterfall
AS
SELECT
  r.run_id,
  r.job_id,
  r.episode_id,
  s.trace_id,
  s.span_id,
  s.parent_span_id,
  s.span_name,
  s.span_kind,
  s.started_at_utc,
  s.finished_at_utc,
  s.start_monotonic_ms,
  s.duration_ms,
  s.status,
  s.error_class,
  s.attributes_json
FROM telemetry.spans AS s
INNER JOIN telemetry.traces AS r ON r.trace_id = s.trace_id;
