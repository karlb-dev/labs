SET XACT_ABORT ON;

IF OBJECT_ID(N'eval.ground_truth_episodes', N'U') IS NULL
BEGIN
  CREATE TABLE eval.ground_truth_episodes
  (
    episode_id varchar(120) NOT NULL CONSTRAINT pk_eval_ground_truth PRIMARY KEY,
    scenario_variant_id varchar(120) NOT NULL,
    scenario_group_id varchar(100) NOT NULL,
    family varchar(80) NOT NULL,
    regime char(1) NOT NULL,
    split_role varchar(40) NOT NULL,
    expected_class varchar(80) NOT NULL,
    expected_severity varchar(24) NOT NULL,
    expected_action varchar(100) NOT NULL,
    acceptable_actions_json nvarchar(max) NOT NULL,
    should_abstain bit NOT NULL,
    expected_runbooks_json nvarchar(max) NOT NULL,
    protected_truth_json nvarchar(max) NOT NULL,
    truth_sha256 char(64) NOT NULL CONSTRAINT uq_eval_ground_truth_hash UNIQUE,
    frozen_at_utc datetime2(7) NOT NULL,
    CONSTRAINT fk_eval_truth_packet FOREIGN KEY(episode_id) REFERENCES ingest.incident_packets(episode_id),
    CONSTRAINT fk_eval_truth_variant FOREIGN KEY(scenario_variant_id) REFERENCES workload.scenario_variants(scenario_variant_id),
    CONSTRAINT ck_eval_truth_regime CHECK (regime IN ('K','C','U','M','N')),
    CONSTRAINT ck_eval_truth_actions_json CHECK (ISJSON(acceptable_actions_json) = 1),
    CONSTRAINT ck_eval_truth_runbooks_json CHECK (ISJSON(expected_runbooks_json) = 1),
    CONSTRAINT ck_eval_truth_protected_json CHECK (ISJSON(protected_truth_json) = 1)
  );
  CREATE INDEX ix_eval_truth_group ON eval.ground_truth_episodes(split_role, scenario_group_id, family, regime, episode_id);
END;

IF OBJECT_ID(N'eval.predictions', N'U') IS NULL
BEGIN
  CREATE TABLE eval.predictions
  (
    prediction_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_eval_predictions PRIMARY KEY,
    job_id bigint NOT NULL CONSTRAINT uq_eval_predictions_job UNIQUE,
    agent_run_id bigint NULL,
    decision_id bigint NULL,
    episode_id varchar(120) NOT NULL,
    model_profile_id varchar(80) NULL,
    agent_arm_id varchar(80) NOT NULL,
    prediction_source varchar(32) NOT NULL,
    predicted_class varchar(80) NULL,
    predicted_severity varchar(24) NULL,
    predicted_action varchar(100) NULL,
    confidence decimal(9,6) NULL,
    abstained bit NULL,
    outcome varchar(32) NOT NULL,
    failure_stage varchar(80) NULL,
    failure_class varchar(80) NULL,
    eligible bit NOT NULL CONSTRAINT df_eval_predictions_eligible DEFAULT(1),
    complete_case_eligible bit NOT NULL,
    prediction_json nvarchar(max) NOT NULL,
    created_at_utc datetime2(7) NOT NULL CONSTRAINT df_eval_predictions_created DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_eval_predictions_job FOREIGN KEY(job_id) REFERENCES control.jobs(job_id),
    CONSTRAINT fk_eval_predictions_agent_run FOREIGN KEY(agent_run_id) REFERENCES agent.agent_runs(agent_run_id),
    CONSTRAINT fk_eval_predictions_decision FOREIGN KEY(decision_id) REFERENCES agent.decisions(decision_id),
    CONSTRAINT fk_eval_predictions_truth FOREIGN KEY(episode_id) REFERENCES eval.ground_truth_episodes(episode_id),
    CONSTRAINT fk_eval_predictions_model FOREIGN KEY(model_profile_id) REFERENCES control.model_profiles(model_profile_id),
    CONSTRAINT fk_eval_predictions_arm FOREIGN KEY(agent_arm_id) REFERENCES control.agent_arms(agent_arm_id),
    CONSTRAINT ck_eval_predictions_source CHECK (prediction_source IN ('agent','B0','B1','B2','B3','derived_router')),
    CONSTRAINT ck_eval_predictions_outcome CHECK (outcome IN ('decision','abstention','failure')),
    CONSTRAINT ck_eval_predictions_confidence CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    CONSTRAINT ck_eval_predictions_json CHECK (ISJSON(prediction_json) = 1)
  );
  CREATE INDEX ix_eval_predictions_cell ON eval.predictions(model_profile_id, agent_arm_id, episode_id, prediction_id);
END;

IF OBJECT_ID(N'eval.tool_expectations', N'U') IS NULL
BEGIN
  CREATE TABLE eval.tool_expectations
  (
    episode_id varchar(120) NOT NULL,
    tool_name varchar(80) NOT NULL,
    expectation varchar(24) NOT NULL,
    acceptable_args_json nvarchar(max) NOT NULL,
    reason nvarchar(1000) NOT NULL,
    CONSTRAINT pk_eval_tool_expectations PRIMARY KEY(episode_id, tool_name),
    CONSTRAINT fk_eval_tool_expectations_truth FOREIGN KEY(episode_id) REFERENCES eval.ground_truth_episodes(episode_id),
    CONSTRAINT ck_eval_tool_expectation CHECK (expectation IN ('required','optional','forbidden')),
    CONSTRAINT ck_eval_tool_expectation_json CHECK (ISJSON(acceptable_args_json) = 1)
  );
END;

IF OBJECT_ID(N'eval.tool_scores', N'U') IS NULL
BEGIN
  CREATE TABLE eval.tool_scores
  (
    prediction_id bigint NOT NULL,
    tool_name varchar(80) NOT NULL,
    expectation varchar(24) NOT NULL,
    called_count int NOT NULL,
    valid_call_count int NOT NULL,
    snapshot_miss_count int NOT NULL,
    tool_error_count int NOT NULL,
    correct_use bit NOT NULL,
    score decimal(9,6) NOT NULL,
    detail_json nvarchar(max) NOT NULL,
    CONSTRAINT pk_eval_tool_scores PRIMARY KEY(prediction_id, tool_name),
    CONSTRAINT fk_eval_tool_scores_prediction FOREIGN KEY(prediction_id) REFERENCES eval.predictions(prediction_id),
    CONSTRAINT ck_eval_tool_score CHECK (score BETWEEN 0 AND 1),
    CONSTRAINT ck_eval_tool_scores_counts CHECK (called_count >= 0 AND valid_call_count >= 0 AND snapshot_miss_count >= 0 AND tool_error_count >= 0),
    CONSTRAINT ck_eval_tool_scores_json CHECK (ISJSON(detail_json) = 1)
  );
END;

IF OBJECT_ID(N'eval.retrieval_scores', N'U') IS NULL
BEGIN
  CREATE TABLE eval.retrieval_scores
  (
    prediction_id bigint NOT NULL CONSTRAINT pk_eval_retrieval_scores PRIMARY KEY,
    retrieval_run_id bigint NULL,
    answerable bit NOT NULL,
    retrieved_any bit NOT NULL,
    recall_at_k decimal(9,6) NULL,
    reciprocal_rank decimal(9,6) NULL,
    ndcg_at_k decimal(9,6) NULL,
    no_answer_correct bit NULL,
    citation_precision decimal(9,6) NULL,
    citation_recall decimal(9,6) NULL,
    grounded bit NOT NULL,
    detail_json nvarchar(max) NOT NULL,
    CONSTRAINT fk_eval_retrieval_prediction FOREIGN KEY(prediction_id) REFERENCES eval.predictions(prediction_id),
    CONSTRAINT fk_eval_retrieval_run FOREIGN KEY(retrieval_run_id) REFERENCES kb.retrieval_runs(retrieval_run_id),
    CONSTRAINT ck_eval_retrieval_json CHECK (ISJSON(detail_json) = 1)
  );
END;

IF OBJECT_ID(N'eval.decision_scores', N'U') IS NULL
BEGIN
  CREATE TABLE eval.decision_scores
  (
    prediction_id bigint NOT NULL CONSTRAINT pk_eval_decision_scores PRIMARY KEY,
    end_to_end_success bit NOT NULL,
    class_correct bit NOT NULL,
    severity_correct bit NOT NULL,
    severity_ordinal_cost decimal(9,4) NOT NULL,
    action_correct bit NOT NULL,
    abstention_correct bit NOT NULL,
    contract_success bit NOT NULL,
    required_tools_satisfied bit NOT NULL,
    forbidden_tools_avoided bit NOT NULL,
    citation_policy_satisfied bit NOT NULL,
    composite_score decimal(9,6) NOT NULL,
    detail_json nvarchar(max) NOT NULL,
    CONSTRAINT fk_eval_decision_prediction FOREIGN KEY(prediction_id) REFERENCES eval.predictions(prediction_id),
    CONSTRAINT ck_eval_decision_composite CHECK (composite_score BETWEEN 0 AND 1),
    CONSTRAINT ck_eval_decision_detail_json CHECK (ISJSON(detail_json) = 1)
  );
END;

IF OBJECT_ID(N'eval.calibration_models', N'U') IS NULL
BEGIN
  CREATE TABLE eval.calibration_models
  (
    calibration_model_id varchar(100) NOT NULL CONSTRAINT pk_eval_calibration_models PRIMARY KEY,
    model_profile_id varchar(80) NOT NULL,
    agent_arm_id varchar(80) NOT NULL,
    fit_role varchar(40) NOT NULL,
    method varchar(40) NOT NULL,
    feature_manifest_json nvarchar(max) NOT NULL,
    coefficients_json nvarchar(max) NOT NULL,
    training_row_count int NOT NULL,
    model_sha256 char(64) NOT NULL CONSTRAINT uq_eval_calibration_model_hash UNIQUE,
    fitted_at_utc datetime2(7) NOT NULL,
    CONSTRAINT fk_eval_calibration_profile FOREIGN KEY(model_profile_id) REFERENCES control.model_profiles(model_profile_id),
    CONSTRAINT fk_eval_calibration_arm FOREIGN KEY(agent_arm_id) REFERENCES control.agent_arms(agent_arm_id),
    CONSTRAINT ck_eval_calibration_features_json CHECK (ISJSON(feature_manifest_json) = 1),
    CONSTRAINT ck_eval_calibration_coefficients_json CHECK (ISJSON(coefficients_json) = 1)
  );
END;

IF OBJECT_ID(N'eval.metric_results', N'U') IS NULL
BEGIN
  CREATE TABLE eval.metric_results
  (
    metric_result_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_eval_metric_results PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    metric_name varchar(160) NOT NULL,
    model_profile_id varchar(80) NULL,
    agent_arm_id varchar(80) NULL,
    split_role varchar(40) NULL,
    family varchar(80) NULL,
    regime char(1) NULL,
    analysis_set varchar(32) NOT NULL,
    numerator float NULL,
    denominator bigint NOT NULL,
    eligible_denominator bigint NOT NULL,
    complete_case_denominator bigint NOT NULL,
    metric_value float NULL,
    standard_error float NULL,
    detail_json nvarchar(max) NOT NULL,
    computed_at_utc datetime2(7) NOT NULL CONSTRAINT df_eval_metric_results_computed DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_eval_metric_profile FOREIGN KEY(model_profile_id) REFERENCES control.model_profiles(model_profile_id),
    CONSTRAINT fk_eval_metric_arm FOREIGN KEY(agent_arm_id) REFERENCES control.agent_arms(agent_arm_id),
    CONSTRAINT ck_eval_metric_denominators CHECK (denominator >= 0 AND eligible_denominator >= 0 AND complete_case_denominator >= 0),
    CONSTRAINT ck_eval_metric_json CHECK (ISJSON(detail_json) = 1)
  );
  CREATE INDEX ix_eval_metric_cell ON eval.metric_results(run_id, metric_name, model_profile_id, agent_arm_id, split_role);
END;

IF OBJECT_ID(N'eval.bootstrap_results', N'U') IS NULL
BEGIN
  CREATE TABLE eval.bootstrap_results
  (
    bootstrap_result_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_eval_bootstrap_results PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    contrast_id varchar(160) NOT NULL,
    metric_name varchar(160) NOT NULL,
    left_cell_json nvarchar(max) NOT NULL,
    right_cell_json nvarchar(max) NOT NULL,
    grouped_by varchar(80) NOT NULL,
    paired bit NOT NULL,
    repetitions int NOT NULL,
    seed bigint NOT NULL,
    observed_difference float NOT NULL,
    confidence_level float NOT NULL,
    ci_low float NOT NULL,
    ci_high float NOT NULL,
    p_value float NULL,
    sample_count int NOT NULL,
    result_sha256 char(64) NOT NULL CONSTRAINT uq_eval_bootstrap_hash UNIQUE,
    CONSTRAINT uq_eval_bootstrap_contrast UNIQUE(run_id, contrast_id, metric_name),
    CONSTRAINT ck_eval_bootstrap_left_json CHECK (ISJSON(left_cell_json) = 1),
    CONSTRAINT ck_eval_bootstrap_right_json CHECK (ISJSON(right_cell_json) = 1),
    CONSTRAINT ck_eval_bootstrap_repetitions CHECK (repetitions > 0),
    CONSTRAINT ck_eval_bootstrap_confidence CHECK (confidence_level > 0 AND confidence_level < 1)
  );
END;

IF OBJECT_ID(N'eval.permutation_results', N'U') IS NULL
BEGIN
  CREATE TABLE eval.permutation_results
  (
    permutation_result_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_eval_permutation_results PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    permutation_id varchar(160) NOT NULL,
    metric_name varchar(160) NOT NULL,
    permutations int NOT NULL,
    seed bigint NOT NULL,
    observed_value float NOT NULL,
    null_mean float NOT NULL,
    null_sd float NOT NULL,
    p_value float NOT NULL,
    detail_json nvarchar(max) NOT NULL,
    CONSTRAINT uq_eval_permutation UNIQUE(run_id, permutation_id, metric_name),
    CONSTRAINT ck_eval_permutation_count CHECK (permutations > 0),
    CONSTRAINT ck_eval_permutation_p CHECK (p_value BETWEEN 0 AND 1),
    CONSTRAINT ck_eval_permutation_json CHECK (ISJSON(detail_json) = 1)
  );
END;

IF OBJECT_ID(N'eval.taxonomy_assignments', N'U') IS NULL
BEGIN
  CREATE TABLE eval.taxonomy_assignments
  (
    taxonomy_assignment_id bigint IDENTITY(1,1) NOT NULL CONSTRAINT pk_eval_taxonomy PRIMARY KEY,
    prediction_id bigint NULL,
    run_id varchar(120) NOT NULL,
    taxonomy_code varchar(100) NOT NULL,
    stage varchar(80) NOT NULL,
    severity varchar(24) NOT NULL,
    evidence_json nvarchar(max) NOT NULL,
    assigned_at_utc datetime2(7) NOT NULL CONSTRAINT df_eval_taxonomy_assigned DEFAULT SYSUTCDATETIME(),
    CONSTRAINT fk_eval_taxonomy_prediction FOREIGN KEY(prediction_id) REFERENCES eval.predictions(prediction_id),
    CONSTRAINT ck_eval_taxonomy_json CHECK (ISJSON(evidence_json) = 1)
  );
  CREATE INDEX ix_eval_taxonomy_run ON eval.taxonomy_assignments(run_id, taxonomy_code, taxonomy_assignment_id);
END;

IF OBJECT_ID(N'eval.claims', N'U') IS NULL
BEGIN
  CREATE TABLE eval.claims
  (
    claim_id varchar(100) NOT NULL CONSTRAINT pk_eval_claims PRIMARY KEY,
    run_id varchar(120) NOT NULL,
    claim_text nvarchar(2000) NOT NULL,
    evidence_tag varchar(40) NOT NULL,
    status varchar(32) NOT NULL,
    support_query nvarchar(max) NOT NULL,
    support_artifacts_json nvarchar(max) NOT NULL,
    limitations_json nvarchar(max) NOT NULL,
    claim_sha256 char(64) NOT NULL CONSTRAINT uq_eval_claim_hash UNIQUE,
    updated_at_utc datetime2(7) NOT NULL CONSTRAINT df_eval_claim_updated DEFAULT SYSUTCDATETIME(),
    CONSTRAINT ck_eval_claim_tag CHECK (evidence_tag IN ('OBS','PAIRED','CAUSAL-APPLICATION','SYSTEMS','AUDIT','HARNESS','DERIVED')),
    CONSTRAINT ck_eval_claim_status CHECK (status IN ('planned','supported','qualified','rejected','unavailable')),
    CONSTRAINT ck_eval_claim_artifacts_json CHECK (ISJSON(support_artifacts_json) = 1),
    CONSTRAINT ck_eval_claim_limitations_json CHECK (ISJSON(limitations_json) = 1)
  );
END;
