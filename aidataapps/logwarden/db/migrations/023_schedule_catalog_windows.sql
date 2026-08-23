SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE workload.usp_build_schedule
  @campaign_id bigint,
  @schedule_name varchar(80),
  @seed bigint,
  @scenario_role_filter varchar(40) = NULL,
  @repeat_count int = 1,
  @rate_profile varchar(40) = 'smoke-fast',
  @catalog_id varchar(80) = NULL
AS
BEGIN
  SET NOCOUNT ON;
  SET XACT_ABORT ON;
  IF @repeat_count NOT BETWEEN 1 AND 1000 THROW 51500, 'repeat_count must be between 1 and 1000', 1;
  IF @rate_profile NOT IN ('smoke-fast','capture-gapped','storm') THROW 51501, 'unknown rate profile', 1;
  IF NOT EXISTS (SELECT 1 FROM control.campaigns WHERE campaign_id=@campaign_id AND status='building')
    THROW 51502, 'campaign is missing or no longer mutable', 1;

  DECLARE @selected_manifest nvarchar(max) =
  (
    SELECT v.scenario_variant_id,v.variant_group_id,v.split_role,v.variant_sha256,
           s.scenario_id,s.scenario_group_id,s.scenario_sha256,
           TRY_CONVERT(int,JSON_VALUE(s.config_json,'$.packetWindow.beforeSeconds')) AS before_seconds,
           TRY_CONVERT(int,JSON_VALUE(s.config_json,'$.packetWindow.afterSeconds')) AS after_seconds
    FROM workload.scenario_variants AS v
    INNER JOIN workload.scenario_definitions AS s ON s.scenario_id=v.scenario_id
    WHERE s.enabled=1
      AND (@scenario_role_filter IS NULL OR v.split_role=@scenario_role_filter)
      AND (@catalog_id IS NULL OR JSON_VALUE(s.config_json,'$.catalogId')=@catalog_id)
    ORDER BY v.scenario_variant_id
    FOR JSON PATH
  );
  IF @selected_manifest IS NULL OR @selected_manifest=N'[]' THROW 51503, 'schedule selection is empty', 1;

  DECLARE @config_json nvarchar(max) =
  (
    SELECT 2 AS schemaVersion,@seed AS seed,@scenario_role_filter AS scenarioRoleFilter,
           @repeat_count AS repeatCount,@rate_profile AS rateProfile,@catalog_id AS catalogId,
           'packet-window-v2' AS gapPolicyVersion,
           CONVERT(bit,CASE WHEN @rate_profile='capture-gapped' THEN 1 ELSE 0 END) AS intentionalWithinGroupOverlap,
           JSON_QUERY(@selected_manifest) AS selectedVariants
    FOR JSON PATH, WITHOUT_ARRAY_WRAPPER, INCLUDE_NULL_VALUES
  );
  DECLARE @schedule_hash char(64) = LOWER(CONVERT(char(64),HASHBYTES('SHA2_256',@config_json),2));
  DECLARE @schedule_id bigint;

  SELECT @schedule_id=schedule_id FROM workload.schedules
  WHERE campaign_id=@campaign_id AND schedule_name=@schedule_name;
  IF @schedule_id IS NOT NULL
  BEGIN
    IF NOT EXISTS (SELECT 1 FROM workload.schedules WHERE schedule_id=@schedule_id AND schedule_hash=@schedule_hash)
      THROW 51504, 'existing schedule name has a different frozen input hash', 1;
    SELECT * FROM workload.schedules WHERE schedule_id=@schedule_id;
    SELECT * FROM workload.schedule_items WHERE schedule_id=@schedule_id ORDER BY ordinal;
    RETURN;
  END;

  BEGIN TRANSACTION;
  INSERT workload.schedules(campaign_id,schedule_name,seed,rate_profile,schedule_hash,status,config_json)
  VALUES(@campaign_id,@schedule_name,@seed,@rate_profile,@schedule_hash,'building',@config_json);
  SET @schedule_id=SCOPE_IDENTITY();

  ;WITH selected AS
  (
    SELECT v.scenario_variant_id,v.variant_group_id,v.split_role,
      (COALESCE(TRY_CONVERT(int,JSON_VALUE(s.config_json,'$.packetWindow.beforeSeconds')),30)+
       COALESCE(TRY_CONVERT(int,JSON_VALUE(s.config_json,'$.packetWindow.afterSeconds')),90))*1000 AS window_ms,
      DENSE_RANK() OVER
      (ORDER BY HASHBYTES('SHA2_256',CONCAT(@seed,N':',v.variant_group_id)),v.variant_group_id)-1 AS group_ordinal
    FROM workload.scenario_variants AS v
    INNER JOIN workload.scenario_definitions AS s ON s.scenario_id=v.scenario_id
    WHERE s.enabled=1
      AND (@scenario_role_filter IS NULL OR v.split_role=@scenario_role_filter)
      AND (@catalog_id IS NULL OR JSON_VALUE(s.config_json,'$.catalogId')=@catalog_id)
  ), group_stats AS
  (
    SELECT variant_group_id,group_ordinal,MAX(window_ms) AS window_ms,
           COUNT_BIG(*)*@repeat_count AS episode_count
    FROM selected GROUP BY variant_group_id,group_ordinal
  ), group_timing AS
  (
    SELECT *,COALESCE(SUM(window_ms+(episode_count-1)*1000+1000) OVER
      (ORDER BY group_ordinal ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS group_start_ms
    FROM group_stats
  ), expanded AS
  (
    SELECT selected.*,series.value AS repeat_ordinal,timing.group_start_ms,
      HASHBYTES('SHA2_256',CONCAT(@seed,N':',selected.scenario_variant_id,N':',series.value)) AS order_hash
    FROM selected
    INNER JOIN group_timing AS timing ON timing.variant_group_id=selected.variant_group_id
    CROSS JOIN GENERATE_SERIES(0,@repeat_count-1,1) AS series
  ), positioned AS
  (
    SELECT *,ROW_NUMBER() OVER(PARTITION BY variant_group_id ORDER BY order_hash,scenario_variant_id,repeat_ordinal)-1 AS within_group_ordinal
    FROM expanded
  ), planned AS
  (
    SELECT *,CASE
      WHEN @rate_profile='capture-gapped' THEN group_start_ms+within_group_ordinal*1000
      WHEN @rate_profile='storm' THEN (ROW_NUMBER() OVER(ORDER BY group_ordinal,within_group_ordinal,order_hash)-1)*CONVERT(bigint,100)
      ELSE (ROW_NUMBER() OVER(ORDER BY group_ordinal,within_group_ordinal,order_hash)-1)*CONVERT(bigint,2500)
      END AS planned_offset_ms
    FROM positioned
  ), ordered AS
  (
    SELECT *,ROW_NUMBER() OVER(ORDER BY planned_offset_ms,order_hash,scenario_variant_id)-1 AS ordinal
    FROM planned
  )
  INSERT workload.schedule_items
    (schedule_id,job_key,scenario_variant_id,ordinal,planned_offset_ms,episode_seed,expected_role)
  SELECT @schedule_id,
    LOWER(CONVERT(char(64),HASHBYTES('SHA2_256',CONCAT(@schedule_hash,N':',scenario_variant_id,N':',repeat_ordinal)),2)),
    scenario_variant_id,ordinal,planned_offset_ms,
    CONVERT(bigint,SUBSTRING(order_hash,1,8)),split_role
  FROM ordered ORDER BY ordinal;
  COMMIT;

  SELECT * FROM workload.schedules WHERE schedule_id=@schedule_id;
  SELECT * FROM workload.schedule_items WHERE schedule_id=@schedule_id ORDER BY ordinal;
END;
