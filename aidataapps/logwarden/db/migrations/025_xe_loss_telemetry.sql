SET XACT_ABORT ON;

IF COL_LENGTH(N'telemetry.xe_pipeline_samples',N'dropped_buffers_total') IS NULL
  ALTER TABLE telemetry.xe_pipeline_samples ADD dropped_buffers_total bigint NULL;
IF COL_LENGTH(N'telemetry.xe_pipeline_samples',N'blocked_event_fire_time_ms') IS NULL
  ALTER TABLE telemetry.xe_pipeline_samples ADD blocked_event_fire_time_ms bigint NULL;
IF COL_LENGTH(N'telemetry.xe_pipeline_samples',N'failed_target_buffers_total') IS NULL
  ALTER TABLE telemetry.xe_pipeline_samples ADD failed_target_buffers_total bigint NULL;
