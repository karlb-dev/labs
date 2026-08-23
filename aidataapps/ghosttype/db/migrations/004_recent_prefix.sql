-- Cursor offsets in the package are statement-relative: the insertion base is
-- current_statement_prefix. recent_document_prefix is retained separately as
-- document context (it participates in prompts, not in cursor math).
IF COL_LENGTH(N'dataset.cases', N'recent_prefix') IS NULL
  ALTER TABLE dataset.cases ADD recent_prefix nvarchar(max) NOT NULL CONSTRAINT df_cases_recent DEFAULT N'';
GO
