-- logwarden:migration-mode=non-transactional
-- SQL Server forbids CREATE FULLTEXT CATALOG in a user transaction. Every
-- statement in this migration is guarded so an interrupted apply is retryable.
SET XACT_ABORT ON;

IF FULLTEXTSERVICEPROPERTY('IsFullTextInstalled') <> 1
  THROW 51110, 'Full-text search is required by the frozen LogWarden environment', 1;

IF NOT EXISTS (SELECT 1 FROM sys.fulltext_catalogs WHERE name = N'logwarden_runbooks_fts')
  CREATE FULLTEXT CATALOG logwarden_runbooks_fts WITH ACCENT_SENSITIVITY = OFF;

IF NOT EXISTS (SELECT 1 FROM sys.fulltext_indexes WHERE object_id = OBJECT_ID(N'kb.runbook_chunks'))
  CREATE FULLTEXT INDEX ON kb.runbook_chunks(content LANGUAGE 1033)
    KEY INDEX pk_kb_runbook_chunks ON logwarden_runbooks_fts
    WITH CHANGE_TRACKING AUTO, STOPLIST = SYSTEM;
