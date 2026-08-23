-- logwarden:migration-mode=non-transactional
-- The governing retrieval contract indexes both heading and content. Migration
-- 011 created the initial content-only feasibility index before runbooks existed.
SET XACT_ABORT ON;

IF EXISTS (SELECT 1 FROM sys.fulltext_indexes WHERE object_id=OBJECT_ID(N'kb.runbook_chunks'))
   AND NOT EXISTS
   (
     SELECT 1 FROM sys.fulltext_index_columns
     WHERE object_id=OBJECT_ID(N'kb.runbook_chunks')
       AND column_id=COLUMNPROPERTY(OBJECT_ID(N'kb.runbook_chunks'),N'heading_path','ColumnId')
   )
  DROP FULLTEXT INDEX ON kb.runbook_chunks;

IF NOT EXISTS (SELECT 1 FROM sys.fulltext_indexes WHERE object_id=OBJECT_ID(N'kb.runbook_chunks'))
  CREATE FULLTEXT INDEX ON kb.runbook_chunks
    (heading_path LANGUAGE 1033, content LANGUAGE 1033)
    KEY INDEX pk_kb_runbook_chunks ON logwarden_runbooks_fts
    WITH CHANGE_TRACKING AUTO, STOPLIST = SYSTEM;
