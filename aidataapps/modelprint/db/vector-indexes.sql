-- Run only after the training corpora are complete and frozen. Local SQL
-- Server 2025 currently creates the earlier DiskANN format; these tables are
-- intentionally immutable after the indexes are built.
ALTER DATABASE SCOPED CONFIGURATION SET PREVIEW_FEATURES = ON;
GO
IF NOT EXISTS (SELECT 1 FROM sys.vector_indexes v JOIN sys.indexes i ON i.object_id=v.object_id AND i.index_id=v.index_id WHERE i.name=N'vix_search_semantic')
  CREATE VECTOR INDEX vix_search_semantic ON dbo.search_semantic_train(embedding) WITH (TYPE='DISKANN', METRIC='COSINE');
GO
IF NOT EXISTS (SELECT 1 FROM sys.vector_indexes v JOIN sys.indexes i ON i.object_id=v.object_id AND i.index_id=v.index_id WHERE i.name=N'vix_search_style')
  CREATE VECTOR INDEX vix_search_style ON dbo.search_style_train(embedding) WITH (TYPE='DISKANN', METRIC='COSINE');
GO
IF NOT EXISTS (SELECT 1 FROM sys.vector_indexes v JOIN sys.indexes i ON i.object_id=v.object_id AND i.index_id=v.index_id WHERE i.name=N'vix_search_fingerprint')
  CREATE VECTOR INDEX vix_search_fingerprint ON dbo.search_fingerprint_train(embedding) WITH (TYPE='DISKANN', METRIC='COSINE');
GO
