-- Candidate fixture DDL for ghost_vector_search.
-- The Lab 04 implementation must replace or verify this against the pinned SQL Server build before freeze.
-- Catalog source follows.
/*
-- connection: ghosttype / VectorLab, default schema dbo, engine: SQL Server 2025 Developer
-- inferred system query: no
-- capabilities: VECTOR type and VECTOR_DISTANCE available; approximate vector index syntax capability-gated
TABLE dbo.Documents (DocumentId bigint NOT NULL PK, Title nvarchar(240) NOT NULL, Body nvarchar(max) NOT NULL, Embedding vector(1024) NULL, MetadataJson nvarchar(max) NULL, CreatedAt datetime2 NOT NULL)
TABLE dbo.DocumentChunks (ChunkId bigint NOT NULL PK, DocumentId bigint NOT NULL FK->dbo.Documents.DocumentId, ChunkOrdinal int NOT NULL, ChunkText nvarchar(max) NOT NULL, Embedding vector(1024) NOT NULL)
TABLE dbo.SearchQueries (QueryId bigint NOT NULL PK, QueryText nvarchar(1000) NOT NULL, QueryEmbedding vector(1024) NOT NULL, CreatedAt datetime2 NOT NULL)
TABLE dbo.SearchResults (QueryId bigint NOT NULL FK->dbo.SearchQueries.QueryId, ChunkId bigint NOT NULL FK->dbo.DocumentChunks.ChunkId, Distance float NOT NULL, Rank int NOT NULL)
SYSTEM OBJECTS: sys.tables, sys.columns, sys.indexes, sys.fulltext_indexes, sys.fulltext_catalogs
FULLTEXT INDEX: dbo.Documents(Title, Body) KEY INDEX PK_Documents
*/
