SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE kb.usp_search_runbooks
  @query nvarchar(1000),
  @top_k int = 5,
  @corpus_id varchar(80) = 'primary-v1',
  @retrieval_mode varchar(40) = 'lexical_fulltext',
  @embedding_profile_id varchar(80) = NULL,
  @query_embedding vector(1024) = NULL,
  @run_id varchar(120) = NULL,
  @episode_id varchar(120) = NULL,
  @turn_id bigint = NULL,
  @query_sha256 char(64) = NULL
AS
BEGIN
  SET NOCOUNT ON;
  IF NULLIF(LTRIM(RTRIM(@query)),N'') IS NULL THROW 51701, 'query must not be empty', 1;
  IF @top_k NOT BETWEEN 1 AND 20 THROW 51702, 'top_k must be between 1 and 20', 1;
  IF NOT EXISTS (SELECT 1 FROM kb.search_corpora WHERE corpus_id=@corpus_id AND corpus_kind='primary')
    THROW 51703, 'unknown or non-primary corpus', 1;
  IF @retrieval_mode NOT IN ('lexical_fulltext','vector_exact','hybrid_rrf')
    THROW 51810, 'unsupported retrieval mode', 1;
  IF @retrieval_mode IN ('vector_exact','hybrid_rrf')
  BEGIN
    IF @query_embedding IS NULL THROW 51811, 'query embedding is required for vector retrieval', 1;
    IF @embedding_profile_id IS NULL OR NOT EXISTS
      (SELECT 1 FROM control.embedding_profiles WHERE embedding_profile_id=@embedding_profile_id AND dimensions=1024)
      THROW 51812, 'unknown or incompatible embedding profile', 1;
    IF EXISTS
    (
      SELECT 1 FROM kb.runbook_chunks AS chunk
      INNER JOIN kb.runbooks AS book ON book.runbook_id=chunk.runbook_id
      LEFT JOIN kb.chunk_embeddings AS embedding
        ON embedding.chunk_id=chunk.chunk_id AND embedding.embedding_profile_id=@embedding_profile_id
      WHERE chunk.corpus_id=@corpus_id AND book.enabled=1 AND embedding.chunk_id IS NULL
    ) THROW 51813, 'the selected corpus has incomplete embeddings', 1;
    DECLARE @query_norm float=CONVERT(float,VECTOR_NORM(@query_embedding,'norm2'));
    IF @query_norm NOT BETWEEN 0.9999 AND 1.0001 THROW 51814, 'query embedding is not unit normalized', 1;
  END;
  IF @run_id IS NOT NULL
  BEGIN
    IF NOT EXISTS (SELECT 1 FROM control.runs WHERE run_id=@run_id) THROW 51815, 'unknown run identity', 1;
    IF @query_sha256 IS NULL OR @query_sha256 LIKE '%[^0-9a-f]%' THROW 51816, 'valid lowercase query hash required for retained retrieval', 1;
  END;

  DECLARE @started_at_utc datetime2(7)=SYSUTCDATETIME();
  DECLARE @retrieval_run_id bigint=NULL;
  DECLARE @candidate_k int=50;
  DECLARE @rrf_k int=60;
  IF @run_id IS NOT NULL
  BEGIN
    INSERT kb.retrieval_runs
      (run_id,episode_id,turn_id,corpus_id,embedding_profile_id,retrieval_mode,
       query_text,query_sha256,query_embedding,requested_k,candidate_count,
       started_at_utc,status)
    VALUES
      (@run_id,@episode_id,@turn_id,@corpus_id,
       CASE WHEN @retrieval_mode='lexical_fulltext' THEN NULL ELSE @embedding_profile_id END,
       @retrieval_mode,@query,@query_sha256,
       CASE WHEN @retrieval_mode='lexical_fulltext' THEN NULL ELSE @query_embedding END,
       @top_k,0,@started_at_utc,'running');
    SET @retrieval_run_id=SCOPE_IDENTITY();
  END;

  BEGIN TRY
    DECLARE @candidates TABLE
    (
      rank_ordinal int NOT NULL PRIMARY KEY,
      chunk_id varchar(120) NOT NULL UNIQUE,
      runbook_id varchar(100) NOT NULL,
      lexical_rank int NULL,
      vector_rank int NULL,
      lexical_score float NULL,
      vector_distance float NULL,
      fused_score float NOT NULL
    );

    ;WITH lexical_base AS
    (
      SELECT hit.[KEY] AS chunk_id,CONVERT(float,hit.[RANK]) AS lexical_score
      FROM FREETEXTTABLE(kb.runbook_chunks,(heading_path,content),@query,LANGUAGE 1033,@candidate_k) AS hit
      INNER JOIN kb.runbook_chunks AS chunk ON chunk.chunk_id=hit.[KEY]
      INNER JOIN kb.runbooks AS book ON book.runbook_id=chunk.runbook_id
      WHERE @retrieval_mode IN ('lexical_fulltext','hybrid_rrf')
        AND chunk.corpus_id=@corpus_id AND book.enabled=1
    ),
    lexical AS
    (
      SELECT chunk_id,
        ROW_NUMBER() OVER (ORDER BY lexical_score DESC,chunk_id) AS lexical_rank,
        lexical_score
      FROM lexical_base
    ),
    vector_base AS
    (
      SELECT TOP (@candidate_k) chunk.chunk_id,
        CONVERT(float,VECTOR_DISTANCE('cosine',embedding.embedding,@query_embedding)) AS vector_distance
      FROM kb.chunk_embeddings AS embedding
      INNER JOIN kb.runbook_chunks AS chunk ON chunk.chunk_id=embedding.chunk_id
      INNER JOIN kb.runbooks AS book ON book.runbook_id=chunk.runbook_id
      WHERE @retrieval_mode IN ('vector_exact','hybrid_rrf')
        AND embedding.embedding_profile_id=@embedding_profile_id
        AND chunk.corpus_id=@corpus_id AND book.enabled=1
      ORDER BY VECTOR_DISTANCE('cosine',embedding.embedding,@query_embedding),chunk.chunk_id
    ),
    vector_ranked AS
    (
      SELECT chunk_id,
        ROW_NUMBER() OVER (ORDER BY vector_distance,chunk_id) AS vector_rank,
        vector_distance
      FROM vector_base
    ),
    fused AS
    (
      SELECT COALESCE(lexical.chunk_id,vector_ranked.chunk_id) AS chunk_id,
        lexical.lexical_rank,vector_ranked.vector_rank,
        lexical.lexical_score,vector_ranked.vector_distance,
        CONVERT(float,
          ISNULL(1.0/(@rrf_k+lexical.lexical_rank),0.0)
          + ISNULL(1.0/(@rrf_k+vector_ranked.vector_rank),0.0)) AS fused_score
      FROM lexical
      FULL OUTER JOIN vector_ranked ON vector_ranked.chunk_id=lexical.chunk_id
    ),
    ranked AS
    (
      SELECT chunk_id,lexical_rank,vector_rank,lexical_score,vector_distance,fused_score,
        ROW_NUMBER() OVER (ORDER BY fused_score DESC,chunk_id) AS rank_ordinal
      FROM fused
    )
    INSERT @candidates
      (rank_ordinal,chunk_id,runbook_id,lexical_rank,vector_rank,
       lexical_score,vector_distance,fused_score)
    SELECT CONVERT(int,ranked.rank_ordinal),ranked.chunk_id,chunk.runbook_id,
      CONVERT(int,ranked.lexical_rank),CONVERT(int,ranked.vector_rank),
      ranked.lexical_score,ranked.vector_distance,ranked.fused_score
    FROM ranked
    INNER JOIN kb.runbook_chunks AS chunk ON chunk.chunk_id=ranked.chunk_id;

    DECLARE @candidate_count int=(SELECT COUNT(*) FROM @candidates);
    IF @retrieval_run_id IS NOT NULL
    BEGIN
      INSERT kb.retrieval_results
        (retrieval_run_id,rank_ordinal,chunk_id,runbook_id,lexical_rank,vector_rank,
         lexical_score,vector_distance,fused_score,returned_to_agent)
      SELECT @retrieval_run_id,rank_ordinal,chunk_id,runbook_id,lexical_rank,vector_rank,
        lexical_score,vector_distance,fused_score,
        CONVERT(bit,CASE WHEN rank_ordinal<=@top_k THEN 1 ELSE 0 END)
      FROM @candidates;
      UPDATE kb.retrieval_runs
      SET candidate_count=@candidate_count,finished_at_utc=SYSUTCDATETIME(),
          latency_ms=CONVERT(decimal(18,3),DATEDIFF_BIG(MICROSECOND,@started_at_utc,SYSUTCDATETIME())/1000.0),
          status='complete'
      WHERE retrieval_run_id=@retrieval_run_id;
    END;

    SELECT candidate.rank_ordinal,candidate.chunk_id,candidate.runbook_id,
      candidate.lexical_rank,candidate.vector_rank,candidate.lexical_score,
      candidate.vector_distance,candidate.fused_score,
      @retrieval_mode AS execution_mode,@retrieval_run_id AS retrieval_run_id,
      chunk.heading_path,chunk.content
    FROM @candidates AS candidate
    INNER JOIN kb.runbook_chunks AS chunk ON chunk.chunk_id=candidate.chunk_id
    WHERE candidate.rank_ordinal<=@top_k
    ORDER BY candidate.rank_ordinal;
  END TRY
  BEGIN CATCH
    IF @retrieval_run_id IS NOT NULL
      UPDATE kb.retrieval_runs
      SET finished_at_utc=SYSUTCDATETIME(),
          latency_ms=CONVERT(decimal(18,3),DATEDIFF_BIG(MICROSECOND,@started_at_utc,SYSUTCDATETIME())/1000.0),
          status='failed',error_detail=ERROR_MESSAGE()
      WHERE retrieval_run_id=@retrieval_run_id;
    THROW;
  END CATCH;
END;
GO

GRANT EXECUTE ON OBJECT::kb.usp_search_runbooks TO [lw_agent_role];
