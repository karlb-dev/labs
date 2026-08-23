SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE kb.usp_search_runbooks
  @query nvarchar(1000),
  @top_k int = 5,
  @corpus_id varchar(80) = 'primary-v1'
AS
BEGIN
  SET NOCOUNT ON;
  IF NULLIF(LTRIM(RTRIM(@query)),N'') IS NULL THROW 51701, 'query must not be empty', 1;
  IF @top_k NOT BETWEEN 1 AND 20 THROW 51702, 'top_k must be between 1 and 20', 1;
  IF NOT EXISTS (SELECT 1 FROM kb.search_corpora WHERE corpus_id=@corpus_id AND corpus_kind='primary')
    THROW 51703, 'unknown or non-primary corpus', 1;
  DECLARE @candidate_k int = @top_k * 4;
  SELECT TOP (@top_k)
    chunk.chunk_id,chunk.runbook_id,chunk.heading_path,chunk.content,
    hit.[RANK] AS lexical_score,
    ROW_NUMBER() OVER (ORDER BY hit.[RANK] DESC,chunk.chunk_id) AS rank_ordinal
  FROM FREETEXTTABLE(kb.runbook_chunks,(heading_path,content),@query,LANGUAGE 1033,@candidate_k) AS hit
  INNER JOIN kb.runbook_chunks AS chunk ON chunk.chunk_id=hit.[KEY]
  INNER JOIN kb.runbooks AS book ON book.runbook_id=chunk.runbook_id
  WHERE chunk.corpus_id=@corpus_id AND book.enabled=1
  ORDER BY hit.[RANK] DESC,chunk.chunk_id;
END;
GO

GRANT EXECUTE ON OBJECT::kb.usp_search_runbooks TO [lw_agent_role];
