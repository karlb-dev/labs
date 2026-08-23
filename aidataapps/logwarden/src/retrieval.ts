import { randomUUID } from "node:crypto";
import { performance } from "node:perf_hooks";
import sql from "mssql";
import { callOpenAiCompatibleEmbeddings, type EmbeddingCallEvidence } from "./embeddings.js";
import { canonicalJson, sha256 } from "./hash.js";
import type { EmbeddingProfile } from "./models.js";
import { atomicWrite } from "./run.js";
import type { FileTelemetryJournal } from "./telemetry.js";

export type RetrievalMode = "lexical_fulltext" | "vector_exact" | "hybrid_rrf";

export interface QueryEmbedding {
  query: string;
  querySha256: string;
  vector: number[];
  vectorSha256: string;
  evidence: EmbeddingCallEvidence;
}

export interface RetrievalRow {
  rankOrdinal: number;
  chunkId: string;
  runbookId: string;
  lexicalRank: number | null;
  vectorRank: number | null;
  lexicalScore: number | null;
  vectorDistance: number | null;
  fusedScore: number;
  executionMode: RetrievalMode;
  retrievalRunId: number;
  headingPath: string;
  content: string;
}

export interface RetrievalResult {
  operationId: string;
  traceId: string;
  spanId: string;
  query: string;
  querySha256: string;
  mode: RetrievalMode;
  topK: number;
  corpusId: string;
  embeddingProfileId: string | null;
  queryEmbeddingSha256: string | null;
  queryEmbeddingOperationId: string | null;
  rows: RetrievalRow[];
  resultPath: string;
  resultSha256: string;
  metadataPath: string;
  sqlElapsedMs: number;
  totalElapsedMs: number;
}

export async function createRetrievalQueryEmbedding(options: {
  endpoint: string;
  profile: EmbeddingProfile;
  query: string;
  journal: FileTelemetryJournal;
  runDirectory: string;
  context?: { traceId?: string; parentSpanId?: string; episodeId?: string; attemptId?: number; jobId?: number };
}): Promise<QueryEmbedding> {
  const query = canonicalRetrievalQuery(options.query);
  const result = await callOpenAiCompatibleEmbeddings({
    endpoint: options.endpoint,
    model: options.profile.modelId,
    inputs: [query],
    dimensions: options.profile.dimensions,
    journal: options.journal,
    runDirectory: options.runDirectory,
    ...(options.context === undefined ? {} : { context: options.context }),
  });
  if (result.status !== "success" || result.vectors === null) {
    throw new Error(`Retrieval query embedding failed: ${result.evidence.errorClass}: ${result.evidence.errorDetail}`);
  }
  const vector = result.vectors[0]!;
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (Math.abs(norm - 1) > 0.0001) throw new Error(`Retrieval query embedding has invalid L2 norm ${norm}`);
  return {
    query,
    querySha256: sha256(query),
    vector,
    vectorSha256: result.evidence.vectorSha256[0]!,
    evidence: result.evidence,
  };
}

export async function executeRunbookRetrieval(options: {
  pool: sql.ConnectionPool;
  runId: string;
  episodeId?: string;
  turnId?: number;
  query: string;
  topK: number;
  corpusId: string;
  mode: RetrievalMode;
  profile: EmbeddingProfile;
  embeddingEndpoint: string;
  queryEmbedding?: QueryEmbedding;
  journal: FileTelemetryJournal;
  runDirectory: string;
  context?: { traceId?: string; parentSpanId?: string; episodeId?: string; attemptId?: number; jobId?: number };
}): Promise<RetrievalResult> {
  const operationId = randomUUID();
  const query = canonicalRetrievalQuery(options.query);
  if (!Number.isSafeInteger(options.topK) || options.topK < 1 || options.topK > 20) throw new Error("Retrieval topK must be between 1 and 20");
  if (!/^[A-Za-z][A-Za-z0-9_-]{0,79}$/.test(options.corpusId)) throw new Error("Invalid retrieval corpus ID");
  const querySha256 = sha256(query);
  if (options.queryEmbedding !== undefined && (options.queryEmbedding.query !== query || options.queryEmbedding.querySha256 !== querySha256)) {
    throw new Error("Provided query embedding does not match the canonical retrieval query");
  }
  const root = await options.journal.startSpan("retrieval.operation", options.context ?? {}, {
    operationId,
    mode: options.mode,
    querySha256,
    topK: options.topK,
    corpusId: options.corpusId,
    embeddingProfileId: options.mode === "lexical_fulltext" ? null : options.profile.key,
  });
  const started = performance.now();
  let queryEmbedding = options.queryEmbedding;
  try {
    if (options.mode !== "lexical_fulltext" && queryEmbedding === undefined) {
      queryEmbedding = await createRetrievalQueryEmbedding({
        endpoint: options.embeddingEndpoint,
        profile: options.profile,
        query,
        journal: options.journal,
        runDirectory: options.runDirectory,
        context: { ...(options.context ?? {}), traceId: root.traceId, parentSpanId: root.spanId },
      });
    }
    const embeddingJson = queryEmbedding === undefined ? null : canonicalJson(queryEmbedding.vector);
    const sqlStarted = performance.now();
    const response = await options.pool.request()
      .input("query", sql.NVarChar(1000), query)
      .input("top_k", sql.Int, options.topK)
      .input("corpus_id", sql.VarChar(80), options.corpusId)
      .input("retrieval_mode", sql.VarChar(40), options.mode)
      .input("embedding_profile_id", sql.VarChar(80), options.mode === "lexical_fulltext" ? null : options.profile.key)
      .input("embedding_json", sql.NVarChar(sql.MAX), embeddingJson)
      .input("run_id", sql.VarChar(120), options.runId)
      .input("episode_id", sql.VarChar(120), options.episodeId ?? null)
      .input("turn_id", sql.BigInt, options.turnId ?? null)
      .input("query_sha256", sql.Char(64), querySha256)
      .batch(`
        DECLARE @query_vector vector(1024)=NULL;
        IF @embedding_json IS NOT NULL SET @query_vector=CAST(@embedding_json AS vector(1024));
        EXEC kb.usp_search_runbooks
          @query=@query,@top_k=@top_k,@corpus_id=@corpus_id,
          @retrieval_mode=@retrieval_mode,@embedding_profile_id=@embedding_profile_id,
          @query_embedding=@query_vector,@run_id=@run_id,@episode_id=@episode_id,
          @turn_id=@turn_id,@query_sha256=@query_sha256;
      `);
    const sqlElapsedMs = round(performance.now() - sqlStarted);
    const rawRows = JSON.parse(JSON.stringify(response.recordset)) as unknown[];
    const resultPath = `${options.runDirectory}/raw/retrieval/${operationId}/result.json`;
    const resultBytes = Buffer.from(canonicalJson(rawRows));
    await atomicWrite(resultPath, resultBytes);
    await options.journal.record("point", "retrieval.result.durable", {
      ...(options.context ?? {}),
      traceId: root.traceId,
      parentSpanId: root.spanId,
    }, {
      operationId,
      mode: options.mode,
      path: resultPath,
      bytes: resultBytes.length,
      sha256: sha256(resultBytes),
      sqlElapsedMs,
    });
    const rows = validateRetrievalRows(rawRows, options.mode, options.topK);
    const totalElapsedMs = round(performance.now() - started);
    const metadataPath = `${options.runDirectory}/raw/retrieval/${operationId}/metadata.json`;
    const metadata = {
      schemaVersion: 1,
      operationId,
      traceId: root.traceId,
      spanId: root.spanId,
      runId: options.runId,
      episodeId: options.episodeId ?? null,
      turnId: options.turnId ?? null,
      query,
      querySha256,
      mode: options.mode,
      topK: options.topK,
      corpusId: options.corpusId,
      embeddingProfileId: options.mode === "lexical_fulltext" ? null : options.profile.key,
      queryEmbeddingSha256: queryEmbedding?.vectorSha256 ?? null,
      queryEmbeddingOperationId: queryEmbedding?.evidence.operationId ?? null,
      resultPath,
      resultSha256: sha256(resultBytes),
      resultBytes: resultBytes.length,
      returnedRows: rows.length,
      retrievalRunId: rows[0]?.retrievalRunId ?? null,
      sqlElapsedMs,
      totalElapsedMs,
      status: "success",
    };
    await atomicWrite(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);
    await options.journal.record("point", "state.retrieval_validated", {
      ...(options.context ?? {}),
      traceId: root.traceId,
      parentSpanId: root.spanId,
    }, {
      operationId,
      mode: options.mode,
      querySha256,
      retrievalRunId: rows[0]?.retrievalRunId ?? null,
      returnedRows: rows.length,
      resultSha256: sha256(resultBytes),
      metadataPath,
    }, { status: "success" });
    await options.journal.endSpan(root, "success", {
      operationId,
      mode: options.mode,
      returnedRows: rows.length,
      sqlElapsedMs,
      totalElapsedMs,
      queryEmbeddingOperationId: queryEmbedding?.evidence.operationId ?? null,
    });
    await options.journal.flush();
    return {
      operationId,
      traceId: root.traceId,
      spanId: root.spanId,
      query,
      querySha256,
      mode: options.mode,
      topK: options.topK,
      corpusId: options.corpusId,
      embeddingProfileId: options.mode === "lexical_fulltext" ? null : options.profile.key,
      queryEmbeddingSha256: queryEmbedding?.vectorSha256 ?? null,
      queryEmbeddingOperationId: queryEmbedding?.evidence.operationId ?? null,
      rows,
      resultPath,
      resultSha256: sha256(resultBytes),
      metadataPath,
      sqlElapsedMs,
      totalElapsedMs,
    };
  } catch (error) {
    const errorDetail = (error as { message?: string }).message ?? String(error);
    await options.journal.record("point", "state.retrieval_failed", {
      ...(options.context ?? {}),
      traceId: root.traceId,
      parentSpanId: root.spanId,
    }, { operationId, mode: options.mode, querySha256, errorDetail }, { status: "failed" }).catch(() => undefined);
    await options.journal.endSpan(root, "failed", { operationId, mode: options.mode, errorDetail }).catch(() => undefined);
    await options.journal.flush().catch(() => undefined);
    throw error;
  }
}

export function canonicalRetrievalQuery(value: string): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length < 1 || normalized.length > 1000) throw new Error("Retrieval query must contain 1..1000 normalized characters");
  return normalized;
}

export function validateRetrievalRows(rawRows: unknown[], mode: RetrievalMode, topK: number): RetrievalRow[] {
  if (rawRows.length > topK) throw new Error(`Retrieval returned ${rawRows.length} rows for topK=${topK}`);
  const rows = rawRows.map((raw, index): RetrievalRow => {
    if (raw === null || typeof raw !== "object" || Array.isArray(raw)) throw new Error(`Retrieval row ${index} is not an object`);
    const value = raw as Record<string, unknown>;
    const row: RetrievalRow = {
      rankOrdinal: positiveInteger(value.rank_ordinal, `row ${index} rank`),
      chunkId: requiredString(value.chunk_id, `row ${index} chunk_id`),
      runbookId: requiredString(value.runbook_id, `row ${index} runbook_id`),
      lexicalRank: nullablePositiveInteger(value.lexical_rank, `row ${index} lexical_rank`),
      vectorRank: nullablePositiveInteger(value.vector_rank, `row ${index} vector_rank`),
      lexicalScore: nullableFiniteNumber(value.lexical_score, `row ${index} lexical_score`),
      vectorDistance: nullableFiniteNumber(value.vector_distance, `row ${index} vector_distance`),
      fusedScore: finiteNumber(value.fused_score, `row ${index} fused_score`),
      executionMode: requiredString(value.execution_mode, `row ${index} execution_mode`) as RetrievalMode,
      retrievalRunId: positiveInteger(value.retrieval_run_id, `row ${index} retrieval_run_id`),
      headingPath: requiredString(value.heading_path, `row ${index} heading_path`),
      content: requiredString(value.content, `row ${index} content`),
    };
    if (row.executionMode !== mode) throw new Error(`Retrieval row ${index} execution mode drift`);
    if (mode === "lexical_fulltext" && (row.lexicalRank === null || row.lexicalScore === null || row.vectorRank !== null || row.vectorDistance !== null)) {
      throw new Error(`Retrieval row ${index} violates lexical component shape`);
    }
    if (mode === "vector_exact" && (row.vectorRank === null || row.vectorDistance === null || row.lexicalRank !== null || row.lexicalScore !== null)) {
      throw new Error(`Retrieval row ${index} violates vector component shape`);
    }
    if (mode === "hybrid_rrf" && row.lexicalRank === null && row.vectorRank === null) throw new Error(`Retrieval row ${index} has no hybrid component rank`);
    return row;
  });
  if (rows.some((row, index) => row.rankOrdinal !== index + 1)) throw new Error("Retrieval ranks are not the exact ordered sequence 1..N");
  if (new Set(rows.map((row) => row.chunkId)).size !== rows.length) throw new Error("Retrieval returned duplicate chunk IDs");
  if (new Set(rows.map((row) => row.retrievalRunId)).size > 1) throw new Error("Retrieval rows disagree on retrieval run identity");
  return rows;
}

function positiveInteger(value: unknown, label: string): number {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < 1) throw new Error(`${label} must be a positive integer`);
  return number;
}

function nullablePositiveInteger(value: unknown, label: string): number | null {
  return value === null || value === undefined ? null : positiveInteger(value, label);
}

function finiteNumber(value: unknown, label: string): number {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`${label} must be finite`);
  return number;
}

function nullableFiniteNumber(value: unknown, label: string): number | null {
  return value === null || value === undefined ? null : finiteNumber(value, label);
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) throw new Error(`${label} must be a non-empty string`);
  return value;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
