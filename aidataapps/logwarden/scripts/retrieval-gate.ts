import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { summarizeEmbeddingMetrics, validateEmbeddingMetricDelta } from "../src/embedding-metrics.js";
import { hashJson, sha256 } from "../src/hash.js";
import { resolveEmbeddingProfile } from "../src/models.js";
import { connect } from "../src/repository.js";
import {
  createRetrievalQueryEmbedding,
  executeRunbookRetrieval,
  type RetrievalMode,
  type RetrievalResult,
} from "../src/retrieval.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal, type FileTelemetryJournal, type StartedSpan } from "../src/telemetry.js";

const cases = [
  ["deadlock victim cycle retry", "TSG-DLK-"],
  ["head blocker open transaction waiters", "TSG-BLK-"],
  ["transaction log full 9002 active transaction", "TSG-LOG-"],
  ["login failed 18456 default database", "TSG-AUT-"],
  ["integrity checksum allocation consistency", "TSG-INT-"],
  ["backup destination failure restore checksum", "TSG-BAK-"],
  ["query high CPU memory grant timeout", "TSG-QRY-"],
  ["conversion constraint missing object truncation", "TSG-DAT-"],
  ["ambiguous conflicting truncated unsupported", "TSG-UNK-"],
  ["successful duplicate subthreshold no action", "TSG-NOI-"],
  ["timeout (error 9002) 'active transaction'", "TSG-LOG-"],
] as const;
const modes: RetrievalMode[] = ["lexical_fulltext", "vector_exact", "hybrid_rrf"];
const config = loadConfig();
const profile = resolveEmbeddingProfile("qwen3-embedding-0.6b");
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const gateId = randomUUID();
const rawDirectory = `${runDirectory}/raw/retrieval-gate/${gateId}`;
const receiptPath = `${runDirectory}/knowledge/retrieval-gate.json`;
const created = await createComponentTelemetryJournal(runDirectory, run.runId, "retrieval-gate");
const root = await created.journal.startSpan("retrieval.gate", {}, {
  gateId,
  corpusId: "primary-v1",
  profileKey: profile.key,
  queryCount: cases.length,
  modes,
  topK: 5,
  rrfK: 60,
  candidateK: 50,
});
const agent = await connect(config.databases.agent, config.databases.controlName, 120_000);
const lab = await connect(config.databases.lab, config.databases.controlName, 120_000);

try {
  const metricsBefore = await retainMetricSnapshot("before", created.journal, root);
  const results: Array<{
    query: string;
    expectedPrefix: string;
    queryEmbedding: Record<string, unknown>;
    modes: Array<Record<string, unknown>>;
  }> = [];
  const allRetrievals: RetrievalResult[] = [];

  for (const [caseIndex, [query, expectedPrefix]] of cases.entries()) {
    const episodeId = `retrieval-gate-${gateId.slice(0, 24)}-${caseIndex}`;
    const queryEmbedding = await createRetrievalQueryEmbedding({
      endpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`,
      profile,
      query,
      journal: created.journal,
      runDirectory,
      context: { traceId: root.traceId, parentSpanId: root.spanId, episodeId },
    });
    const modeResults: Array<Record<string, unknown>> = [];
    for (const mode of modes) {
      const retrieval = await executeRunbookRetrieval({
        pool: agent,
        runId: run.runId,
        episodeId,
        query,
        topK: 5,
        corpusId: "primary-v1",
        mode,
        profile,
        embeddingEndpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`,
        ...(mode === "lexical_fulltext" ? {} : { queryEmbedding }),
        journal: created.journal,
        runDirectory,
        context: { traceId: root.traceId, parentSpanId: root.spanId, episodeId },
      });
      if (retrieval.rows.length !== 5 || !retrieval.rows.some((row) => row.runbookId.startsWith(expectedPrefix))) {
        throw new Error(`${mode} did not retrieve ${expectedPrefix} for ${query}`);
      }
      if (mode === "hybrid_rrf" && !retrieval.rows.some((row) => row.lexicalRank !== null && row.vectorRank !== null)) {
        throw new Error(`Hybrid retrieval had no dual-channel returned candidate for ${query}`);
      }
      const firstRelevant = retrieval.rows.findIndex((row) => row.runbookId.startsWith(expectedPrefix));
      modeResults.push({
        mode,
        retrievalRunId: retrieval.rows[0]!.retrievalRunId,
        returnedChunkIds: retrieval.rows.map((row) => row.chunkId),
        returnedRunbookIds: retrieval.rows.map((row) => row.runbookId),
        relevantRank: firstRelevant + 1,
        reciprocalRank: 1 / (firstRelevant + 1),
        recallAt5: 1,
        sqlElapsedMs: retrieval.sqlElapsedMs,
        totalElapsedMs: retrieval.totalElapsedMs,
        resultPath: retrieval.resultPath,
        resultSha256: retrieval.resultSha256,
        disposition: "PASS",
      });
      allRetrievals.push(retrieval);
    }
    results.push({
      query,
      expectedPrefix,
      queryEmbedding: {
        operationId: queryEmbedding.evidence.operationId,
        querySha256: queryEmbedding.querySha256,
        vectorSha256: queryEmbedding.vectorSha256,
        clientElapsedMs: queryEmbedding.evidence.clientElapsedMs,
        promptTokens: queryEmbedding.evidence.usage?.promptTokens ?? null,
        requestPath: queryEmbedding.evidence.requestPath,
        requestBodySha256: queryEmbedding.evidence.requestBodySha256,
        responsePath: queryEmbedding.evidence.responsePath,
        responseBodySha256: queryEmbedding.evidence.responseBodySha256,
      },
      modes: modeResults,
    });
  }

  const metricsAfter = await retainMetricSnapshot("after", created.journal, root);
  const metricDelta = validateEmbeddingMetricDelta(metricsBefore.summary, metricsAfter.summary, cases.length, cases.length);
  if (metricDelta.embeddingHttpRequests !== cases.length || metricDelta.successfulRequests !== cases.length || metricDelta.latencyObservations !== cases.length) {
    throw new Error(`Retrieval gate embedding metrics include unrelated traffic: ${JSON.stringify(metricDelta)}`);
  }
  const database = await verifyDatabaseEvidence(lab, allRetrievals);
  const aggregates = aggregateModes(results);
  await created.journal.record("point", "state.retrieval_gate_passed", {
    traceId: root.traceId,
    parentSpanId: root.spanId,
  }, { gateId, queryCount: cases.length, retrievalCount: allRetrievals.length, metricDelta, aggregates }, { status: "success" });
  await created.journal.endSpan(root, "success", {
    gateId,
    queryCount: cases.length,
    retrievalCount: allRetrievals.length,
    embeddingRequestCount: cases.length,
  });
  await created.journal.flush();
  const ingestion = await ingestTelemetryJournal(lab, run.runId, created.path, { batchSize: 100 });
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    gateId,
    createdAtUtc: new Date().toISOString(),
    corpusId: "primary-v1",
    profileKey: profile.key,
    modelId: profile.modelId,
    revision: profile.revision,
    dimensions: profile.dimensions,
    queryInputFormat: "query-text-v1",
    topK: 5,
    candidateK: 50,
    rrfK: 60,
    cases: results,
    aggregates,
    database,
    embeddingMetrics: { before: metricsBefore, after: metricsAfter, delta: metricDelta },
    telemetry: { journalPath: created.path, ingestion },
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const serialized = `${JSON.stringify(receipt, null, 2)}\n`;
  await atomicWrite(`${rawDirectory}/receipt.json`, serialized);
  await atomicWrite(receiptPath, serialized);
  console.log(JSON.stringify({
    runId: run.runId,
    gateId,
    queryCount: cases.length,
    retrievalCount: allRetrievals.length,
    aggregates,
    metricDelta,
    disposition: "PASS",
    receiptSha256: receipt.receiptSha256,
  }, null, 2));
} catch (error) {
  const errorDetail = safeError(error);
  await created.journal.record("point", "state.retrieval_gate_failed", {
    traceId: root.traceId,
    parentSpanId: root.spanId,
  }, { gateId, errorDetail }, { status: "failed" }).catch(() => undefined);
  await created.journal.endSpan(root, "failed", { gateId, errorDetail }).catch(() => undefined);
  await created.journal.flush().catch(() => undefined);
  const ingestion = await ingestTelemetryJournal(lab, run.runId, created.path, { batchSize: 100 }).catch((ingestionError) => ({ status: "failed", errorDetail: safeError(ingestionError) }));
  const failureBody = {
    schemaVersion: 1,
    runId: run.runId,
    gateId,
    createdAtUtc: new Date().toISOString(),
    errorDetail,
    telemetryJournalPath: created.path,
    ingestion,
    disposition: "FAIL",
  };
  const failure = { ...failureBody, receiptSha256: hashJson(failureBody) };
  await atomicWrite(`${rawDirectory}/failure-receipt.json`, `${JSON.stringify(failure, null, 2)}\n`).catch(() => undefined);
  console.error(JSON.stringify(failure, null, 2));
  process.exitCode = 1;
} finally {
  await Promise.all([agent.close(), lab.close()]);
}

async function verifyDatabaseEvidence(pool: sql.ConnectionPool, retrievals: RetrievalResult[]): Promise<Record<string, unknown>> {
  const ids = retrievals.map((retrieval) => retrieval.rows[0]!.retrievalRunId);
  if (new Set(ids).size !== retrievals.length) throw new Error("Retrieval gate reused a SQL retrieval run identity");
  const result = await pool.request()
    .input("ids", sql.NVarChar(sql.MAX), JSON.stringify(ids))
    .query<{
      retrieval_run_id: string;
      retrieval_mode: RetrievalMode;
      query_sha256: string;
      requested_k: number;
      candidate_count: number;
      status: string;
      dimensions: number | null;
      query_norm: number | null;
      result_count: number;
      returned_count: number;
      invalid_lexical_shape: number;
      invalid_vector_shape: number;
      invalid_hybrid_shape: number;
      dual_channel_count: number;
    }>(`
      WITH selected AS
      (
        SELECT retrieval_run_id FROM OPENJSON(@ids) WITH (retrieval_run_id bigint '$')
      ),
      result_summary AS
      (
        SELECT result.retrieval_run_id,
          COUNT(result.rank_ordinal) AS result_count,
          SUM(CASE WHEN result.returned_to_agent=1 THEN 1 ELSE 0 END) AS returned_count,
          SUM(CASE WHEN run.retrieval_mode='lexical_fulltext' AND
            (result.lexical_rank IS NULL OR result.lexical_score IS NULL OR result.vector_rank IS NOT NULL OR result.vector_distance IS NOT NULL) THEN 1 ELSE 0 END) AS invalid_lexical_shape,
          SUM(CASE WHEN run.retrieval_mode='vector_exact' AND
            (result.vector_rank IS NULL OR result.vector_distance IS NULL OR result.lexical_rank IS NOT NULL OR result.lexical_score IS NOT NULL) THEN 1 ELSE 0 END) AS invalid_vector_shape,
          SUM(CASE WHEN run.retrieval_mode='hybrid_rrf' AND result.lexical_rank IS NULL AND result.vector_rank IS NULL THEN 1 ELSE 0 END) AS invalid_hybrid_shape,
          SUM(CASE WHEN run.retrieval_mode='hybrid_rrf' AND result.lexical_rank IS NOT NULL AND result.vector_rank IS NOT NULL THEN 1 ELSE 0 END) AS dual_channel_count
        FROM kb.retrieval_results AS result
        INNER JOIN kb.retrieval_runs AS run ON run.retrieval_run_id=result.retrieval_run_id
        INNER JOIN selected ON selected.retrieval_run_id=result.retrieval_run_id
        GROUP BY result.retrieval_run_id
      )
      SELECT run.retrieval_run_id,run.retrieval_mode,run.query_sha256,run.requested_k,
        run.candidate_count,run.status,
        CASE WHEN run.query_embedding IS NULL THEN NULL ELSE CONVERT(int,VECTORPROPERTY(run.query_embedding,'Dimensions')) END AS dimensions,
        CASE WHEN run.query_embedding IS NULL THEN NULL ELSE CONVERT(float,VECTOR_NORM(run.query_embedding,'norm2')) END AS query_norm,
        summary.result_count,summary.returned_count,summary.invalid_lexical_shape,
        summary.invalid_vector_shape,summary.invalid_hybrid_shape,summary.dual_channel_count
      FROM kb.retrieval_runs AS run
      INNER JOIN selected ON selected.retrieval_run_id=run.retrieval_run_id
      LEFT JOIN result_summary AS summary ON summary.retrieval_run_id=run.retrieval_run_id;
    `);
  if (result.recordset.length !== retrievals.length) throw new Error(`SQL retained ${result.recordset.length}/${retrievals.length} retrieval runs`);
  const expected = new Map(retrievals.map((retrieval) => [retrieval.rows[0]!.retrievalRunId, retrieval]));
  for (const row of result.recordset) {
    const retrieval = expected.get(Number(row.retrieval_run_id));
    if (retrieval === undefined || row.status !== "complete" || row.query_sha256 !== retrieval.querySha256 || Number(row.requested_k) !== retrieval.topK ||
        Number(row.candidate_count) !== Number(row.result_count) || Number(row.returned_count) !== retrieval.rows.length ||
        Number(row.invalid_lexical_shape) !== 0 || Number(row.invalid_vector_shape) !== 0 || Number(row.invalid_hybrid_shape) !== 0) {
      throw new Error(`SQL retrieval evidence mismatch: ${JSON.stringify(row)}`);
    }
    if (row.retrieval_mode === "lexical_fulltext") {
      if (row.dimensions !== null || row.query_norm !== null) throw new Error("Lexical retrieval unexpectedly retained a vector");
    } else if (Number(row.dimensions) !== profile.dimensions || Math.abs(Number(row.query_norm) - 1) > 0.0001) {
      throw new Error("Vector retrieval query-vector storage validation failed");
    }
    if (row.retrieval_mode === "hybrid_rrf" && Number(row.dual_channel_count) < 1) throw new Error("SQL hybrid candidates never combined both channels");
  }
  return {
    retrievalRunCount: result.recordset.length,
    resultRowCount: result.recordset.reduce((sum, row) => sum + Number(row.result_count), 0),
    returnedRowCount: result.recordset.reduce((sum, row) => sum + Number(row.returned_count), 0),
    identitySetSha256: hashJson([...ids].sort((left, right) => left - right)),
    rowsSha256: hashJson(result.recordset),
    disposition: "PASS",
  };
}

function aggregateModes(results: Array<{ modes: Array<Record<string, unknown>> }>): Record<string, unknown> {
  return Object.fromEntries(modes.map((mode) => {
    const rows = results.map((result) => result.modes.find((value) => value.mode === mode)!);
    const sql = rows.map((row) => Number(row.sqlElapsedMs)).sort((left, right) => left - right);
    const total = rows.map((row) => Number(row.totalElapsedMs)).sort((left, right) => left - right);
    return [mode, {
      queries: rows.length,
      recallAt5: rows.reduce((sum, row) => sum + Number(row.recallAt5), 0) / rows.length,
      meanReciprocalRank: rows.reduce((sum, row) => sum + Number(row.reciprocalRank), 0) / rows.length,
      sqlLatencyP50Ms: percentile(sql, 0.5),
      sqlLatencyP95Ms: percentile(sql, 0.95),
      totalLatencyP50Ms: percentile(total, 0.5),
      totalLatencyP95Ms: percentile(total, 0.95),
    }];
  }));
}

async function retainMetricSnapshot(
  phase: "before" | "after",
  journal: FileTelemetryJournal,
  root: StartedSpan,
) {
  const endpoint = new URL("/metrics", config.inference.qwenEmbeddingBaseUrl);
  const started = performance.now();
  const response = await fetch(endpoint, { signal: AbortSignal.timeout(10_000) });
  const body = Buffer.from(await response.arrayBuffer());
  const path = `${rawDirectory}/embedding-metrics-${phase}.prom`;
  await atomicWrite(path, body);
  if (!response.ok) throw new Error(`Embedding metrics endpoint returned HTTP ${response.status}`);
  const snapshot = {
    phase,
    observedAtUtc: new Date().toISOString(),
    endpoint: endpoint.toString(),
    path,
    payloadSha256: sha256(body),
    payloadBytes: body.length,
    latencyMs: round(performance.now() - started),
    summary: summarizeEmbeddingMetrics(body.toString("utf8"), profile.modelId),
  };
  await journal.record("raw_snapshot", "retrieval.gate.embedding_metrics", { traceId: root.traceId, parentSpanId: root.spanId }, snapshot, { status: "available" });
  return snapshot;
}

function percentile(sorted: number[], fraction: number): number {
  return sorted[Math.ceil(sorted.length * fraction) - 1] ?? 0;
}

function safeError(error: unknown): string {
  return (error as { message?: string }).message ?? String(error);
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
