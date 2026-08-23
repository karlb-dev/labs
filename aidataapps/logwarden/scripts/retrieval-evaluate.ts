import { randomUUID } from "node:crypto";
import { performance } from "node:perf_hooks";
import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { summarizeEmbeddingMetrics, validateEmbeddingMetricDelta } from "../src/embedding-metrics.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { resolveEmbeddingProfile } from "../src/models.js";
import { packetLeakageFindings } from "../src/packets.js";
import { connect } from "../src/repository.js";
import { mean, percentile, scoreRunbookRetrieval, type RetrievalMetrics } from "../src/retrieval-metrics.js";
import {
  canonicalRetrievalQuery,
  createRetrievalQueryEmbedding,
  executeRunbookRetrieval,
  type QueryEmbedding,
  type RetrievalMode,
} from "../src/retrieval.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal, type StartedSpan } from "../src/telemetry.js";

const primaryModes = ["lexical_fulltext", "vector_exact", "hybrid_rrf"] as const satisfies readonly RetrievalMode[];
const benchmarkModes = [...primaryModes, "oracle_runbook", "shuffled_runbook"] as const;
type BenchmarkMode = typeof benchmarkModes[number];
const allowedRoles = ["dev", "calibration", "test_id", "test_variant_holdout", "test_unknown", "test_live_parity", "test_storm"] as const;
const roles = roleArgument();
const topK = integerArgument("--top-k") ?? 5;
const limit = integerArgument("--limit");
if (topK < 1 || topK > 20) throw new Error("--top-k must be between 1 and 20");
if (limit !== undefined && limit < 1) throw new Error("--limit must be positive");

const config = loadConfig();
const profile = resolveEmbeddingProfile(argument("--profile") ?? "qwen3-embedding-0.6b");
const corpusId = argument("--corpus") ?? "primary-v1";
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const evaluationId = randomUUID();
const startedAtUtc = new Date().toISOString();
const receiptPath = `${runDirectory}/knowledge/retrieval-evaluation-${safeName(roles.join("-"))}.json`;
const tablePath = `${runDirectory}/tables/retrieval-benchmark-${safeName(roles.join("-"))}.jsonl`;
const rawDirectory = `${runDirectory}/raw/retrieval-evaluation/${evaluationId}`;
const created = await createComponentTelemetryJournal(runDirectory, run.runId, "retrieval-evaluation");
const root = await created.journal.startSpan("retrieval.evaluation", {}, {
  evaluationId, roles, topK, corpusId, profileKey: profile.key, modes: benchmarkModes, limit: limit ?? null,
});
const lab = await connect(config.databases.lab, config.databases.controlName, 600_000);
const agent = await connect(config.databases.agent, config.databases.controlName, 600_000);
let embeddingCalls = 0;

try {
  await requireSchemaAndMutableCorpus();
  const packets = await loadPackets();
  if (packets.length === 0) throw new Error(`No valid frozen packets exist for roles ${roles.join(",")}`);
  const selected = limit === undefined ? packets : packets.slice(0, limit);
  const chunks = await loadControlChunks();
  const existing = await loadExistingCells();
  const metricsBefore = await retainEmbeddingMetrics("before", root);
  const rows: BenchmarkCell[] = [];

  for (const [episodeOrdinal, packet] of selected.entries()) {
    const query = packetRetrievalQuery(packet.packet_json);
    const querySha256 = sha256(query);
    const expected = parseRunbookArray(packet.expected_runbooks_json);
    const episodeSpan = await created.journal.startSpan("retrieval.evaluation.episode", {
      traceId: root.traceId, parentSpanId: root.spanId, episodeId: packet.episode_id,
    }, { episodeOrdinal, splitRole: packet.split_role, querySha256, expectedRunbookCount: expected.length });
    try {
      const missing = benchmarkModes.filter((mode) => !existing.has(cellKey(packet.episode_id, mode)));
      let queryEmbedding: QueryEmbedding | undefined;
      if (missing.some((mode) => mode === "vector_exact" || mode === "hybrid_rrf")) {
        queryEmbedding = await createRetrievalQueryEmbedding({
          endpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`, profile, query,
          journal: created.journal, runDirectory,
          context: { traceId: episodeSpan.traceId, parentSpanId: episodeSpan.spanId, episodeId: packet.episode_id },
        });
        embeddingCalls += 1;
      }

      for (const mode of missing) {
        const modeStarted = performance.now();
        let retrievalRunId: number;
        let returnedRunbooks: string[];
        let sqlElapsedMs: number;
        let totalElapsedMs: number;
        if (isPrimaryMode(mode)) {
          const retrieval = await executeRunbookRetrieval({
            pool: agent, runId: run.runId, episodeId: packet.episode_id, query, topK, corpusId,
            mode, profile, embeddingEndpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`,
            ...(mode === "lexical_fulltext" ? {} : { queryEmbedding: queryEmbedding! }),
            journal: created.journal, runDirectory,
            context: { traceId: episodeSpan.traceId, parentSpanId: episodeSpan.spanId, episodeId: packet.episode_id },
          });
          retrievalRunId = await markPrimaryEvaluatorOnly(packet.episode_id, querySha256, mode, retrieval.rows[0]?.retrievalRunId);
          returnedRunbooks = retrieval.rows.map((row) => row.runbookId);
          sqlElapsedMs = retrieval.sqlElapsedMs;
          totalElapsedMs = retrieval.totalElapsedMs;
        } else {
          const selectedChunks = mode === "oracle_runbook"
            ? oracleChunks(chunks, expected, topK)
            : shuffledChunks(chunks, expected, `${run.runId}\0${packet.split_role}\0${packet.episode_id}`, topK);
          const control = await persistControlRetrieval(packet.episode_id, query, querySha256, mode, selectedChunks);
          retrievalRunId = control.retrievalRunId;
          returnedRunbooks = selectedChunks.map((chunk) => chunk.runbookId);
          sqlElapsedMs = control.sqlElapsedMs;
          totalElapsedMs = round(performance.now() - modeStarted);
        }
        const metrics = scoreRunbookRetrieval(expected, returnedRunbooks, topK);
        const cell = await persistBenchmarkCell({
          episodeId: packet.episode_id, splitRole: packet.split_role, scenarioGroupId: packet.scenario_group_id,
          family: packet.family, mode, retrievalRunId, topK, querySha256, expectedRunbooks: expected,
          returnedRunbooks, metrics, sqlElapsedMs, totalElapsedMs,
        });
        existing.add(cellKey(packet.episode_id, mode));
        rows.push(cell);
        await created.journal.record("metric", "retrieval.evaluation.cell", {
          traceId: episodeSpan.traceId, parentSpanId: episodeSpan.spanId, episodeId: packet.episode_id,
        }, {
          mode, retrievalRunId, splitRole: packet.split_role, recallAtK: metrics.recallAtK,
          reciprocalRank: metrics.reciprocalRank, ndcgAtK: metrics.ndcgAtK,
          noAnswerCorrect: metrics.noAnswerCorrect, sqlElapsedMs, totalElapsedMs, resultSha256: cell.resultSha256,
        }, { status: "success" });
      }
      await created.journal.endSpan(episodeSpan, "success", { missingCellsCompleted: missing.length });
      if ((episodeOrdinal + 1) % 10 === 0 || episodeOrdinal + 1 === selected.length) {
        console.log(JSON.stringify({ stage: "retrieval-evaluation", episodes: episodeOrdinal + 1, total: selected.length, newCells: rows.length }));
      }
    } catch (error) {
      await created.journal.endSpan(episodeSpan, "failed", { errorDetail: safeError(error) }).catch(() => undefined);
      throw error;
    }
  }

  const metricsAfter = await retainEmbeddingMetrics("after", root);
  const embeddingMetricDelta = embeddingCalls === 0
    ? zeroEmbeddingDelta(metricsBefore.summary, metricsAfter.summary)
    : validateEmbeddingMetricDelta(metricsBefore.summary, metricsAfter.summary, embeddingCalls, embeddingCalls);
  const complete = await loadCompletedCells(selected.map((packet) => packet.episode_id));
  const expectedCellCount = selected.length * benchmarkModes.length;
  if (complete.length !== expectedCellCount) throw new Error(`Retrieval benchmark retained ${complete.length}/${expectedCellCount} required cells`);
  const aggregates = aggregate(complete);
  const orderedRowsSha256 = hashJson(complete.map((row) => ({
    episodeId: row.episodeId, mode: row.mode, resultSha256: row.resultSha256,
  })));
  await atomicWrite(tablePath, complete.map((row) => canonicalJson(row)).join("\n") + "\n", 0o600);
  await created.journal.record("point", "state.retrieval_evaluation_validated", {
    traceId: root.traceId, parentSpanId: root.spanId,
  }, { evaluationId, episodes: selected.length, cells: complete.length, orderedRowsSha256, embeddingCalls }, { status: "success" });
  await created.journal.endSpan(root, "success", { episodes: selected.length, cells: complete.length, embeddingCalls });
  await created.journal.flush();
  const ingestion = await ingestTelemetryJournal(lab, run.runId, created.path, { batchSize: 100 });
  const receiptBody = {
    schemaVersion: 1, runId: run.runId, evaluationId, startedAtUtc, finishedAtUtc: new Date().toISOString(),
    roles, topK, corpusId, profileKey: profile.key, modelId: profile.modelId, revision: profile.revision,
    modes: benchmarkModes, episodeCount: selected.length, expectedCellCount, completedCellCount: complete.length,
    newCellsThisInvocation: rows.length, resumedCells: complete.length - rows.length, embeddingCallsThisInvocation: embeddingCalls,
    limited: limit !== undefined, queryPolicy: "packet-visible-events-v1", metricPolicy: "unique-runbook-binary-relevance-v1",
    shuffledControlPolicy: "deterministic-hash-ranked-nonacceptable-runbooks-v1",
    aggregates, embeddingMetrics: { before: metricsBefore, after: metricsAfter, delta: embeddingMetricDelta },
    tablePath, orderedRowsSha256, telemetry: { journalPath: created.path, ingestion },
    disposition: limit === undefined ? "PASS" : "DEVELOPMENT_ONLY",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  await appendExperimentLog(`Retrieval evaluation ${evaluationId} retained ${complete.length} evaluator-only cells over ${selected.length} packets (${roles.join(",")}); disposition ${receipt.disposition}; receipt ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({
    runId: run.runId, evaluationId, roles, episodes: selected.length, cells: complete.length,
    embeddingCalls, aggregates, disposition: receipt.disposition, receiptPath, receiptSha256: receipt.receiptSha256,
  }, null, 2));
} catch (error) {
  const errorDetail = safeError(error);
  await created.journal.record("point", "state.retrieval_evaluation_failed", {
    traceId: root.traceId, parentSpanId: root.spanId,
  }, { evaluationId, errorDetail, embeddingCalls }, { status: "failed" }).catch(() => undefined);
  await created.journal.endSpan(root, "failed", { errorDetail, embeddingCalls }).catch(() => undefined);
  await created.journal.flush().catch(() => undefined);
  const ingestion = await ingestTelemetryJournal(lab, run.runId, created.path, { batchSize: 100 })
    .catch((ingestionError) => ({ status: "failed", errorDetail: safeError(ingestionError) }));
  const failureBody = {
    schemaVersion: 1, runId: run.runId, evaluationId, startedAtUtc, finishedAtUtc: new Date().toISOString(),
    roles, topK, corpusId, embeddingCalls, errorDetail, telemetryJournalPath: created.path, ingestion, disposition: "STOP_DATA",
  };
  const failure = { ...failureBody, receiptSha256: hashJson(failureBody) };
  await atomicWrite(`${rawDirectory}/failure-receipt.json`, `${JSON.stringify(failure, null, 2)}\n`, 0o600).catch(() => undefined);
  console.error(JSON.stringify(failure, null, 2));
  process.exitCode = 1;
} finally {
  await agent.close();
  await lab.close();
}

interface PacketRow {
  episode_id: string;
  split_role: string;
  scenario_group_id: string;
  family: string;
  packet_json: string;
  packet_sha256: string;
  expected_runbooks_json: string;
}

interface ControlChunk {
  chunkId: string;
  runbookId: string;
  headingPath: string;
  content: string;
}

interface BenchmarkCell {
  episodeId: string;
  splitRole: string;
  scenarioGroupId: string;
  family: string;
  mode: BenchmarkMode;
  retrievalRunId: number;
  topK: number;
  querySha256: string;
  expectedRunbooks: string[];
  returnedRunbooks: string[];
  metrics: RetrievalMetrics;
  sqlElapsedMs: number;
  totalElapsedMs: number;
  resultSha256: string;
}

async function requireSchemaAndMutableCorpus(): Promise<void> {
  const state = await lab.request().input("corpus", sql.VarChar(80), corpusId)
    .input("profile", sql.VarChar(80), profile.key).query<{
    frozen_at_utc: Date | null; embedding_count: number; chunk_count: number; migration_present: number;
  }>(`
    SELECT corpus.frozen_at_utc,
      (SELECT COUNT(*) FROM kb.chunk_embeddings embedding INNER JOIN kb.runbook_chunks chunk ON chunk.chunk_id=embedding.chunk_id WHERE chunk.corpus_id=@corpus AND embedding.embedding_profile_id=@profile) AS embedding_count,
      (SELECT COUNT(*) FROM kb.runbook_chunks WHERE corpus_id=@corpus) AS chunk_count,
      CONVERT(int,CASE WHEN OBJECT_ID(N'eval.retrieval_benchmark_results',N'U') IS NULL THEN 0 ELSE 1 END) AS migration_present
    FROM kb.search_corpora corpus WHERE corpus.corpus_id=@corpus;
  `);
  const row = state.recordset[0];
  if (row === undefined) throw new Error(`Unknown corpus ${corpusId}`);
  if (Number(row.migration_present) !== 1) throw new Error("Migration 031 must be applied before retrieval evaluation");
  if (row.frozen_at_utc !== null) throw new Error(`Corpus ${corpusId} is already frozen`);
  if (Number(row.chunk_count) < 1 || Number(row.embedding_count) !== Number(row.chunk_count)) {
    throw new Error(`Corpus embedding completeness failed: ${row.embedding_count}/${row.chunk_count}`);
  }
}

async function loadPackets(): Promise<PacketRow[]> {
  const result = await lab.request().input("run", sql.VarChar(120), run.runId).query<PacketRow>(`
    SELECT packet.episode_id,packet.split_role,truth.scenario_group_id,truth.family,
      packet.packet_json,packet.packet_sha256,truth.expected_runbooks_json
    FROM ingest.incident_packets packet
    INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=packet.episode_id
    INNER JOIN workload.injection_executions execution ON execution.episode_id=packet.episode_id AND execution.run_id=@run
    INNER JOIN workload.schedule_items schedule_item ON schedule_item.schedule_item_id=execution.schedule_item_id
    INNER JOIN workload.schedules schedule ON schedule.schedule_id=schedule_item.schedule_id
    WHERE packet.is_valid=1 AND schedule.schedule_name='standard-v1'
    ORDER BY packet.split_role,truth.scenario_group_id,packet.episode_id;
  `);
  const filtered = result.recordset.filter((row) => roles.includes(row.split_role as typeof roles[number]));
  for (const row of filtered) {
    const parsed = JSON.parse(row.packet_json) as unknown;
    if (hashJson(parsed) !== row.packet_sha256) throw new Error(`Packet hash drift for ${row.episode_id}`);
    const leaks = packetLeakageFindings(row.packet_json, []);
    if (leaks.length > 0) throw new Error(`Packet protected-field leakage for ${row.episode_id}: ${leaks.join(",")}`);
  }
  return filtered;
}

async function loadControlChunks(): Promise<ControlChunk[]> {
  const result = await lab.request().input("corpus", sql.VarChar(80), corpusId).query<{
    chunk_id: string; runbook_id: string; heading_path: string; content: string;
  }>(`
    SELECT chunk.chunk_id,chunk.runbook_id,chunk.heading_path,chunk.content
    FROM kb.runbook_chunks chunk INNER JOIN kb.runbooks book ON book.runbook_id=chunk.runbook_id
    WHERE chunk.corpus_id=@corpus AND chunk.ordinal=1 AND book.enabled=1 ORDER BY chunk.runbook_id;
  `);
  return result.recordset.map((row) => ({ chunkId: row.chunk_id, runbookId: row.runbook_id, headingPath: row.heading_path, content: row.content }));
}

async function loadExistingCells(): Promise<Set<string>> {
  const result = await lab.request().input("run", sql.VarChar(120), run.runId).query<{ episode_id: string; retrieval_mode: BenchmarkMode }>(`
    SELECT episode_id,retrieval_mode FROM eval.retrieval_benchmark_results WHERE run_id=@run;
  `);
  return new Set(result.recordset.map((row) => cellKey(row.episode_id, row.retrieval_mode)));
}

async function markPrimaryEvaluatorOnly(
  episodeId: string, querySha256: string, mode: RetrievalMode, knownId: number | undefined,
): Promise<number> {
  const request = lab.request().input("run", sql.VarChar(120), run.runId)
    .input("episode", sql.VarChar(120), episodeId).input("query", sql.Char(64), querySha256)
    .input("mode", sql.VarChar(40), mode).input("known", sql.BigInt, knownId ?? null);
  const result = await request.query<{ retrieval_run_id: string }>(`
    DECLARE @id bigint=@known;
    IF @id IS NULL SELECT TOP(1) @id=retrieval_run_id FROM kb.retrieval_runs
      WHERE run_id=@run AND episode_id=@episode AND query_sha256=@query AND retrieval_mode=@mode
      ORDER BY retrieval_run_id DESC;
    IF @id IS NULL THROW 51910,'retained retrieval run identity was not found',1;
    UPDATE kb.retrieval_runs SET evaluator_only=1 WHERE retrieval_run_id=@id AND status='complete';
    UPDATE kb.retrieval_results SET returned_to_agent=0 WHERE retrieval_run_id=@id;
    SELECT @id AS retrieval_run_id;
  `);
  return Number(result.recordset[0]!.retrieval_run_id);
}

async function persistControlRetrieval(
  episodeId: string, query: string, querySha256: string,
  mode: "oracle_runbook" | "shuffled_runbook", chunks: ControlChunk[],
): Promise<{ retrievalRunId: number; sqlElapsedMs: number }> {
  const started = performance.now();
  const transaction = new sql.Transaction(lab);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    const inserted = await new sql.Request(transaction)
      .input("run", sql.VarChar(120), run.runId).input("episode", sql.VarChar(120), episodeId)
      .input("corpus", sql.VarChar(80), corpusId).input("mode", sql.VarChar(40), mode)
      .input("query", sql.NVarChar(sql.MAX), query).input("hash", sql.Char(64), querySha256)
      .input("k", sql.Int, topK).input("count", sql.Int, chunks.length)
      .query<{ retrieval_run_id: string }>(`
        INSERT kb.retrieval_runs(run_id,episode_id,corpus_id,retrieval_mode,query_text,query_sha256,
          requested_k,candidate_count,started_at_utc,finished_at_utc,latency_ms,status,evaluator_only)
        OUTPUT inserted.retrieval_run_id
        VALUES(@run,@episode,@corpus,@mode,@query,@hash,@k,@count,SYSUTCDATETIME(),SYSUTCDATETIME(),0,'complete',1);
      `);
    const retrievalRunId = Number(inserted.recordset[0]!.retrieval_run_id);
    for (const [index, chunk] of chunks.entries()) await new sql.Request(transaction)
      .input("run_id", sql.BigInt, retrievalRunId).input("rank", sql.Int, index + 1)
      .input("chunk", sql.VarChar(120), chunk.chunkId).input("runbook", sql.VarChar(100), chunk.runbookId)
      .input("score", sql.Float, 1 / (index + 1)).query(`
        INSERT kb.retrieval_results(retrieval_run_id,rank_ordinal,chunk_id,runbook_id,fused_score,returned_to_agent)
        VALUES(@run_id,@rank,@chunk,@runbook,@score,0);
      `);
    const elapsed = round(performance.now() - started);
    await new sql.Request(transaction).input("id", sql.BigInt, retrievalRunId).input("elapsed", sql.Decimal(18, 3), elapsed)
      .query("UPDATE kb.retrieval_runs SET latency_ms=@elapsed WHERE retrieval_run_id=@id;");
    await transaction.commit();
    return { retrievalRunId, sqlElapsedMs: elapsed };
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
}

async function persistBenchmarkCell(input: Omit<BenchmarkCell, "resultSha256">): Promise<BenchmarkCell> {
  const body = { schemaVersion: 1, runId: run.runId, ...input };
  const resultSha256 = hashJson(body);
  await lab.request()
    .input("run", sql.VarChar(120), run.runId).input("episode", sql.VarChar(120), input.episodeId)
    .input("role", sql.VarChar(40), input.splitRole).input("mode", sql.VarChar(40), input.mode)
    .input("retrieval", sql.BigInt, input.retrievalRunId).input("k", sql.Int, input.topK)
    .input("query", sql.Char(64), input.querySha256)
    .input("expected", sql.NVarChar(sql.MAX), canonicalJson(input.expectedRunbooks))
    .input("returned", sql.NVarChar(sql.MAX), canonicalJson(input.returnedRunbooks))
    .input("recall", sql.Decimal(9, 6), input.metrics.recallAtK)
    .input("mrr", sql.Decimal(9, 6), input.metrics.reciprocalRank)
    .input("ndcg", sql.Decimal(9, 6), input.metrics.ndcgAtK)
    .input("noanswer", sql.Bit, input.metrics.noAnswerCorrect)
    .input("hash", sql.Char(64), resultSha256).query(`
      INSERT eval.retrieval_benchmark_results(run_id,episode_id,split_role,retrieval_mode,retrieval_run_id,
        requested_k,query_sha256,expected_runbooks_json,returned_runbooks_json,recall_at_k,reciprocal_rank,
        ndcg_at_k,no_answer_correct,evaluator_only,result_sha256)
      VALUES(@run,@episode,@role,@mode,@retrieval,@k,@query,@expected,@returned,@recall,@mrr,@ndcg,@noanswer,1,@hash);
    `);
  return { ...input, resultSha256 };
}

async function loadCompletedCells(episodeIds: string[]): Promise<BenchmarkCell[]> {
  const episodeSet = new Set(episodeIds);
  const result = await lab.request().input("run", sql.VarChar(120), run.runId).query<{
    episode_id: string; split_role: string; scenario_group_id: string; family: string; retrieval_mode: BenchmarkMode;
    retrieval_run_id: string; requested_k: number; query_sha256: string; expected_runbooks_json: string;
    returned_runbooks_json: string; recall_at_k: number; reciprocal_rank: number; ndcg_at_k: number;
    no_answer_correct: boolean | null; result_sha256: string; latency_ms: number;
  }>(`
    SELECT score.episode_id,score.split_role,truth.scenario_group_id,truth.family,score.retrieval_mode,
      score.retrieval_run_id,score.requested_k,score.query_sha256,score.expected_runbooks_json,
      score.returned_runbooks_json,score.recall_at_k,score.reciprocal_rank,score.ndcg_at_k,
      score.no_answer_correct,score.result_sha256,run.latency_ms
    FROM eval.retrieval_benchmark_results score
    INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=score.episode_id
    INNER JOIN kb.retrieval_runs run ON run.retrieval_run_id=score.retrieval_run_id
    WHERE score.run_id=@run ORDER BY score.split_role,truth.scenario_group_id,score.episode_id,score.retrieval_mode;
  `);
  return result.recordset.filter((row) => episodeSet.has(row.episode_id)).map((row) => {
    const expectedRunbooks = parseRunbookArray(row.expected_runbooks_json);
    const returnedRunbooks = parseRunbookArray(row.returned_runbooks_json);
    const metrics = scoreRunbookRetrieval(expectedRunbooks, returnedRunbooks, Number(row.requested_k));
    if (Math.abs(metrics.recallAtK - Number(row.recall_at_k)) > 0.000001 ||
        Math.abs(metrics.reciprocalRank - Number(row.reciprocal_rank)) > 0.000001 ||
        Math.abs(metrics.ndcgAtK - Number(row.ndcg_at_k)) > 0.000001 || metrics.noAnswerCorrect !== row.no_answer_correct) {
      throw new Error(`Retained retrieval score drift for ${row.episode_id}/${row.retrieval_mode}`);
    }
    return {
      episodeId: row.episode_id, splitRole: row.split_role, scenarioGroupId: row.scenario_group_id, family: row.family,
      mode: row.retrieval_mode, retrievalRunId: Number(row.retrieval_run_id), topK: Number(row.requested_k),
      querySha256: row.query_sha256, expectedRunbooks, returnedRunbooks, metrics,
      sqlElapsedMs: Number(row.latency_ms), totalElapsedMs: Number(row.latency_ms), resultSha256: row.result_sha256,
    };
  });
}

function packetRetrievalQuery(packetJson: string): string {
  const packet = JSON.parse(packetJson) as {
    sourceEvents?: Array<{ sourceKind?: unknown; eventName?: unknown; errorNumber?: unknown; severity?: unknown; state?: unknown; message?: unknown }>;
    recentHistory?: Record<string, unknown>;
  };
  if (!Array.isArray(packet.sourceEvents) || packet.sourceEvents.length === 0) throw new Error("Packet has no visible source events");
  const fields: string[] = [];
  for (const event of packet.sourceEvents) {
    fields.push(
      stringField(event.sourceKind), stringField(event.eventName),
      event.errorNumber === null || event.errorNumber === undefined ? "" : `error ${String(event.errorNumber)}`,
      event.severity === null || event.severity === undefined ? "" : `severity ${String(event.severity)}`,
      event.state === null || event.state === undefined ? "" : `state ${String(event.state)}`,
      stringField(event.message),
    );
  }
  if (packet.recentHistory !== undefined) {
    for (const [key, value] of Object.entries(packet.recentHistory).sort(([left], [right]) => left.localeCompare(right))) {
      fields.push(`${key} ${String(value)}`);
    }
  }
  const normalized = fields.filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
  return canonicalRetrievalQuery(normalized.slice(0, 1000));
}

function oracleChunks(chunks: ControlChunk[], expected: string[], k: number): ControlChunk[] {
  const byRunbook = new Map(chunks.map((chunk) => [chunk.runbookId, chunk]));
  return expected.map((runbookId) => byRunbook.get(runbookId)).filter((chunk): chunk is ControlChunk => chunk !== undefined).slice(0, k);
}

function shuffledChunks(chunks: ControlChunk[], expected: string[], seed: string, k: number): ControlChunk[] {
  const excluded = new Set(expected);
  return chunks.filter((chunk) => !excluded.has(chunk.runbookId))
    .sort((left, right) => sha256(`${seed}\0${left.chunkId}`).localeCompare(sha256(`${seed}\0${right.chunkId}`)))
    .slice(0, k);
}

function aggregate(rows: BenchmarkCell[]) {
  return Object.fromEntries(benchmarkModes.map((mode) => {
    const selected = rows.filter((row) => row.mode === mode);
    const answerable = selected.filter((row) => row.metrics.answerable);
    const noAnswer = selected.filter((row) => !row.metrics.answerable);
    return [mode, {
      cells: selected.length, answerableCells: answerable.length, noAnswerCells: noAnswer.length,
      meanRecallAtK: mean(answerable.map((row) => row.metrics.recallAtK)),
      meanReciprocalRank: mean(answerable.map((row) => row.metrics.reciprocalRank)),
      meanNdcgAtK: mean(answerable.map((row) => row.metrics.ndcgAtK)),
      noAnswerAccuracy: noAnswer.length === 0 ? null : mean(noAnswer.map((row) => row.metrics.noAnswerCorrect ? 1 : 0)),
      sqlLatencyP50Ms: percentile(selected.map((row) => row.sqlElapsedMs), 0.5),
      sqlLatencyP95Ms: percentile(selected.map((row) => row.sqlElapsedMs), 0.95),
    }];
  }));
}

async function retainEmbeddingMetrics(phase: "before" | "after", span: StartedSpan) {
  const endpoint = new URL("/metrics", config.inference.qwenEmbeddingBaseUrl);
  const response = await fetch(endpoint, { signal: AbortSignal.timeout(30_000) });
  const body = Buffer.from(await response.arrayBuffer());
  if (!response.ok) throw new Error(`Embedding metrics returned HTTP ${response.status}`);
  const path = `${rawDirectory}/embedding-metrics-${phase}.prom`;
  await atomicWrite(path, body, 0o600);
  const snapshot = {
    phase, observedAtUtc: new Date().toISOString(), path, payloadBytes: body.length,
    payloadSha256: sha256(body), summary: summarizeEmbeddingMetrics(body.toString("utf8"), profile.modelId),
  };
  await created.journal.record("raw_snapshot", "retrieval.evaluation.embedding_metrics", {
    traceId: span.traceId, parentSpanId: span.spanId,
  }, snapshot, { status: "available" });
  return snapshot;
}

function zeroEmbeddingDelta(before: ReturnType<typeof summarizeEmbeddingMetrics>, after: ReturnType<typeof summarizeEmbeddingMetrics>) {
  const fields = ["embeddingHttpRequests", "successfulRequests", "erroredRequests", "promptTokens", "latencyObservations", "preemptions"] as const;
  const delta = Object.fromEntries(fields.map((field) => [field, after[field] - before[field]])) as Record<typeof fields[number], number>;
  if (fields.some((field) => !Number.isFinite(delta[field]) || delta[field] < 0)) throw new Error("Embedding metric counter reset during resumed evaluation");
  if (delta.erroredRequests !== 0 || delta.preemptions !== 0) throw new Error("Embedding service reported an error or preemption during resumed evaluation");
  return delta;
}

function parseRunbookArray(source: string): string[] {
  const value = JSON.parse(source) as unknown;
  if (!Array.isArray(value) || !value.every((entry) => typeof entry === "string" && /^TSG-[A-Z]{3,5}-\d{2}$/.test(entry))) {
    throw new Error("Expected runbook JSON violates its protected contract");
  }
  return [...new Set(value)];
}

function roleArgument(): Array<typeof allowedRoles[number]> {
  const source = argument("--roles") ?? argument("--split") ?? "test_id,test_variant_holdout,test_unknown";
  const values = [...new Set(source.split(",").map((value) => value.trim()).filter(Boolean))];
  if (values.length === 0 || !values.every((value) => (allowedRoles as readonly string[]).includes(value))) {
    throw new Error(`Invalid roles; allowed values are ${allowedRoles.join(",")}`);
  }
  return values as Array<typeof allowedRoles[number]>;
}

function isPrimaryMode(mode: BenchmarkMode): mode is RetrievalMode {
  return (primaryModes as readonly string[]).includes(mode);
}

function stringField(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function cellKey(episodeId: string, mode: BenchmarkMode): string {
  return `${episodeId}\0${mode}`;
}

function safeName(value: string): string {
  return value.replace(/[^a-z0-9_.-]+/gi, "-").toLowerCase();
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}

function integerArgument(name: string): number | undefined {
  const value = argument(name);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function safeError(error: unknown): string {
  return (error as { message?: string }).message ?? String(error);
}
