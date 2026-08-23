import { execFile as execFileCallback } from "node:child_process";
import { randomUUID } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";
import { relative } from "node:path";
import { promisify } from "node:util";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import {
  summarizeEmbeddingMetrics,
  validateEmbeddingMetricDelta,
  type EmbeddingMetricSummary,
} from "../src/embedding-metrics.js";
import { callOpenAiCompatibleEmbeddings, parseEmbeddingResponse, type EmbeddingCallEvidence } from "../src/embeddings.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { resolveEmbeddingProfile } from "../src/models.js";
import { connect } from "../src/repository.js";
import { atomicWrite, LAB_ROOT, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal, type TelemetryIngestionResult } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal, type FileTelemetryJournal, type StartedSpan } from "../src/telemetry.js";

const execFile = promisify(execFileCallback);
const INPUT_FORMAT_VERSION = "chunk-content-v1";
const CORPUS_ID = argument("--corpus") ?? "primary-v1";
const PROFILE_KEY = argument("--profile") ?? "qwen3-embedding-0.6b";
const BATCH_SIZE = integerArgument("--batch-size") ?? 32;
if (BATCH_SIZE < 1 || BATCH_SIZE > 64) throw new Error("--batch-size must be between 1 and 64");

const config = loadConfig();
const profile = resolveEmbeddingProfile(PROFILE_KEY);
const profileRecord = { modelId: profile.modelId, revision: profile.revision, dimensions: profile.dimensions, vllmImage: profile.vllmImage };
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const buildId = randomUUID();
const startedAtUtc = new Date().toISOString();
const rawDirectory = `${runDirectory}/raw/embedding-corpus-build/${buildId}`;
const batchReceiptDirectory = `${runDirectory}/knowledge/embedding-batches/${PROFILE_KEY}`;
const receiptPath = `${runDirectory}/knowledge/embeddings-${CORPUS_ID}-${PROFILE_KEY}.json`;
const portGatePath = `${runDirectory}/metrics/embedding-port-gate-${PROFILE_KEY}.json`;
const portGate = await validatedPortGate(portGatePath);
const created = await createComponentTelemetryJournal(runDirectory, run.runId, `embedding-corpus-build-${PROFILE_KEY}`);
const root = await created.journal.startSpan("embedding.corpus_build", {}, {
  buildId,
  corpusId: CORPUS_ID,
  profileKey: PROFILE_KEY,
  modelId: profile.modelId,
  revision: profile.revision,
  dimensions: profile.dimensions,
  inputFormatVersion: INPUT_FORMAT_VERSION,
  batchSize: BATCH_SIZE,
  portGateReceiptSha256: portGate.receiptSha256,
});
const pool = await connect(config.databases.lab, config.databases.controlName, 600_000);

try {
  const state = await loadBuildState(pool);
  const missing = state.chunks.filter((chunk) => !state.existing.has(chunk.chunkId));
  const metricsBefore = await retainMetricSnapshot("before", created.journal, root);
  const gpuBefore = await retainGpuSnapshot("before", created.journal, root);
  const batchReceipts: BatchReceipt[] = [];
  let generated = 0;

  for (let offset = 0; offset < missing.length; offset += BATCH_SIZE) {
    const batch = missing.slice(offset, offset + BATCH_SIZE);
    const result = await callOpenAiCompatibleEmbeddings({
      endpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`,
      model: profile.modelId,
      inputs: batch.map((chunk) => chunk.input),
      dimensions: profile.dimensions,
      journal: created.journal,
      runDirectory,
      context: { traceId: root.traceId, parentSpanId: root.spanId },
      timeoutMs: 300_000,
    });
    if (result.status !== "success" || result.vectors === null || result.evidence.usage === null) {
      throw new Error(`Embedding batch failed closed: ${result.evidence.errorClass}: ${result.evidence.errorDetail}`);
    }
    const rows = batch.map((chunk, ordinal) => {
      const vector = result.vectors![ordinal]!;
      const norm = result.evidence.vectorNorms[ordinal]!;
      if (Math.abs(norm - 1) > 0.0001) throw new Error(`Chunk ${chunk.chunkId} embedding norm ${norm} is outside tolerance`);
      return {
        chunkId: chunk.chunkId,
        inputSha256: chunk.inputSha256,
        embeddingJson: canonicalJson(vector),
        embeddingSha256: result.evidence.vectorSha256[ordinal]!,
        normalized: true,
        latencyMs: result.evidence.clientElapsedMs,
        inputFormatVersion: INPUT_FORMAT_VERSION,
        operationId: result.evidence.operationId,
        requestBodySha256: result.evidence.requestBodySha256,
        responseBodySha256: result.evidence.responseBodySha256,
        batchPromptTokens: result.evidence.usage!.promptTokens,
        batchInputCount: batch.length,
        batchOrdinal: ordinal,
        generatedByRunId: run.runId,
        requestArtifactPath: runRelative(result.evidence.requestPath),
        responseArtifactPath: runRelative(result.evidence.responsePath),
      };
    });
    const inserted = await insertBatch(pool, rows);
    if (inserted !== batch.length) throw new Error(`Embedding batch inserted ${inserted}/${batch.length} rows`);
    generated += inserted;
    const batchBody = {
      schemaVersion: 1,
      runId: run.runId,
      buildId,
      corpusId: CORPUS_ID,
      profileKey: PROFILE_KEY,
      inputFormatVersion: INPUT_FORMAT_VERSION,
      batchIndex: Math.floor(offset / BATCH_SIZE),
      batchSize: batch.length,
      chunkIds: batch.map((chunk) => chunk.chunkId),
      orderedInputSetSha256: hashJson(batch.map((chunk) => ({ chunkId: chunk.chunkId, inputSha256: chunk.inputSha256 }))),
      orderedEmbeddingSetSha256: hashJson(rows.map((row) => ({ chunkId: row.chunkId, embeddingSha256: row.embeddingSha256 }))),
      evidence: evidenceForReceipt(result.evidence),
      insertedRows: inserted,
      disposition: "PASS",
    };
    const receipt = { ...batchBody, receiptSha256: hashJson(batchBody) };
    const batchPath = `${batchReceiptDirectory}/${result.evidence.operationId}.json`;
    await atomicWrite(batchPath, `${JSON.stringify(receipt, null, 2)}\n`);
    await created.journal.record("point", "embedding.batch.persisted", {
      traceId: root.traceId,
      parentSpanId: root.spanId,
    }, {
      buildId,
      corpusId: CORPUS_ID,
      profileKey: PROFILE_KEY,
      operationId: result.evidence.operationId,
      chunkCount: batch.length,
      insertedRows: inserted,
      receiptPath: batchPath,
      receiptSha256: receipt.receiptSha256,
    }, { status: "success" });
    batchReceipts.push({ path: batchPath, receiptSha256: receipt.receiptSha256, operationId: result.evidence.operationId, chunkCount: batch.length });
    console.log(JSON.stringify({ stage: "embeddings", profile: PROFILE_KEY, generated, missingAtStart: missing.length, total: state.chunks.length }));
  }

  const metricsAfter = await retainMetricSnapshot("after", created.journal, root);
  const gpuAfter = await retainGpuSnapshot("after", created.journal, root);
  const metricDelta = missing.length === 0
    ? zeroMetricDelta(metricsBefore.summary, metricsAfter.summary)
    : validateEmbeddingMetricDelta(metricsBefore.summary, metricsAfter.summary, Math.ceil(missing.length / BATCH_SIZE), missing.length);
  const final = await verifyFinalState(pool, state.chunks);
  const generationMetrics = missing.length > 0
    ? generationMetricWindow("current_invocation", metricsBefore, metricsAfter, metricDelta, null)
    : await recoverGenerationMetricWindow(metricsBefore, final.operationSummary.length, final.rowCount);
  await created.journal.record("point", "state.embedding_corpus_validated", {
    traceId: root.traceId,
    parentSpanId: root.spanId,
  }, {
    buildId,
    corpusId: CORPUS_ID,
    profileKey: PROFILE_KEY,
    finalRowCount: final.rowCount,
    orderedEmbeddingSetSha256: final.orderedEmbeddingSetSha256,
    metricDelta,
  }, { status: "success" });
  await created.journal.endSpan(root, "success", {
    buildId,
    corpusId: CORPUS_ID,
    profileKey: PROFILE_KEY,
    preexistingRows: state.existing.size,
    generatedRows: generated,
    finalRows: final.rowCount,
    batchCount: batchReceipts.length,
  });
  await created.journal.flush();
  const ingestion = await ingestTelemetryJournal(pool, run.runId, created.path, { batchSize: 100 });
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    buildId,
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    corpusId: CORPUS_ID,
    corpusManifestSha256: state.corpusManifestSha256,
    profileKey: PROFILE_KEY,
    profileHash: hashJson(profileRecord),
    modelId: profile.modelId,
    revision: profile.revision,
    dimensions: profile.dimensions,
    vllmImage: profile.vllmImage,
    inputFormatVersion: INPUT_FORMAT_VERSION,
    portGate: { path: portGatePath, receiptSha256: portGate.receiptSha256 },
    batchSize: BATCH_SIZE,
    preexistingRows: state.existing.size,
    generatedRowsThisInvocation: generated,
    finalRows: final.rowCount,
    batchReceiptsCreatedThisInvocation: batchReceipts,
    batches: final.batchReceipts,
    orderedInputSetSha256: state.orderedInputSetSha256,
    orderedEmbeddingSetSha256: final.orderedEmbeddingSetSha256,
    orderedStoredEmbeddingSetSha256: final.orderedStoredEmbeddingSetSha256,
    operationSetSha256: final.operationSetSha256,
    operationSummary: final.operationSummary,
    normalization: final.normalization,
    sqlStorageConversion: final.sqlStorageConversion,
    rawArtifacts: final.rawArtifacts,
    metrics: {
      invocation: { before: metricsBefore, after: metricsAfter, delta: metricDelta },
      generation: generationMetrics,
    },
    gpu: { before: gpuBefore, after: gpuAfter },
    telemetry: { journalPath: created.path, ingestion },
    frozen: false,
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
  console.log(JSON.stringify({
    runId: run.runId,
    buildId,
    corpusId: CORPUS_ID,
    profileKey: PROFILE_KEY,
    generatedRowsThisInvocation: generated,
    finalRows: final.rowCount,
    batchCount: final.batchReceipts.length,
    orderedEmbeddingSetSha256: final.orderedEmbeddingSetSha256,
    metricDelta,
    generationMetricDelta: generationMetrics.delta,
    disposition: "PASS",
    receiptPath,
    receiptSha256: receipt.receiptSha256,
  }, null, 2));
} catch (error) {
  const errorDetail = safeError(error);
  await created.journal.record("point", "state.embedding_corpus_failed", {
    traceId: root.traceId,
    parentSpanId: root.spanId,
  }, { buildId, corpusId: CORPUS_ID, profileKey: PROFILE_KEY, errorDetail }, { status: "failed" }).catch(() => undefined);
  await created.journal.endSpan(root, "failed", { buildId, errorDetail }).catch(() => undefined);
  await created.journal.flush().catch(() => undefined);
  const ingestion = await ingestTelemetryJournal(pool, run.runId, created.path, { batchSize: 100 }).catch((ingestionError) => ({
    status: "failed",
    errorDetail: safeError(ingestionError),
  }));
  const failureBody = {
    schemaVersion: 1,
    runId: run.runId,
    buildId,
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    corpusId: CORPUS_ID,
    profileKey: PROFILE_KEY,
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
  await pool.close();
}

interface ChunkInput {
  chunkId: string;
  contentSha256: string;
  input: string;
  inputSha256: string;
}

interface ExistingEmbedding {
  chunkId: string;
  inputSha256: string | null;
  inputFormatVersion: string | null;
  embeddingSha256: string;
  normalized: boolean;
}

interface BatchInsertRow {
  chunkId: string;
  inputSha256: string;
  embeddingJson: string;
  embeddingSha256: string;
  normalized: boolean;
  latencyMs: number;
  inputFormatVersion: string;
  operationId: string;
  requestBodySha256: string;
  responseBodySha256: string;
  batchPromptTokens: number;
  batchInputCount: number;
  batchOrdinal: number;
  generatedByRunId: string;
  requestArtifactPath: string;
  responseArtifactPath: string;
}

interface BatchReceipt {
  path: string;
  receiptSha256: string;
  operationId: string;
  chunkCount: number;
}

async function loadBuildState(pool: sql.ConnectionPool): Promise<{
  corpusManifestSha256: string;
  chunks: ChunkInput[];
  existing: Map<string, ExistingEmbedding>;
  orderedInputSetSha256: string;
}> {
  const corpus = await pool.request()
    .input("corpus", sql.VarChar(80), CORPUS_ID)
    .query<{ source_manifest_sha256: string; frozen_at_utc: Date | null }>(`
      SELECT source_manifest_sha256,frozen_at_utc FROM kb.search_corpora WHERE corpus_id=@corpus;
    `);
  const corpusRow = corpus.recordset[0];
  if (corpusRow === undefined) throw new Error(`Unknown corpus ${CORPUS_ID}`);
  if (corpusRow.frozen_at_utc !== null) throw new Error(`Corpus ${CORPUS_ID} is already frozen`);
  const databaseProfile = await pool.request()
    .input("profile", sql.VarChar(80), PROFILE_KEY)
    .query<{ profile_hash: string; model_id: string; revision: string; image_digest: string; dimensions: number }>(`
      SELECT profile_hash,model_id,revision,image_digest,dimensions
      FROM control.embedding_profiles WHERE embedding_profile_id=@profile;
    `);
  const dbProfile = databaseProfile.recordset[0];
  if (dbProfile === undefined) throw new Error(`Embedding profile ${PROFILE_KEY} is absent from SQL`);
  if (dbProfile.profile_hash !== hashJson(profileRecord) || dbProfile.model_id !== profile.modelId ||
      dbProfile.revision !== profile.revision || dbProfile.image_digest !== profile.vllmImage ||
      Number(dbProfile.dimensions) !== profile.dimensions) {
    throw new Error(`SQL embedding profile ${PROFILE_KEY} drifted from the pinned registry`);
  }
  const result = await pool.request()
    .input("corpus", sql.VarChar(80), CORPUS_ID)
    .input("profile", sql.VarChar(80), PROFILE_KEY)
    .query<{
      chunk_id: string;
      content: string;
      content_sha256: string;
      embedding_sha256: string | null;
      input_sha256: string | null;
      input_format_version: string | null;
      normalized: boolean | null;
    }>(`
      SELECT chunk.chunk_id,chunk.content,chunk.content_sha256,
        embedding.embedding_sha256,embedding.input_sha256,
        embedding.input_format_version,embedding.normalized
      FROM kb.runbook_chunks AS chunk
      LEFT JOIN kb.chunk_embeddings AS embedding
        ON embedding.chunk_id=chunk.chunk_id AND embedding.embedding_profile_id=@profile
      WHERE chunk.corpus_id=@corpus
      ORDER BY chunk.chunk_id;
    `);
  if (result.recordset.length !== 480) throw new Error(`Expected 480 primary chunks, found ${result.recordset.length}`);
  const chunks: ChunkInput[] = [];
  const existing = new Map<string, ExistingEmbedding>();
  for (const row of result.recordset) {
    if (sha256(row.content) !== row.content_sha256) throw new Error(`Chunk content hash drift: ${row.chunk_id}`);
    const input = row.content;
    const inputSha256 = sha256(input);
    chunks.push({ chunkId: row.chunk_id, contentSha256: row.content_sha256, input, inputSha256 });
    if (row.embedding_sha256 !== null) {
      const prior = {
        chunkId: row.chunk_id,
        embeddingSha256: row.embedding_sha256,
        inputSha256: row.input_sha256,
        inputFormatVersion: row.input_format_version,
        normalized: row.normalized === true,
      };
      if (prior.inputSha256 !== inputSha256 || prior.inputFormatVersion !== INPUT_FORMAT_VERSION || !prior.normalized) {
        throw new Error(`Existing embedding provenance drift: ${row.chunk_id}`);
      }
      existing.set(row.chunk_id, prior);
    }
  }
  return {
    corpusManifestSha256: corpusRow.source_manifest_sha256,
    chunks,
    existing,
    orderedInputSetSha256: hashJson(chunks.map((chunk) => ({ chunkId: chunk.chunkId, inputSha256: chunk.inputSha256 }))),
  };
}

async function insertBatch(pool: sql.ConnectionPool, rows: BatchInsertRow[]): Promise<number> {
  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    const result = await new sql.Request(transaction)
      .input("corpus", sql.VarChar(80), CORPUS_ID)
      .input("profile", sql.VarChar(80), PROFILE_KEY)
      .input("rows", sql.NVarChar(sql.MAX), JSON.stringify(rows))
      .query<{ inserted: number }>(`
        IF EXISTS (SELECT 1 FROM kb.search_corpora WITH (UPDLOCK,HOLDLOCK) WHERE corpus_id=@corpus AND frozen_at_utc IS NOT NULL)
          THROW 51800, 'corpus froze during embedding build', 1;
        INSERT kb.chunk_embeddings
          (chunk_id,embedding_profile_id,embedding,embedding_sha256,normalized,latency_ms,
           input_sha256,input_format_version,embedding_operation_id,request_body_sha256,
           response_body_sha256,batch_prompt_tokens,batch_input_count,batch_ordinal,
           generated_by_run_id,request_artifact_path,response_artifact_path)
        SELECT source.chunk_id,@profile,CAST(source.embedding_json AS vector(1024)),
          source.embedding_sha256,source.normalized,source.latency_ms,source.input_sha256,
          source.input_format_version,source.operation_id,source.request_body_sha256,
          source.response_body_sha256,source.batch_prompt_tokens,source.batch_input_count,
          source.batch_ordinal,source.generated_by_run_id,source.request_artifact_path,
          source.response_artifact_path
        FROM OPENJSON(@rows) WITH
        (
          chunk_id varchar(120) '$.chunkId',
          embedding_json nvarchar(max) '$.embeddingJson',
          embedding_sha256 char(64) '$.embeddingSha256',
          normalized bit '$.normalized',
          latency_ms decimal(18,3) '$.latencyMs',
          input_sha256 char(64) '$.inputSha256',
          input_format_version varchar(40) '$.inputFormatVersion',
          operation_id uniqueidentifier '$.operationId',
          request_body_sha256 char(64) '$.requestBodySha256',
          response_body_sha256 char(64) '$.responseBodySha256',
          batch_prompt_tokens int '$.batchPromptTokens',
          batch_input_count int '$.batchInputCount',
          batch_ordinal int '$.batchOrdinal',
          generated_by_run_id varchar(120) '$.generatedByRunId',
          request_artifact_path nvarchar(1000) '$.requestArtifactPath',
          response_artifact_path nvarchar(1000) '$.responseArtifactPath'
        ) AS source
        WHERE NOT EXISTS
          (SELECT 1 FROM kb.chunk_embeddings WITH (UPDLOCK,HOLDLOCK)
           WHERE chunk_id=source.chunk_id AND embedding_profile_id=@profile);
        DECLARE @inserted int=@@ROWCOUNT;
        SELECT @inserted AS inserted;
      `);
    await transaction.commit();
    return Number(result.recordset[0]?.inserted ?? -1);
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
}

async function verifyFinalState(pool: sql.ConnectionPool, chunks: ChunkInput[]): Promise<{
  rowCount: number;
  orderedEmbeddingSetSha256: string;
  orderedStoredEmbeddingSetSha256: string;
  operationSetSha256: string;
  operationSummary: Array<Record<string, unknown>>;
  batchReceipts: BatchReceipt[];
  normalization: Record<string, number>;
  sqlStorageConversion: Record<string, number>;
  rawArtifacts: Array<Record<string, unknown>>;
}> {
  const result = await pool.request()
    .input("corpus", sql.VarChar(80), CORPUS_ID)
    .input("profile", sql.VarChar(80), PROFILE_KEY)
    .query<{
      chunk_id: string;
      embedding_json: string;
      dimensions: number;
      sql_norm: number;
      embedding_sha256: string;
      normalized: boolean;
      latency_ms: number;
      input_sha256: string;
      input_format_version: string;
      embedding_operation_id: string;
      request_body_sha256: string;
      response_body_sha256: string;
      batch_prompt_tokens: number;
      batch_input_count: number;
      batch_ordinal: number;
      generated_by_run_id: string;
      request_artifact_path: string;
      response_artifact_path: string;
    }>(`
      SELECT embedding.chunk_id,CAST(embedding.embedding AS nvarchar(max)) AS embedding_json,
        CONVERT(int,VECTORPROPERTY(embedding.embedding,'Dimensions')) AS dimensions,
        CONVERT(float,VECTOR_NORM(embedding.embedding,'norm2')) AS sql_norm,
        embedding.embedding_sha256,embedding.normalized,embedding.latency_ms,
        embedding.input_sha256,embedding.input_format_version,embedding.embedding_operation_id,
        embedding.request_body_sha256,embedding.response_body_sha256,
        embedding.batch_prompt_tokens,embedding.batch_input_count,embedding.batch_ordinal,
        embedding.generated_by_run_id,embedding.request_artifact_path,embedding.response_artifact_path
      FROM kb.chunk_embeddings AS embedding
      INNER JOIN kb.runbook_chunks AS chunk ON chunk.chunk_id=embedding.chunk_id
      WHERE chunk.corpus_id=@corpus AND embedding.embedding_profile_id=@profile
      ORDER BY embedding.chunk_id;
    `);
  if (result.recordset.length !== chunks.length) throw new Error(`Final embedding count ${result.recordset.length}/${chunks.length}`);
  const expectedInputs = new Map(chunks.map((chunk) => [chunk.chunkId, chunk.inputSha256]));
  type FinalRow = (typeof result.recordset)[number];
  const operations = new Map<string, FinalRow[]>();
  const storedVectors = new Map<string, number[]>();
  const storedEmbeddingSha256 = new Map<string, string>();
  const norms: number[] = [];
  for (const row of result.recordset) {
    const vector = JSON.parse(row.embedding_json) as unknown;
    if (!Array.isArray(vector) || vector.length !== profile.dimensions ||
        !vector.every((value) => typeof value === "number" && Number.isFinite(value))) {
      throw new Error(`SQL vector validation failed: ${row.chunk_id}`);
    }
    storedVectors.set(row.chunk_id, vector as number[]);
    storedEmbeddingSha256.set(row.chunk_id, hashJson(vector));
    if (Number(row.dimensions) !== profile.dimensions || !row.normalized || Math.abs(Number(row.sql_norm) - 1) > 0.0001) {
      throw new Error(`SQL vector dimension/norm validation failed: ${row.chunk_id}`);
    }
    if (row.input_sha256 !== expectedInputs.get(row.chunk_id) || row.input_format_version !== INPUT_FORMAT_VERSION || row.generated_by_run_id !== run.runId) {
      throw new Error(`SQL embedding provenance validation failed: ${row.chunk_id}`);
    }
    if (!Number.isFinite(Number(row.latency_ms)) || Number(row.latency_ms) <= 0 ||
        !Number.isSafeInteger(Number(row.batch_prompt_tokens)) || Number(row.batch_prompt_tokens) < 0 ||
        !Number.isSafeInteger(Number(row.batch_input_count)) || Number(row.batch_input_count) < 1 ||
        !Number.isSafeInteger(Number(row.batch_ordinal)) || Number(row.batch_ordinal) < 0) {
      throw new Error(`SQL embedding batch metrics validation failed: ${row.chunk_id}`);
    }
    norms.push(Number(row.sql_norm));
    const operationId = row.embedding_operation_id.toLowerCase();
    const group = operations.get(operationId) ?? [];
    group.push(row);
    operations.set(operationId, group);
  }
  const rawArtifacts: Array<Record<string, unknown>> = [];
  const operationSummary: Array<Record<string, unknown>> = [];
  const validatedBatchReceipts: BatchReceipt[] = [];
  const conversionDrifts: ReturnType<typeof vectorDrift>[] = [];
  for (const [operationId, rows] of [...operations].sort(([left], [right]) => left.localeCompare(right))) {
    const ordered = [...rows].sort((left, right) => Number(left.batch_ordinal) - Number(right.batch_ordinal));
    const expectedCount = Number(ordered[0]!.batch_input_count);
    if (ordered.length !== expectedCount || ordered.some((row, index) => Number(row.batch_ordinal) !== index || Number(row.batch_input_count) !== expectedCount)) {
      throw new Error(`Embedding operation ${operationId} has an incomplete batch`);
    }
    const invariant = <K extends keyof (typeof ordered)[number]>(key: K) => new Set(ordered.map((row) => String(row[key])));
    for (const key of ["request_body_sha256", "response_body_sha256", "batch_prompt_tokens", "latency_ms", "request_artifact_path", "response_artifact_path"] as const) {
      if (invariant(key).size !== 1) throw new Error(`Embedding operation ${operationId} has inconsistent ${key}`);
    }
    const requestPath = `${runDirectory}/${ordered[0]!.request_artifact_path}`;
    const responsePath = `${runDirectory}/${ordered[0]!.response_artifact_path}`;
    const metadataRelativePath = ordered[0]!.request_artifact_path.replace(/request\.json$/, "metadata.json");
    if (metadataRelativePath === ordered[0]!.request_artifact_path) throw new Error(`Embedding operation ${operationId} has an invalid request artifact path`);
    const metadataPath = `${runDirectory}/${metadataRelativePath}`;
    const [requestBytes, responseBytes, metadataBytes] = await Promise.all([readFile(requestPath), readFile(responsePath), readFile(metadataPath)]);
    if (sha256(requestBytes) !== ordered[0]!.request_body_sha256 || sha256(responseBytes) !== ordered[0]!.response_body_sha256) {
      throw new Error(`Embedding operation ${operationId} raw artifact hash mismatch`);
    }
    const rawRequest = JSON.parse(requestBytes.toString("utf8")) as { model?: unknown; input?: unknown; encoding_format?: unknown };
    if (rawRequest.model !== profile.modelId || rawRequest.encoding_format !== "float" || !Array.isArray(rawRequest.input) || rawRequest.input.length !== expectedCount) {
      throw new Error(`Embedding operation ${operationId} raw request contract mismatch`);
    }
    for (const [index, input] of rawRequest.input.entries()) {
      if (typeof input !== "string" || sha256(input) !== ordered[index]!.input_sha256) {
        throw new Error(`Embedding operation ${operationId} raw input ${index} hash mismatch`);
      }
    }
    const rawResponse = parseEmbeddingResponse(responseBytes, expectedCount, profile.dimensions, profile.modelId);
    if (rawResponse.usage.promptTokens !== Number(ordered[0]!.batch_prompt_tokens)) {
      throw new Error(`Embedding operation ${operationId} prompt-token provenance mismatch`);
    }
    const rawMetadata = JSON.parse(metadataBytes.toString("utf8")) as Record<string, unknown>;
    if (rawMetadata.operationId !== operationId || rawMetadata.requestBodySha256 !== ordered[0]!.request_body_sha256 ||
        rawMetadata.responseBodySha256 !== ordered[0]!.response_body_sha256 || rawMetadata.status !== "success") {
      throw new Error(`Embedding operation ${operationId} metadata provenance mismatch`);
    }
    for (const [index, row] of ordered.entries()) {
      const sourceVector = rawResponse.vectors[index]!;
      if (hashJson(sourceVector) !== row.embedding_sha256) throw new Error(`Service vector hash mismatch: ${row.chunk_id}`);
      const storedVector = storedVectors.get(row.chunk_id)!;
      const drift = vectorDrift(sourceVector, storedVector);
      if (drift.maxAbsoluteDifference > 0.0000001 || drift.cosineSimilarity < 0.999999999) {
        throw new Error(`SQL float32 storage drift exceeded tolerance for ${row.chunk_id}: ${JSON.stringify(drift)}`);
      }
      conversionDrifts.push(drift);
    }
    const batchReceiptPath = `${batchReceiptDirectory}/${operationId}.json`;
    const batchReceipt = JSON.parse(await readFile(batchReceiptPath, "utf8")) as Record<string, unknown>;
    const { receiptSha256: batchReceiptSha256, ...batchReceiptBody } = batchReceipt;
    const batchEvidence = batchReceipt.evidence as Record<string, unknown> | undefined;
    if (batchReceipt.disposition !== "PASS" || typeof batchReceiptSha256 !== "string" || batchReceiptSha256 !== hashJson(batchReceiptBody) ||
        String(batchEvidence?.operationId).toLowerCase() !== operationId || Number(batchReceipt.batchSize) !== expectedCount ||
        canonicalJson(batchReceipt.chunkIds) !== canonicalJson(ordered.map((row) => row.chunk_id))) {
      throw new Error(`Embedding operation ${operationId} batch receipt is missing or invalid`);
    }
    validatedBatchReceipts.push({ path: batchReceiptPath, receiptSha256: batchReceiptSha256, operationId, chunkCount: expectedCount });
    rawArtifacts.push({
      operationId,
      requestPath: ordered[0]!.request_artifact_path,
      requestSha256: ordered[0]!.request_body_sha256,
      requestBytes: requestBytes.length,
      responsePath: ordered[0]!.response_artifact_path,
      responseSha256: ordered[0]!.response_body_sha256,
      responseBytes: responseBytes.length,
      metadataPath: runRelative(metadataPath),
      metadataSha256: sha256(metadataBytes),
      metadataBytes: metadataBytes.length,
    });
    operationSummary.push({
      operationId,
      inputCount: expectedCount,
      promptTokens: Number(ordered[0]!.batch_prompt_tokens),
      clientElapsedMs: Number(ordered[0]!.latency_ms),
      firstChunkId: ordered[0]!.chunk_id,
      lastChunkId: ordered.at(-1)!.chunk_id,
    });
  }
  const latencies = operationSummary.map((operation) => Number(operation.clientElapsedMs)).sort((left, right) => left - right);
  const orderedRows = result.recordset.map((row) => ({ chunkId: row.chunk_id, embeddingSha256: row.embedding_sha256 }));
  const orderedStoredRows = result.recordset.map((row) => ({ chunkId: row.chunk_id, storedEmbeddingSha256: storedEmbeddingSha256.get(row.chunk_id)! }));
  return {
    rowCount: result.recordset.length,
    orderedEmbeddingSetSha256: hashJson(orderedRows),
    orderedStoredEmbeddingSetSha256: hashJson(orderedStoredRows),
    operationSetSha256: hashJson(operationSummary),
    operationSummary,
    batchReceipts: validatedBatchReceipts,
    normalization: {
      minimumNorm: Math.min(...norms),
      maximumNorm: Math.max(...norms),
      maximumAbsoluteUnitDeviation: Math.max(...norms.map((norm) => Math.abs(norm - 1))),
      requestLatencyP50Ms: percentile(latencies, 0.5),
      requestLatencyP95Ms: percentile(latencies, 0.95),
      requestLatencyMaximumMs: Math.max(...latencies),
    },
    sqlStorageConversion: {
      comparedVectors: conversionDrifts.length,
      comparedComponents: conversionDrifts.reduce((sum, drift) => sum + drift.dimensions, 0),
      maximumAbsoluteDifference: Math.max(...conversionDrifts.map((drift) => drift.maxAbsoluteDifference)),
      maximumMeanAbsoluteDifference: Math.max(...conversionDrifts.map((drift) => drift.meanAbsoluteDifference)),
      maximumRootMeanSquareError: Math.max(...conversionDrifts.map((drift) => drift.rootMeanSquareError)),
      minimumCosineSimilarity: Math.min(...conversionDrifts.map((drift) => drift.cosineSimilarity)),
      toleranceMaximumAbsoluteDifference: 0.0000001,
      toleranceMinimumCosineSimilarity: 0.999999999,
    },
    rawArtifacts,
  };
}

async function validatedPortGate(path: string): Promise<{ receiptSha256: string }> {
  const parsed = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
  const { receiptSha256, ...body } = parsed;
  if (parsed.disposition !== "PASS" || typeof receiptSha256 !== "string" || receiptSha256 !== hashJson(body)) {
    throw new Error(`Embedding port gate is absent, failed, or hash-invalid: ${path}`);
  }
  const model = parsed.model as Record<string, unknown> | undefined;
  if (model?.modelId !== profile.modelId || model.revision !== profile.revision || model.dimensions !== profile.dimensions || model.vllmImage !== profile.vllmImage) {
    throw new Error("Embedding port gate identity does not match the selected profile");
  }
  return { receiptSha256 };
}

function evidenceForReceipt(evidence: EmbeddingCallEvidence): Record<string, unknown> {
  return {
    operationId: evidence.operationId,
    clientRequestId: evidence.clientRequestId,
    traceId: evidence.traceId,
    spanId: evidence.spanId,
    requestBodySha256: evidence.requestBodySha256,
    requestPath: evidence.requestPath,
    responseBodySha256: evidence.responseBodySha256,
    responsePath: evidence.responsePath,
    metadataPath: evidence.metadataPath,
    startedAtUtc: evidence.startedAtUtc,
    responseHeadersAtUtc: evidence.responseHeadersAtUtc,
    firstContentAtUtc: evidence.firstContentAtUtc,
    bodyFinishedAtUtc: evidence.bodyFinishedAtUtc,
    headersWaitMs: evidence.headersWaitMs,
    bodyReadMs: evidence.bodyReadMs,
    parseMs: evidence.parseMs,
    clientElapsedMs: evidence.clientElapsedMs,
    httpStatus: evidence.httpStatus,
    serviceRequestId: evidence.serviceRequestId,
    responseBytes: evidence.responseBytes,
    vectorCount: evidence.vectorCount,
    vectorSha256: evidence.vectorSha256,
    vectorNorms: evidence.vectorNorms,
    usage: evidence.usage,
    status: evidence.status,
  };
}

async function retainMetricSnapshot(
  phase: "before" | "after",
  journal: FileTelemetryJournal,
  root: StartedSpan,
): Promise<{ phase: string; observedAtUtc: string; path: string; payloadSha256: string; payloadBytes: number; latencyMs: number; summary: EmbeddingMetricSummary }> {
  const endpoint = new URL("/metrics", config.inference.qwenEmbeddingBaseUrl);
  const started = performance.now();
  const response = await fetch(endpoint, { signal: AbortSignal.timeout(10_000) });
  const body = Buffer.from(await response.arrayBuffer());
  const path = `${rawDirectory}/metrics-${phase}.prom`;
  await atomicWrite(path, body);
  if (!response.ok) throw new Error(`Embedding metrics endpoint returned HTTP ${response.status}`);
  const snapshot = {
    phase,
    observedAtUtc: new Date().toISOString(),
    path,
    payloadSha256: sha256(body),
    payloadBytes: body.length,
    latencyMs: round(performance.now() - started),
    summary: summarizeEmbeddingMetrics(body.toString("utf8"), profile.modelId),
  };
  await journal.record("raw_snapshot", "embedding.corpus.metrics", { traceId: root.traceId, parentSpanId: root.spanId }, snapshot, { status: "available" });
  return snapshot;
}

async function recoverGenerationMetricWindow(
  currentBefore: Awaited<ReturnType<typeof retainMetricSnapshot>>,
  expectedRequests: number,
  expectedInputs: number,
): Promise<Record<string, unknown> & { delta: ReturnType<typeof validateEmbeddingMetricDelta> }> {
  const root = `${runDirectory}/raw/embedding-corpus-build`;
  const candidates: Array<{ directory: string; receiptPath: string; receiptSha256: string }> = [];
  for (const entry of await readdir(root, { withFileTypes: true }).catch(() => [])) {
    if (!entry.isDirectory()) continue;
    const receiptPath = `${root}/${entry.name}/failure-receipt.json`;
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(await readFile(receiptPath, "utf8")) as Record<string, unknown>;
    } catch {
      continue;
    }
    const { receiptSha256, ...body } = parsed;
    if (typeof receiptSha256 !== "string" || receiptSha256 !== hashJson(body) || parsed.runId !== run.runId ||
        parsed.corpusId !== CORPUS_ID || parsed.profileKey !== PROFILE_KEY ||
        typeof parsed.errorDetail !== "string" || !parsed.errorDetail.startsWith("SQL vector hash drift:")) continue;
    candidates.push({ directory: `${root}/${entry.name}`, receiptPath, receiptSha256 });
  }
  if (candidates.length !== 1) {
    throw new Error(`Expected one recoverable generation metric baseline, found ${candidates.length}`);
  }
  const candidate = candidates[0]!;
  const beforePath = `${candidate.directory}/metrics-before.prom`;
  const beforeBytes = await readFile(beforePath);
  const beforeSummary = summarizeEmbeddingMetrics(beforeBytes.toString("utf8"), profile.modelId);
  const delta = validateEmbeddingMetricDelta(beforeSummary, currentBefore.summary, expectedRequests, expectedInputs);
  if (delta.embeddingHttpRequests !== expectedRequests || delta.successfulRequests !== expectedInputs || delta.latencyObservations !== expectedInputs) {
    throw new Error(`Recovered generation metrics include unrelated embedding traffic: ${JSON.stringify(delta)}`);
  }
  return generationMetricWindow(
    "recovered_across_fail_closed_sql_storage_verification",
    {
      phase: "generation-before",
      observedAtUtc: null,
      path: beforePath,
      payloadSha256: sha256(beforeBytes),
      payloadBytes: beforeBytes.length,
      latencyMs: null,
      summary: beforeSummary,
    },
    { ...currentBefore, phase: "generation-after" },
    delta,
    { path: candidate.receiptPath, receiptSha256: candidate.receiptSha256 },
  );
}

function generationMetricWindow(
  provenance: string,
  before: Record<string, unknown>,
  after: Record<string, unknown>,
  delta: ReturnType<typeof validateEmbeddingMetricDelta>,
  recoveryReceipt: Record<string, unknown> | null,
): Record<string, unknown> & { delta: ReturnType<typeof validateEmbeddingMetricDelta> } {
  return {
    provenance,
    before,
    after,
    delta,
    recoveryReceipt,
    isolation: "exact request, success, and latency deltas equal the 15 retained generation operations and 480 SQL rows",
  };
}

async function retainGpuSnapshot(
  phase: "before" | "after",
  journal: FileTelemetryJournal,
  root: StartedSpan,
): Promise<{ phase: string; observedAtUtc: string; devicePath: string; processPath: string; deviceSha256: string; processSha256: string }> {
  const [device, processes] = await Promise.all([
    command("nvidia-smi", ["--query-gpu=name,uuid,driver_version,memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader,nounits"]),
    command("nvidia-smi", ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]).catch(() => ""),
  ]);
  const devicePath = `${rawDirectory}/gpu-${phase}.csv`;
  const processPath = `${rawDirectory}/gpu-processes-${phase}.csv`;
  await Promise.all([atomicWrite(devicePath, device), atomicWrite(processPath, processes)]);
  const snapshot = {
    phase,
    observedAtUtc: new Date().toISOString(),
    devicePath,
    processPath,
    deviceSha256: sha256(device),
    processSha256: sha256(processes),
  };
  await journal.record("raw_snapshot", "embedding.corpus.gpu", { traceId: root.traceId, parentSpanId: root.spanId }, snapshot, { status: "available" });
  return snapshot;
}

function zeroMetricDelta(before: EmbeddingMetricSummary, after: EmbeddingMetricSummary) {
  const delta = {
    embeddingHttpRequests: after.embeddingHttpRequests - before.embeddingHttpRequests,
    successfulRequests: after.successfulRequests - before.successfulRequests,
    erroredRequests: after.erroredRequests - before.erroredRequests,
    promptTokens: after.promptTokens - before.promptTokens,
    latencyObservations: after.latencyObservations - before.latencyObservations,
    preemptions: after.preemptions - before.preemptions,
  };
  if (Object.values(delta).some((value) => value < 0)) throw new Error("Embedding metrics reset during resume verification");
  if (delta.erroredRequests !== 0 || delta.preemptions !== 0) throw new Error("Embedding errors or preemptions appeared during resume verification");
  return delta;
}

function runRelative(path: string): string {
  const value = relative(runDirectory, path).replaceAll("\\", "/");
  if (value === "" || value.startsWith("../") || value.startsWith("/")) throw new Error(`Embedding artifact is outside the run directory: ${path}`);
  return value;
}

async function command(executable: string, args: string[]): Promise<string> {
  const result = await execFile(executable, args, { cwd: LAB_ROOT, maxBuffer: 16 * 1024 * 1024 });
  return result.stdout;
}

function percentile(sorted: number[], fraction: number): number {
  if (sorted.length === 0) return 0;
  return sorted[Math.ceil(fraction * sorted.length) - 1]!;
}

function vectorDrift(left: number[], right: number[]): {
  dimensions: number;
  maxAbsoluteDifference: number;
  meanAbsoluteDifference: number;
  rootMeanSquareError: number;
  cosineSimilarity: number;
} {
  if (left.length !== right.length || left.length === 0) throw new Error("Cannot compare SQL and service vectors with unequal dimensions");
  let maximum = 0;
  let absolute = 0;
  let squared = 0;
  let dot = 0;
  let leftSquared = 0;
  let rightSquared = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index]!;
    const rightValue = right[index]!;
    const difference = Math.abs(leftValue - rightValue);
    maximum = Math.max(maximum, difference);
    absolute += difference;
    squared += difference * difference;
    dot += leftValue * rightValue;
    leftSquared += leftValue * leftValue;
    rightSquared += rightValue * rightValue;
  }
  const rawCosine = dot / Math.sqrt(leftSquared * rightSquared);
  return {
    dimensions: left.length,
    maxAbsoluteDifference: maximum,
    meanAbsoluteDifference: absolute / left.length,
    rootMeanSquareError: Math.sqrt(squared / left.length),
    cosineSimilarity: Math.max(-1, Math.min(1, rawCosine)),
  };
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

function safeError(error: unknown): string {
  return (error as { message?: string }).message ?? String(error);
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
