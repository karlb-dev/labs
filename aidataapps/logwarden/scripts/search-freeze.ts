import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { resolveEmbeddingProfile } from "../src/models.js";
import { connect } from "../src/repository.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";

const corpusId = argument("--corpus") ?? "primary-v1";
const profile = resolveEmbeddingProfile(argument("--profile") ?? "qwen3-embedding-0.6b");
const roles = (argument("--roles") ?? "test_id,test_variant_holdout,test_unknown").split(",").map((value) => value.trim()).filter(Boolean);
const expectedModes = ["lexical_fulltext", "vector_exact", "hybrid_rrf", "oracle_runbook", "shuffled_runbook"];
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const manifestPath = `${runDirectory}/manifests/search-freeze.json`;
const receiptPaths = {
  runbooks: `${runDirectory}/knowledge/runbooks-build.json`,
  embeddings: `${runDirectory}/knowledge/embeddings-${corpusId}-${profile.key}.json`,
  retrievalGate: `${runDirectory}/knowledge/retrieval-gate.json`,
  retrievalEvaluation: `${runDirectory}/knowledge/retrieval-evaluation-${safeName(roles.join("-"))}.json`,
  packetAudit: `${runDirectory}/packets/leakage-audit.json`,
};
const receipts = {
  runbooks: await validatedReceipt(receiptPaths.runbooks, "PASS", false),
  embeddings: await validatedReceipt(receiptPaths.embeddings, "PASS"),
  retrievalGate: await validatedReceipt(receiptPaths.retrievalGate, "PASS"),
  retrievalEvaluation: await validatedReceipt(receiptPaths.retrievalEvaluation, "PASS"),
  packetAudit: await validatedReceipt(receiptPaths.packetAudit, "PASS"),
};
const pool = await connect(config.databases.lab, config.databases.controlName, 300_000);

try {
  const corpus = await pool.request().input("corpus", sql.VarChar(80), corpusId).query<{
    corpus_version: string; corpus_kind: string; source_manifest_sha256: string; frozen_at_utc: Date | null;
  }>("SELECT corpus_version,corpus_kind,source_manifest_sha256,frozen_at_utc FROM kb.search_corpora WHERE corpus_id=@corpus;");
  const corpusRow = corpus.recordset[0];
  if (corpusRow === undefined || corpusRow.corpus_kind !== "primary") throw new Error(`Unknown or non-primary corpus ${corpusId}`);

  const inventory = await pool.request().input("corpus", sql.VarChar(80), corpusId)
    .input("profile", sql.VarChar(80), profile.key).query<{
      runbook_count: number; chunk_count: number; embedding_count: number; invalid_norms: number;
      fulltext_status: number; fulltext_items: number;
    }>(`
      SELECT
        (SELECT COUNT(*) FROM kb.runbooks WHERE corpus_id=@corpus AND enabled=1) AS runbook_count,
        (SELECT COUNT(*) FROM kb.runbook_chunks WHERE corpus_id=@corpus) AS chunk_count,
        (SELECT COUNT(*) FROM kb.chunk_embeddings embedding INNER JOIN kb.runbook_chunks chunk ON chunk.chunk_id=embedding.chunk_id WHERE chunk.corpus_id=@corpus AND embedding.embedding_profile_id=@profile) AS embedding_count,
        (SELECT COUNT(*) FROM kb.chunk_embeddings embedding INNER JOIN kb.runbook_chunks chunk ON chunk.chunk_id=embedding.chunk_id WHERE chunk.corpus_id=@corpus AND embedding.embedding_profile_id=@profile AND (embedding.normalized=0 OR ABS(CONVERT(float,VECTOR_NORM(embedding.embedding,'norm2'))-1)>0.0001)) AS invalid_norms,
        CONVERT(int,FULLTEXTCATALOGPROPERTY('logwarden_runbooks_fts','PopulateStatus')) AS fulltext_status,
        CONVERT(int,FULLTEXTCATALOGPROPERTY('logwarden_runbooks_fts','ItemCount')) AS fulltext_items;
    `);
  const counts = inventory.recordset[0]!;
  if (Number(counts.runbook_count) < 1 || Number(counts.chunk_count) < 1) throw new Error("Search corpus is empty");
  if (Number(counts.embedding_count) !== Number(counts.chunk_count) || Number(counts.invalid_norms) !== 0) {
    throw new Error(`Embedding freeze gate failed: ${counts.embedding_count}/${counts.chunk_count}, invalid norms ${counts.invalid_norms}`);
  }
  if (Number(counts.fulltext_status) !== 0 || Number(counts.fulltext_items) < Number(counts.chunk_count)) {
    throw new Error(`Full-text population is incomplete: status ${counts.fulltext_status}, items ${counts.fulltext_items}/${counts.chunk_count}`);
  }

  const evaluation = await pool.request().input("run", sql.VarChar(120), run.runId)
    .input("roles", sql.NVarChar(sql.MAX), JSON.stringify(roles)).query<{
    episode_count: number; cell_count: number; mode_count: number; role_count: number;
    invalid_evaluator_rows: number; query_drift_episodes: number; oracle_failures: number; shuffled_relevance: number;
  }>(`
    WITH selected AS
    (
      SELECT score.*,truth.scenario_group_id,
        run.evaluator_only AS run_evaluator_only,run.status AS run_status
      FROM eval.retrieval_benchmark_results score
      INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=score.episode_id
      INNER JOIN kb.retrieval_runs run ON run.retrieval_run_id=score.retrieval_run_id
      WHERE score.run_id=@run AND score.split_role IN (SELECT value FROM OPENJSON(@roles))
    ), episode_queries AS
    (
      SELECT episode_id FROM selected GROUP BY episode_id HAVING COUNT(DISTINCT query_sha256)<>1
    )
    SELECT COUNT(DISTINCT episode_id) AS episode_count,COUNT(*) AS cell_count,
      COUNT(DISTINCT retrieval_mode) AS mode_count,COUNT(DISTINCT split_role) AS role_count,
      SUM(CASE WHEN evaluator_only<>1 OR run_evaluator_only<>1 OR run_status<>'complete'
        OR EXISTS(SELECT 1 FROM kb.retrieval_results result WHERE result.retrieval_run_id=selected.retrieval_run_id AND result.returned_to_agent=1)
        THEN 1 ELSE 0 END) AS invalid_evaluator_rows,
      (SELECT COUNT(*) FROM episode_queries) AS query_drift_episodes,
      SUM(CASE WHEN retrieval_mode='oracle_runbook' AND
        ((JSON_QUERY(expected_runbooks_json)<>'[]' AND recall_at_k<=0) OR (JSON_QUERY(expected_runbooks_json)='[]' AND ISNULL(no_answer_correct,0)<>1))
        THEN 1 ELSE 0 END) AS oracle_failures,
      SUM(CASE WHEN retrieval_mode='shuffled_runbook' AND recall_at_k<>0 THEN 1 ELSE 0 END) AS shuffled_relevance
    FROM selected;
  `);
  const evaluationRow = evaluation.recordset[0]!;
  const roleRows = await pool.request().input("run", sql.VarChar(120), run.runId)
    .input("roles", sql.NVarChar(sql.MAX), JSON.stringify(roles)).query<{ split_role: string; episodes: number; cells: number }>(`
    SELECT split_role,COUNT(DISTINCT episode_id) episodes,COUNT(*) cells
    FROM eval.retrieval_benchmark_results WHERE run_id=@run AND split_role IN (SELECT value FROM OPENJSON(@roles))
    GROUP BY split_role ORDER BY split_role;
  `);
  const modeRows = await pool.request().input("run", sql.VarChar(120), run.runId)
    .input("roles", sql.NVarChar(sql.MAX), JSON.stringify(roles)).query<{ retrieval_mode: string; cells: number }>(`
    SELECT retrieval_mode,COUNT(*) cells FROM eval.retrieval_benchmark_results
    WHERE run_id=@run AND split_role IN (SELECT value FROM OPENJSON(@roles))
    GROUP BY retrieval_mode ORDER BY retrieval_mode;
  `);
  const actualRoles = roleRows.recordset.map((row) => row.split_role).sort();
  if (hashJson(actualRoles) !== hashJson([...roles].sort())) throw new Error(`Retrieval roles drift: ${actualRoles.join(",")}`);
  const actualModes = modeRows.recordset.map((row) => row.retrieval_mode).sort();
  if (hashJson(actualModes) !== hashJson([...expectedModes].sort())) throw new Error(`Retrieval modes drift: ${actualModes.join(",")}`);
  const expectedCells = Number(evaluationRow.episode_count) * expectedModes.length;
  if (Number(evaluationRow.cell_count) !== expectedCells || Number(evaluationRow.invalid_evaluator_rows) !== 0 ||
      Number(evaluationRow.query_drift_episodes) !== 0 || Number(evaluationRow.oracle_failures) !== 0 ||
      Number(evaluationRow.shuffled_relevance) !== 0) {
    throw new Error(`Retrieval evidence freeze gate failed: ${JSON.stringify(evaluationRow)}`);
  }

  const orderedSets = await orderedInventoryHashes();
  const frozenAtUtc = await freezeCorpus(corpusRow.frozen_at_utc);
  const body = {
    schemaVersion: 1, runId: run.runId, corpusId, corpusVersion: corpusRow.corpus_version,
    corpusManifestSha256: corpusRow.source_manifest_sha256, frozenAtUtc,
    embeddingProfile: { key: profile.key, modelId: profile.modelId, revision: profile.revision, dimensions: profile.dimensions },
    inventory: {
      runbooks: Number(counts.runbook_count), chunks: Number(counts.chunk_count), embeddings: Number(counts.embedding_count),
      invalidNorms: Number(counts.invalid_norms), fulltextStatus: Number(counts.fulltext_status), fulltextItems: Number(counts.fulltext_items),
      ...orderedSets,
    },
    retrievalEvaluation: {
      roles, modes: expectedModes, episodes: Number(evaluationRow.episode_count), cells: Number(evaluationRow.cell_count),
      roleCounts: roleRows.recordset, modeCounts: modeRows.recordset,
      metricPolicy: "unique-runbook-binary-relevance-v1",
      shuffledControlPolicy: "deterministic-hash-ranked-nonacceptable-runbooks-v1",
    },
    evidenceReceipts: Object.fromEntries(Object.entries(receipts).map(([key, value]) => [key, {
      path: receiptPaths[key as keyof typeof receiptPaths], receiptSha256: value.receiptSha256,
    }])),
    exactSearchOnly: true, annEnabled: false, disposition: "FROZEN",
  };
  const manifest = { ...body, freezeSha256: hashJson(body) };
  await atomicWrite(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 0o600);
  await appendExperimentLog(`Search corpus ${corpusId} frozen at ${frozenAtUtc}; ${counts.runbook_count} runbooks, ${counts.chunk_count} chunks, ${counts.embedding_count} embeddings, ${evaluationRow.cell_count} held-out retrieval cells; freeze ${manifest.freezeSha256}.`);
  console.log(JSON.stringify({
    runId: run.runId, corpusId, frozenAtUtc, inventory: body.inventory,
    retrievalEvaluation: body.retrievalEvaluation, disposition: body.disposition,
    manifestPath, freezeSha256: manifest.freezeSha256,
  }, null, 2));
} finally {
  await pool.close();
}

async function freezeCorpus(prior: Date | null): Promise<string> {
  if (prior !== null) return prior.toISOString();
  const result = await pool.request().input("corpus", sql.VarChar(80), corpusId).query<{ frozen_at_utc: Date }>(`
    UPDATE kb.search_corpora SET frozen_at_utc=SYSUTCDATETIME()
    OUTPUT inserted.frozen_at_utc WHERE corpus_id=@corpus AND frozen_at_utc IS NULL;
  `);
  const frozen = result.recordset[0]?.frozen_at_utc;
  if (frozen === undefined) throw new Error(`Corpus ${corpusId} could not be frozen`);
  return frozen.toISOString();
}

async function orderedInventoryHashes() {
  const runbooks = await pool.request().input("corpus", sql.VarChar(80), corpusId)
    .query("SELECT runbook_id,source_sha256,body_sha256,metadata_json,enabled FROM kb.runbooks WHERE corpus_id=@corpus ORDER BY runbook_id;");
  const chunks = await pool.request().input("corpus", sql.VarChar(80), corpusId)
    .query("SELECT chunk_id,runbook_id,ordinal,chunker_version,content_sha256,metadata_json FROM kb.runbook_chunks WHERE corpus_id=@corpus ORDER BY chunk_id;");
  const embeddings = await pool.request().input("corpus", sql.VarChar(80), corpusId).input("profile", sql.VarChar(80), profile.key)
    .query("SELECT embedding.chunk_id,embedding.embedding_sha256,embedding.normalized FROM kb.chunk_embeddings embedding INNER JOIN kb.runbook_chunks chunk ON chunk.chunk_id=embedding.chunk_id WHERE chunk.corpus_id=@corpus AND embedding.embedding_profile_id=@profile ORDER BY embedding.chunk_id;");
  return {
    orderedRunbookSetSha256: hashJson(runbooks.recordset),
    orderedChunkSetSha256: hashJson(chunks.recordset),
    orderedEmbeddingSetSha256: hashJson(embeddings.recordset),
  };
}

async function validatedReceipt(path: string, requiredDisposition: string, requireDisposition = true): Promise<Record<string, unknown> & { receiptSha256: string }> {
  const parsed = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown> & { receiptSha256?: unknown; disposition?: unknown };
  const receiptSha256 = parsed.receiptSha256;
  if (typeof receiptSha256 !== "string") throw new Error(`Receipt has no hash: ${path}`);
  const { receiptSha256: _ignored, ...body } = parsed;
  if (hashJson(body) !== receiptSha256) throw new Error(`Receipt hash drift: ${path}`);
  if (requireDisposition && parsed.disposition !== requiredDisposition) throw new Error(`Receipt is not ${requiredDisposition}: ${path}`);
  return { ...parsed, receiptSha256 };
}

function safeName(value: string): string {
  return value.replace(/[^a-z0-9_.-]+/gi, "-").toLowerCase();
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
