import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { headingAwareChunks, runbookCorpusManifest, runbookCorpusSchema } from "../src/runbooks.js";

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const corpusPath = argument("--corpus") ?? "config/runbooks/primary-v1.json";
const corpusBytes = await readFile(corpusPath, "utf8");
const corpus = runbookCorpusSchema.parse(JSON.parse(corpusBytes));
const manifest = runbookCorpusManifest(corpus);
const manifestJson = canonicalJson(manifest);
const manifestSha256 = hashJson(manifest);
const forbiddenPatterns = [
  { name: "scenario_id", pattern: /\b(?:smoke|scenario)[-_][a-z0-9_-]+/i },
  { name: "disposable_database", pattern: /\bLW_[A-Za-z0-9_]+\b/ },
  { name: "lab_message_number", pattern: /\b51\d{3}\b/ },
  { name: "correlation_token", pattern: /LW_EVT_/i },
];
const leakageFindings = corpus.runbooks.flatMap((runbook) => forbiddenPatterns
  .filter(({ pattern }) => pattern.test(runbook.bodyMarkdown))
  .map(({ name }) => ({ runbookId: runbook.runbookId, finding: name })));
if (leakageFindings.length > 0) throw new Error(`Runbook leakage findings: ${JSON.stringify(leakageFindings)}`);

const pool = await connect(config.databases.lab, config.databases.controlName, 30_000);
const transaction = new sql.Transaction(pool);
try {
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  const request = new sql.Request(transaction);
  const existingCorpus = await request.input("corpus", sql.VarChar(80), corpus.corpusId)
    .query<{ source_manifest_sha256: string; frozen_at_utc: Date | null }>(`
      SELECT source_manifest_sha256,frozen_at_utc FROM kb.search_corpora WHERE corpus_id=@corpus;
    `);
  if (existingCorpus.recordset[0]?.frozen_at_utc !== null && existingCorpus.recordset[0]?.frozen_at_utc !== undefined)
    throw new Error(`Runbook corpus ${corpus.corpusId} is already frozen`);
  if (existingCorpus.recordset[0]?.source_manifest_sha256 !== undefined && existingCorpus.recordset[0]!.source_manifest_sha256 !== manifestSha256)
    throw new Error(`Runbook corpus manifest drift for ${corpus.corpusId}`);
  if (existingCorpus.recordset.length === 0) {
    await new sql.Request(transaction)
      .input("corpus", sql.VarChar(80), corpus.corpusId)
      .input("version", sql.VarChar(40), corpus.corpusVersion)
      .input("kind", sql.VarChar(32), corpus.corpusKind)
      .input("manifest", sql.NVarChar(sql.MAX), manifestJson)
      .input("hash", sql.Char(64), manifestSha256)
      .query(`
        INSERT kb.search_corpora(corpus_id,corpus_version,corpus_kind,source_manifest_json,source_manifest_sha256)
        VALUES(@corpus,@version,@kind,@manifest,@hash);
      `);
  }

  for (const runbook of corpus.runbooks) {
    const sourceSha256 = hashJson({ sourceUri: runbook.sourceUri, bodyMarkdown: runbook.bodyMarkdown });
    const bodySha256 = sha256(runbook.bodyMarkdown);
    const existing = await new sql.Request(transaction).input("id", sql.VarChar(100), runbook.runbookId)
      .query<{ source_sha256: string; body_sha256: string }>("SELECT source_sha256,body_sha256 FROM kb.runbooks WHERE runbook_id=@id;");
    if (existing.recordset[0] !== undefined &&
        (existing.recordset[0].source_sha256 !== sourceSha256 || existing.recordset[0].body_sha256 !== bodySha256))
      throw new Error(`Runbook drift for ${runbook.runbookId}`);
    if (existing.recordset.length === 0) await new sql.Request(transaction)
      .input("id", sql.VarChar(100), runbook.runbookId)
      .input("corpus", sql.VarChar(80), corpus.corpusId)
      .input("title", sql.NVarChar(300), runbook.title)
      .input("class", sql.VarChar(80), runbook.incidentClass)
      .input("severity", sql.VarChar(24), runbook.severityFloor)
      .input("uri", sql.NVarChar(1000), runbook.sourceUri)
      .input("source_kind", sql.VarChar(40), runbook.sourceKind)
      .input("source_hash", sql.Char(64), sourceSha256)
      .input("body", sql.NVarChar(sql.MAX), runbook.bodyMarkdown)
      .input("body_hash", sql.Char(64), bodySha256)
      .input("metadata", sql.NVarChar(sql.MAX), canonicalJson(runbook.metadata))
      .query(`
        INSERT kb.runbooks(runbook_id,corpus_id,title,incident_class,severity_floor,source_uri,
          source_kind,source_sha256,body_markdown,body_sha256,metadata_json)
        VALUES(@id,@corpus,@title,@class,@severity,@uri,@source_kind,@source_hash,@body,@body_hash,@metadata);
      `);
    for (const chunk of headingAwareChunks(runbook)) {
      const existingChunk = await new sql.Request(transaction).input("id", sql.VarChar(120), chunk.chunkId)
        .query<{ content_sha256: string }>("SELECT content_sha256 FROM kb.runbook_chunks WHERE chunk_id=@id;");
      if (existingChunk.recordset[0]?.content_sha256 !== undefined && existingChunk.recordset[0]!.content_sha256 !== chunk.contentSha256)
        throw new Error(`Runbook chunk drift for ${chunk.chunkId}`);
      if (existingChunk.recordset.length === 0) await new sql.Request(transaction)
        .input("id", sql.VarChar(120), chunk.chunkId)
        .input("runbook", sql.VarChar(100), chunk.runbookId)
        .input("corpus", sql.VarChar(80), corpus.corpusId)
        .input("ordinal", sql.Int, chunk.ordinal)
        .input("heading", sql.NVarChar(800), chunk.headingPath)
        .input("content", sql.NVarChar(sql.MAX), chunk.content)
        .input("tokens", sql.Int, chunk.tokenCount)
        .input("hash", sql.Char(64), chunk.contentSha256)
        .input("metadata", sql.NVarChar(sql.MAX), canonicalJson({ schemaVersion: 1, chunker: "heading-v1", tokenCounter: "lexical-v1" }))
        .query(`
          INSERT kb.runbook_chunks(chunk_id,runbook_id,corpus_id,ordinal,heading_path,content,
            token_count,chunker_version,content_sha256,metadata_json)
          VALUES(@id,@runbook,@corpus,@ordinal,@heading,@content,@tokens,'heading-v1',@hash,@metadata);
        `);
    }
  }
  await transaction.commit();
} catch (error) {
  await transaction.rollback().catch(() => undefined);
  throw error;
} finally {
  await pool.close();
}

const population = await waitForPopulation();
const campaignId = await attachBuildingCampaign();
const chunkCount = corpus.runbooks.reduce((total, runbook) => total + headingAwareChunks(runbook).length, 0);
const receiptBody = {
  schemaVersion: 1,
  corpusId: corpus.corpusId,
  corpusVersion: corpus.corpusVersion,
  corpusPath,
  corpusFileSha256: sha256(corpusBytes),
  manifestSha256,
  campaignId,
  runbookCount: corpus.runbooks.length,
  chunkCount,
  chunkerVersion: "heading-v1",
  tokenCounter: "lexical-v1",
  leakageFindings,
  fulltextPopulation: population,
  frozen: false,
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/knowledge/runbooks-build.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));

async function waitForPopulation(): Promise<Record<string, unknown>> {
  const pool = await connect(config.databases.lab, config.databases.controlName, 30_000);
  const deadline = Date.now() + 60_000;
  try {
    while (Date.now() < deadline) {
      const result = await pool.request().query<{ status: number; item_count: number }>(`
        SELECT CONVERT(int,FULLTEXTCATALOGPROPERTY('logwarden_runbooks_fts','PopulateStatus')) AS status,
               CONVERT(int,FULLTEXTCATALOGPROPERTY('logwarden_runbooks_fts','ItemCount')) AS item_count;
      `);
      const row = result.recordset[0]!;
      if (Number(row.status) === 0 && Number(row.item_count) >= corpus.runbooks.length)
        return { status: Number(row.status), itemCount: Number(row.item_count), readyAtUtc: new Date().toISOString() };
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    throw new Error("Full-text runbook population did not complete within 60 seconds");
  } finally {
    await pool.close();
  }
}

async function attachBuildingCampaign(): Promise<string> {
  const pool = await connect(config.databases.lab, config.databases.controlName, 30_000);
  try {
    const result = await pool.request()
      .input("run", sql.VarChar(120), run.runId)
      .input("manifest", sql.Char(64), manifestSha256)
      .query<{ campaign_id: string; runbook_manifest_hash: string }>(`
        UPDATE campaign
        SET runbook_manifest_hash=@manifest
        FROM control.campaigns AS campaign
        INNER JOIN control.runs AS run ON run.campaign_id=campaign.campaign_id
        WHERE run.run_id=@run AND campaign.status='building'
          AND (campaign.runbook_manifest_hash IS NULL OR campaign.runbook_manifest_hash=@manifest);
        SELECT campaign.campaign_id,campaign.runbook_manifest_hash
        FROM control.campaigns AS campaign
        INNER JOIN control.runs AS run ON run.campaign_id=campaign.campaign_id
        WHERE run.run_id=@run;
      `);
    const row = result.recordset[0];
    if (row === undefined || row.runbook_manifest_hash !== manifestSha256)
      throw new Error(`Building campaign refused runbook manifest ${manifestSha256}`);
    return String(row.campaign_id);
  } finally {
    await pool.close();
  }
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
