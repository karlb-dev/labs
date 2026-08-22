import { readFile, writeFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson, sha256 } from "../src/hash.js";
import { VllmGateway } from "../src/inference.js";
import { resolveRunDirectory, valueAfter } from "../src/run.js";
import { STYLE_SCHEMA_HASH, styleVector } from "../src/style.js";

const stage = valueAfter("--stage") ?? "all";
if (!new Set(["segments", "style", "embeddings", "all"]).has(stage)) throw new Error(`Unknown stage ${stage}`);
const runDirectory = resolveRunDirectory();
const freeze = JSON.parse(await readFile(`${runDirectory}/manifests/campaign-freeze.json`, "utf8")) as { campaignId: number; campaignHash: string };
const robustness = await readFile(`${runDirectory}/manifests/robustness-freeze.json`, "utf8").then((value) => JSON.parse(value) as { campaignId: number }).catch(() => null);
const campaignIds = [Number(freeze.campaignId), ...(robustness ? [Number(robustness.campaignId)] : [])];
if (!campaignIds.every(Number.isSafeInteger)) throw new Error("Invalid campaign ID in run manifests");
const campaignSql = campaignIds.join(",");
const config = loadConfig(); const runId = runDirectory.split("/").at(-1)!;
const pool = await new sql.ConnectionPool({ server: config.database.server, port: config.database.port, user: config.database.user,
  password: config.database.password, database: config.database.database, options: { encrypt: false, trustServerCertificate: true }, requestTimeout: 900_000 }).connect();
const gateway = new VllmGateway(900_000);
const batch = <T>(rows: T[], size: number) => Array.from({ length: Math.ceil(rows.length / size) }, (_, index) => rows.slice(index * size, (index + 1) * size));
const insertJson = (rows: unknown[], query: string) => pool.request().input("rows", sql.NVarChar(sql.MAX), JSON.stringify(rows)).query(query);

function scalarStyleFeatures(text: string): Record<string, number> {
  const words = text.match(/[\p{L}\p{N}'’-]+/gu) ?? []; const sentences = text.split(/(?<=[.!?])\s+/).filter(Boolean);
  const paragraphs = text.split(/\n\s*\n/).filter((part) => part.trim()); const lines = text.split("\n");
  const count = (pattern: RegExp) => [...text.matchAll(pattern)].length; const chars = Math.max(text.length, 1); const wordCount = Math.max(words.length, 1);
  const mean = (values: number[]) => values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  const sd = (values: number[]) => { const average = mean(values); return Math.sqrt(mean(values.map((value) => (value - average) ** 2))); };
  return { char_count: text.length, word_count: words.length, sentence_count: sentences.length, paragraph_count: paragraphs.length,
    mean_sentence_chars: mean(sentences.map((row) => row.length)), sentence_chars_sd: sd(sentences.map((row) => row.length)),
    mean_paragraph_chars: mean(paragraphs.map((row) => row.length)), type_token_ratio: new Set(words.map((row) => row.toLowerCase())).size / wordCount,
    em_dash_rate: count(/—/g) / chars, en_dash_rate: count(/–/g) / chars, semicolon_rate: count(/;/g) / chars, colon_rate: count(/:/g) / chars,
    curly_quote_rate: count(/[“”‘’]/g) / chars, straight_quote_rate: count(/["']/g) / chars, header_rate: count(/^#{1,6}\s/gm) / Math.max(lines.length, 1),
    bold_span_rate: count(/\*\*[^*]+\*\*/g) / wordCount, bullet_rate: count(/^\s*[-*•]\s/gm) / Math.max(lines.length, 1),
    numbered_list_rate: count(/^\s*\d+[.)]\s/gm) / Math.max(lines.length, 1), code_fence_rate: count(/```/g) / Math.max(paragraphs.length, 1),
    emoji_rate: count(/\p{Extended_Pictographic}/gu) / chars, opener_polite: /^(?:Certainly|Sure|Great question)\b/i.test(text) ? 1 : 0,
    closer_offer: /(?:Let me know|I hope this helps)[.!]?\s*$/i.test(text) ? 1 : 0, contraction_rate: words.filter((word) => /['’]/.test(word)).length / wordCount,
    parenthetical_rate: count(/\([^)]*\)/g) / Math.max(sentences.length, 1), hedge_rate: count(/\b(?:may|might|could|perhaps|possibly|likely|appears?|seems?)\b/gi) / wordCount,
    first_person_rate: count(/\b(?:I|me|my|mine|we|us|our|ours)\b/gi) / wordCount, second_person_rate: count(/\b(?:you|your|yours)\b/gi) / wordCount,
    ellipsis_char_rate: count(/…/g) / chars, ellipsis_three_rate: count(/\.\.\./g) / chars, trailing_space_rate: count(/[ \t]+$/gm) / Math.max(lines.length, 1),
    double_space_rate: count(/  +/g) / chars };
}

async function style() {
  const result = await pool.request().query<{ id: number; text: string }>(`
    SELECT DISTINCT t.text_artifact_id AS id,t.artifact_text AS text FROM dbo.text_artifacts t
    JOIN dbo.generation_text_artifacts m ON m.text_artifact_id=t.text_artifact_id JOIN dbo.generations g ON g.generation_id=m.generation_id
    WHERE g.campaign_id IN (${campaignSql}) AND NOT EXISTS (SELECT 1 FROM dbo.style_vectors s WHERE s.text_artifact_id=t.text_artifact_id AND s.representation_id='style512-v1');`);
  let done = 0;
  for (const rows of batch(result.recordset, 1000)) {
    const vectors = rows.map((row) => ({ id: row.id, vector: JSON.stringify(styleVector(row.text)), scalars: JSON.stringify(scalarStyleFeatures(row.text)) }));
    await insertJson(vectors, `
      INSERT dbo.output_scalar_features(text_artifact_id,feature_schema_hash,features_json)
      SELECT s.id,@schema,s.scalars FROM OPENJSON(@rows) WITH (id bigint '$.id',scalars nvarchar(max) '$.scalars') s
      WHERE NOT EXISTS (SELECT 1 FROM dbo.output_scalar_features f WHERE f.text_artifact_id=s.id AND f.feature_schema_hash=@schema);
      INSERT dbo.style_vectors(text_artifact_id,representation_id,feature_schema_hash,embedding,created_by_run)
      SELECT s.id,'style512-v1',@schema,CAST(s.vector AS vector(512)),@run FROM OPENJSON(@rows) WITH (id bigint '$.id',vector nvarchar(max) '$.vector') s
      WHERE NOT EXISTS (SELECT 1 FROM dbo.style_vectors v WHERE v.text_artifact_id=s.id AND v.representation_id='style512-v1');`.replaceAll("@schema", `'${STYLE_SCHEMA_HASH}'`).replaceAll("@run", `'${runId}'`));
    done += rows.length; console.log(JSON.stringify({ stage: "style", done, total: result.recordset.length }));
  }
  return done;
}

async function embeddings() {
  const registry = JSON.parse(await readFile("data/manifests/model-registry-snapshot.json", "utf8")) as { embeddings: Record<string, { modelId: string; dimensions: number }> };
  const profiles = [
    { key: "qwen3-embedding-0.6b", baseUrl: config.inference.qwenEmbeddingBaseUrl },
    { key: "bge-large-en-v1.5", baseUrl: config.inference.bgeEmbeddingBaseUrl },
  ];
  let totalDone = 0;
  for (const selected of profiles) {
    const profile = registry.embeddings[selected.key]!; await gateway.ready(selected.baseUrl);
    const prompts = await pool.request().input("profile", sql.VarChar(80), selected.key).query<{ id: string; text: string }>(`
      SELECT DISTINCT v.prompt_variant_id AS id,v.rendered_text AS text FROM dbo.generation_jobs j JOIN dbo.prompt_variants v ON v.prompt_variant_id=j.prompt_variant_id
      WHERE j.campaign_id IN (${campaignSql}) AND NOT EXISTS (SELECT 1 FROM dbo.prompt_embeddings p WHERE p.prompt_variant_id=v.prompt_variant_id AND p.embedding_profile_id=@profile);`);
    for (const rows of batch(prompts.recordset, 64)) {
      const vectors = await gateway.embed(selected.baseUrl, profile.modelId, profile.dimensions, rows.map((row) => row.text));
      await insertJson(rows.map((row, index) => ({ id: row.id, vector: JSON.stringify(vectors[index]), sha: sha256(JSON.stringify(vectors[index])) })), `
        INSERT dbo.prompt_embeddings(prompt_variant_id,embedding_profile_id,embedding,embedding_sha256)
        SELECT s.id,'${selected.key}',CAST(s.vector AS vector(1024)),s.sha FROM OPENJSON(@rows) WITH (id varchar(80) '$.id',vector nvarchar(max) '$.vector',sha char(64) '$.sha') s
        WHERE NOT EXISTS (SELECT 1 FROM dbo.prompt_embeddings p WHERE p.prompt_variant_id=s.id AND p.embedding_profile_id='${selected.key}');`);
      totalDone += rows.length;
    }
    const artifacts = await pool.request().input("profile", sql.VarChar(80), selected.key).query<{ id: number; view: string; text: string }>(`
      SELECT DISTINCT t.text_artifact_id AS id,t.text_view_id AS view,t.artifact_text AS text FROM dbo.text_artifacts t JOIN dbo.generation_text_artifacts m ON m.text_artifact_id=t.text_artifact_id
      JOIN dbo.generations g ON g.generation_id=m.generation_id WHERE g.campaign_id IN (${campaignSql}) AND NOT EXISTS
      (SELECT 1 FROM dbo.semantic_vectors v WHERE v.text_artifact_id=t.text_artifact_id AND v.embedding_profile_id=@profile AND v.representation_id=CONCAT('whole-',t.text_view_id));`);
    let embedded = 0;
    for (const rows of batch(artifacts.recordset, 64)) {
      const vectors = await gateway.embed(selected.baseUrl, profile.modelId, profile.dimensions, rows.map((row) => row.text));
      await insertJson(rows.map((row, index) => ({ id: row.id, representation: `whole-${row.view}`, vector: JSON.stringify(vectors[index]), sha: sha256(JSON.stringify(vectors[index])) })), `
        INSERT dbo.semantic_vectors(text_artifact_id,segment_id,embedding_profile_id,representation_id,embedding,embedding_sha256,created_by_run)
        SELECT s.id,NULL,'${selected.key}',s.representation,CAST(s.vector AS vector(1024)),s.sha,'${runId}' FROM OPENJSON(@rows) WITH
          (id bigint '$.id',representation varchar(80) '$.representation',vector nvarchar(max) '$.vector',sha char(64) '$.sha') s
        WHERE NOT EXISTS (SELECT 1 FROM dbo.semantic_vectors v WHERE v.text_artifact_id=s.id AND v.embedding_profile_id='${selected.key}' AND v.representation_id=s.representation);`);
      embedded += rows.length; totalDone += rows.length;
      if (embedded % 1024 === 0 || embedded === artifacts.recordset.length) console.log(JSON.stringify({ stage: "embeddings", profile: selected.key, embedded, total: artifacts.recordset.length }));
    }
  }
  return totalDone;
}

try {
  const summary: Record<string, unknown> = { schemaVersion: 1, campaignIds, campaignHash: freeze.campaignHash, runId, startedAt: new Date().toISOString() };
  if (stage === "segments" || stage === "all") {
    const { execFileSync } = await import("node:child_process");
    summary.segments = JSON.parse(execFileSync("./scripts/python.sh", ["analysis/prepare_text.py", "--run", runDirectory], { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 }));
  }
  if (stage === "style" || stage === "all") summary.styleArtifacts = await style();
  if (stage === "embeddings" || stage === "all") summary.embeddingRows = await embeddings();
  summary.finishedAt = new Date().toISOString();
  await writeFile(`${runDirectory}/manifests/features-${stage}.json`, `${JSON.stringify({ ...summary, manifestHash: hashJson(summary) }, null, 2)}\n`);
  console.log(JSON.stringify(summary, null, 2));
} finally { await pool.close(); }
