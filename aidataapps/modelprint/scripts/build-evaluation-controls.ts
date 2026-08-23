import { readFile, writeFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson, sha256 } from "../src/hash.js";
import { VllmGateway } from "../src/inference.js";
import { maskNames, stripTemplateResidue } from "../src/prompt-bank.js";
import { resolveRunDirectory } from "../src/run.js";
import { styleVector } from "../src/style.js";

interface EvaluationItem {
  sourceId: string;
  sourceRowId: string;
  promptGroupId: string | null;
  splitRole: "ood-development" | "ood-test";
  text: string;
  metadata: Record<string, unknown>;
}
interface MixedRow { prompt_variant_id: string; prompt_group_id: string; generation_id: number; model_profile_id: string; final_text: string }

const runDirectory = resolveRunDirectory();
const runId = runDirectory.split("/").at(-1)!;
const freeze = JSON.parse(await readFile(`${runDirectory}/manifests/campaign-freeze.json`, "utf8")) as { campaignId: number; campaignHash: string };
const config = loadConfig();
const pool = await new sql.ConnectionPool({ server: config.database.server, port: config.database.port, user: config.database.user,
  password: config.database.password, database: config.database.database, options: { encrypt: false, trustServerCertificate: true }, requestTimeout: 3_600_000 }).connect();
const gateway = new VllmGateway(900_000);
const batch = <T>(rows: T[], size: number) => Array.from({ length: Math.ceil(rows.length / size) }, (_, index) => rows.slice(index * size, (index + 1) * size));
const items: EvaluationItem[] = [];

try {
  const groupRows = await pool.request().query<{ prompt_group_id: string; prompt_source_id: string; source_row_id: string }>(
    "SELECT prompt_group_id,prompt_source_id,source_row_id FROM dbo.prompt_groups;");
  const groups = new Map(groupRows.recordset.map((row) => [`${row.prompt_source_id}:${row.source_row_id}`, row.prompt_group_id]));
  const human = (await readFile("data/ood/human-authored-controls.jsonl", "utf8")).trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as {
    control_id: string; human_response: string; prompt: string; prompt_group_source_id: string; prompt_group_source_row_id: string; provenance: string;
  });
  for (const row of human) items.push({ sourceId: "human-controls", sourceRowId: row.control_id,
    promptGroupId: groups.get(`${row.prompt_group_source_id}:${row.prompt_group_source_row_id}`) ?? null, splitRole: "ood-test", text: row.human_response,
    metadata: { prompt: row.prompt, promptSourceId: row.prompt_group_source_id, promptSourceRowId: row.prompt_group_source_row_id,
      provenance: row.provenance, descriptiveOnly: true } });

  const transformed = await pool.request().input("campaign", sql.BigInt, freeze.campaignId).query<{
    generation_id: number; model_profile_id: string; prompt_group_id: string; final_text: string;
  }>(`WITH ranked AS (SELECT g.generation_id,g.model_profile_id,v.prompt_group_id,g.final_text,
      ROW_NUMBER() OVER(PARTITION BY g.model_profile_id ORDER BY HASHBYTES('SHA2_256',CONVERT(varbinary(8),g.generation_id)),g.generation_id) rn
    FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
    WHERE g.campaign_id=@campaign AND g.split='test_id' AND g.truncated=0 AND g.length_band IN('medium','long'))
    SELECT generation_id,model_profile_id,prompt_group_id,final_text FROM ranked WHERE rn<=100 ORDER BY model_profile_id,rn;`);
  for (const row of transformed.recordset) {
    for (const [transform, text] of [["name-masked-v1", maskNames(row.final_text)], ["template-residue-stripped-v1", stripTemplateResidue(row.final_text)]] as const) {
      items.push({ sourceId: "transformed-controls", sourceRowId: `${transform}:${row.generation_id}`, promptGroupId: row.prompt_group_id,
        splitRole: "ood-development", text, metadata: { transform, sourceGenerationId: row.generation_id, sourceModelProfileId: row.model_profile_id } });
    }
  }

  const mixed = await pool.request().input("campaign", sql.BigInt, freeze.campaignId).query<MixedRow>(`WITH candidates AS (SELECT g.prompt_variant_id,v.prompt_group_id,g.generation_id,g.model_profile_id,g.final_text,
      COUNT(*) OVER(PARTITION BY g.prompt_variant_id) model_rows
    FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id
    WHERE g.campaign_id=@campaign AND g.split='test_id' AND g.truncated=0 AND JSON_VALUE(d.config_json,'$.key')='det')
    SELECT TOP (1200) prompt_variant_id,prompt_group_id,generation_id,model_profile_id,final_text FROM candidates WHERE model_rows>=4
    ORDER BY HASHBYTES('SHA2_256',CONVERT(varbinary(max),prompt_variant_id)),prompt_variant_id,model_profile_id;`);
  const byVariant = new Map<string, MixedRow[]>();
  for (const row of mixed.recordset) byVariant.set(row.prompt_variant_id, [...(byVariant.get(row.prompt_variant_id) ?? []), row]);
  let mixedIndex = 0;
  for (const [variant, values] of byVariant) {
    if (mixedIndex >= 300) break;
    const count = mixedIndex % 2 === 0 ? 2 : 3;
    const selected = values.slice(mixedIndex % values.length).concat(values).slice(0, count);
    if (selected.length < count || new Set(selected.map((row) => row.model_profile_id)).size < count) continue;
    let offset = 0; const boundaries: Array<Record<string, unknown>> = []; const parts: string[] = [];
    for (const row of selected) {
      const text = row.final_text.trim(); const start = offset; parts.push(text); offset += text.length;
      boundaries.push({ start, end: offset, generationId: row.generation_id, modelProfileId: row.model_profile_id }); offset += 2;
    }
    const parentId = `mixed-${String(mixedIndex).padStart(4, "0")}`;
    items.push({ sourceId: "mixed-source", sourceRowId: parentId, promptGroupId: selected[0]!.prompt_group_id,
      splitRole: "ood-test", text: parts.join("\n\n"), metadata: { promptVariantId: variant, boundaries, sourceCount: count } });
    for (const [ordinal, row] of selected.entries()) items.push({ sourceId: "mixed-source-paragraph", sourceRowId: `${parentId}:p${ordinal}`,
      promptGroupId: row.prompt_group_id, splitRole: "ood-test", text: row.final_text.trim(), metadata: { parentId, ordinal,
        sourceGenerationId: row.generation_id, sourceModelProfileId: row.model_profile_id, promptVariantId: variant } });
    mixedIndex += 1;
  }

  for (const rows of batch(items, 200)) {
    const payload = rows.map((row) => ({ sourceId: row.sourceId, sourceRowId: row.sourceRowId, promptGroupId: row.promptGroupId,
      splitRole: row.splitRole, text: row.text, hash: sha256(row.text), metadata: JSON.stringify(row.metadata) }));
    await pool.request().input("run", sql.VarChar(120), runId).input("rows", sql.NVarChar(sql.MAX), JSON.stringify(payload)).query(`
      INSERT dbo.evaluation_items(run_id,source_id,source_row_id,prompt_group_id,split_role,item_text,text_sha256,metadata_json)
      SELECT @run,s.source_id,s.source_row_id,s.prompt_group_id,s.split_role,s.item_text,s.text_sha256,s.metadata_json
      FROM OPENJSON(@rows) WITH(source_id varchar(100) '$.sourceId',source_row_id varchar(180) '$.sourceRowId',prompt_group_id varchar(120) '$.promptGroupId',
        split_role varchar(40) '$.splitRole',item_text nvarchar(max) '$.text',text_sha256 char(64) '$.hash',metadata_json nvarchar(max) '$.metadata') s
      WHERE NOT EXISTS(SELECT 1 FROM dbo.evaluation_items e WHERE e.run_id=@run AND e.source_id=s.source_id AND e.source_row_id=s.source_row_id);`);
  }
  const retained = await pool.request().input("run", sql.VarChar(120), runId).query<{ evaluation_item_id: number; item_text: string }>(`
    SELECT evaluation_item_id,item_text FROM dbo.evaluation_items WHERE run_id=@run ORDER BY evaluation_item_id;`);
  const registry = JSON.parse(await readFile("data/manifests/model-registry-snapshot.json", "utf8")) as { embeddings: Record<string, { modelId: string; dimensions: number }> };
  const profiles = [
    { id: "semantic1024-qwen-v1", registry: "qwen3-embedding-0.6b", baseUrl: config.inference.qwenEmbeddingBaseUrl },
    { id: "semantic1024-bge-v1", registry: "bge-large-en-v1.5", baseUrl: config.inference.bgeEmbeddingBaseUrl },
  ];
  for (const profile of profiles) {
    const missing = await pool.request().input("run", sql.VarChar(120), runId).input("representation", sql.VarChar(80), profile.id).query<{ evaluation_item_id: number; item_text: string }>(`
      SELECT e.evaluation_item_id,e.item_text FROM dbo.evaluation_items e WHERE e.run_id=@run AND NOT EXISTS
      (SELECT 1 FROM dbo.evaluation_vectors v WHERE v.evaluation_item_id=e.evaluation_item_id AND v.representation_id=@representation) ORDER BY e.evaluation_item_id;`);
    const embedding = registry.embeddings[profile.registry]!; await gateway.ready(profile.baseUrl); let done = 0;
    for (const rows of batch(missing.recordset, 64)) {
      const vectors = await gateway.embed(profile.baseUrl, embedding.modelId, embedding.dimensions, rows.map((row) => row.item_text));
      const payload = rows.map((row, index) => { const vector = JSON.stringify(vectors[index]); return { id: row.evaluation_item_id, vector, hash: sha256(vector) }; });
      await pool.request().input("run", sql.VarChar(120), runId).input("representation", sql.VarChar(80), profile.id).input("dimensions", sql.Int, embedding.dimensions)
        .input("rows", sql.NVarChar(sql.MAX), JSON.stringify(payload)).query(`
        INSERT dbo.evaluation_vectors(evaluation_item_id,representation_id,dimensions,vector_json,vector_sha256,created_by_run)
        SELECT s.id,@representation,@dimensions,s.vector,s.hash,@run FROM OPENJSON(@rows) WITH(id bigint '$.id',vector nvarchar(max) '$.vector',hash char(64) '$.hash') s
        WHERE NOT EXISTS(SELECT 1 FROM dbo.evaluation_vectors v WHERE v.evaluation_item_id=s.id AND v.representation_id=@representation);`);
      done += rows.length; if (done % 1024 === 0 || done === missing.recordset.length) console.log(JSON.stringify({ stage: "evaluation-embeddings", representation: profile.id, done, total: missing.recordset.length }));
    }
  }
  const styleMissing = await pool.request().input("run", sql.VarChar(120), runId).query<{ evaluation_item_id: number; item_text: string }>(`
    SELECT e.evaluation_item_id,e.item_text FROM dbo.evaluation_items e WHERE e.run_id=@run AND NOT EXISTS
    (SELECT 1 FROM dbo.evaluation_vectors v WHERE v.evaluation_item_id=e.evaluation_item_id AND v.representation_id='style512-v1');`);
  for (const rows of batch(styleMissing.recordset, 200)) {
    const payload = rows.map((row) => { const vector = JSON.stringify(styleVector(row.item_text)); return { id: row.evaluation_item_id, vector, hash: sha256(vector) }; });
    await pool.request().input("run", sql.VarChar(120), runId).input("rows", sql.NVarChar(sql.MAX), JSON.stringify(payload)).query(`
      INSERT dbo.evaluation_vectors(evaluation_item_id,representation_id,dimensions,vector_json,vector_sha256,created_by_run)
      SELECT s.id,'style512-v1',512,s.vector,s.hash,@run FROM OPENJSON(@rows) WITH(id bigint '$.id',vector nvarchar(max) '$.vector',hash char(64) '$.hash') s
      WHERE NOT EXISTS(SELECT 1 FROM dbo.evaluation_vectors v WHERE v.evaluation_item_id=s.id AND v.representation_id='style512-v1');`);
  }
  const counts = await pool.request().input("run", sql.VarChar(120), runId).query<{ source_id: string; split_role: string; rows: number }>(
    "SELECT source_id,split_role,COUNT(*) rows FROM dbo.evaluation_items WHERE run_id=@run GROUP BY source_id,split_role ORDER BY source_id;");
  const manifest = { schemaVersion: 1, runId, campaignId: freeze.campaignId, campaignHash: freeze.campaignHash, counts: counts.recordset,
    representations: ["semantic1024-qwen-v1", "semantic1024-bge-v1", "style512-v1"], itemCount: retained.recordset.length };
  await writeFile(`${runDirectory}/manifests/evaluation-controls.json`, `${JSON.stringify({ ...manifest, manifestHash: hashJson(manifest) }, null, 2)}\n`);
  await writeFile(`${runDirectory}/raw/evaluation-controls.jsonl`, `${items.map((row) => JSON.stringify(row)).join("\n")}\n`);
  console.log(JSON.stringify(manifest, null, 2));
} finally { await pool.close(); }
