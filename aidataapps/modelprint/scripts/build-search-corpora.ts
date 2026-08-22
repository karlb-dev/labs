import { readFile, writeFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { appendExperimentLog, resolveRunDirectory } from "../src/run.js";

const runDirectory = resolveRunDirectory();
const freeze = JSON.parse(await readFile(`${runDirectory}/manifests/campaign-freeze.json`, "utf8")) as { campaignId: number; campaignHash: string };
const runId = runDirectory.split("/").at(-1)!; const config = loadConfig().database;
const tables = ["search_semantic_train", "search_style_train", "search_residual_train", "search_fingerprint_train", "search_likelihood_train"] as const;
const corpusBase = { schemaVersion: 1, campaignId: Number(freeze.campaignId), campaignHash: freeze.campaignHash, runId,
  eligibility: { split: "train", decode: ["det", "nat-0", "nat-1"], excludedCarrier: "structured-v1", minimumReferenceTokens: 16,
    truncated: false, textView: "raw-final-v1", exactMultiSourceDuplicates: "excluded" } };
const corpusHash = hashJson(corpusBase); const frozenAt = new Date();
const pool = await new sql.ConnectionPool({ server: config.server, port: config.port, user: config.user, password: config.password,
  database: config.database, options: { encrypt: false, trustServerCertificate: true }, requestTimeout: 3_600_000 }).connect();
try {
  const indexes = await pool.request().input("tables", sql.NVarChar(sql.MAX), JSON.stringify(tables)).query<{ table_name: string; index_name: string }>(`
    SELECT OBJECT_NAME(i.object_id) table_name,i.name index_name FROM sys.indexes i
    JOIN OPENJSON(@tables) t ON t.value=OBJECT_NAME(i.object_id) WHERE i.type_desc LIKE '%VECTOR%';`);
  if (indexes.recordset.length) throw new Error(`STOP_CAPABILITY: search corpus already has vector indexes: ${JSON.stringify(indexes.recordset)}`);
  for (const table of tables) await pool.request().batch(`TRUNCATE TABLE dbo.${table};`);
  const request = () => pool.request().input("campaign", sql.BigInt, freeze.campaignId).input("hash", sql.Char(64), corpusHash).input("frozen", sql.DateTime2, frozenAt);
  await request().query(`
    WITH eligible AS (
      SELECT s.semantic_vector_id,s.embedding_profile_id,g.model_profile_id,v.prompt_group_id,t.text_artifact_id,t.text_view_id,g.split,g.length_band,p.stratum,s.embedding,
        ROW_NUMBER() OVER(PARTITION BY s.embedding_profile_id,t.text_artifact_id ORDER BY g.generation_id) rn
      FROM dbo.generations g JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
      JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
      JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id IN('raw-final-v1','name-masked-v1')
      JOIN dbo.semantic_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.representation_id=CONCAT('whole-',t.text_view_id)
      WHERE g.campaign_id=@campaign AND g.split='train' AND JSON_VALUE(d.config_json,'$.key') IN('det','nat-0','nat-1')
        AND v.carrier_id<>'structured-v1' AND g.truncated=0 AND g.reference_token_count>=16 AND t.multi_source_exact_duplicate=0)
    INSERT dbo.search_semantic_train(vector_id,embedding_profile_id,model_profile_id,prompt_group_id,text_artifact_id,split,length_band,stratum,corpus_manifest_hash,frozen_at,embedding,text_view_id)
    SELECT semantic_vector_id,embedding_profile_id,model_profile_id,prompt_group_id,text_artifact_id,split,length_band,stratum,@hash,@frozen,embedding,text_view_id FROM eligible WHERE rn=1;

    WITH eligible AS (
      SELECT s.style_vector_id,g.model_profile_id,v.prompt_group_id,t.text_artifact_id,t.text_view_id,g.split,g.length_band,p.stratum,s.embedding,
        ROW_NUMBER() OVER(PARTITION BY t.text_artifact_id ORDER BY g.generation_id) rn
      FROM dbo.generations g JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
      JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
      JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id IN('raw-final-v1','name-masked-v1')
      JOIN dbo.style_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.representation_id='style512-v1'
      WHERE g.campaign_id=@campaign AND g.split='train' AND JSON_VALUE(d.config_json,'$.key') IN('det','nat-0','nat-1')
        AND v.carrier_id<>'structured-v1' AND g.truncated=0 AND g.reference_token_count>=16 AND t.multi_source_exact_duplicate=0)
    INSERT dbo.search_style_train(vector_id,model_profile_id,prompt_group_id,text_artifact_id,split,length_band,stratum,corpus_manifest_hash,frozen_at,embedding,text_view_id)
    SELECT style_vector_id,model_profile_id,prompt_group_id,text_artifact_id,split,length_band,stratum,@hash,@frozen,embedding,text_view_id FROM eligible WHERE rn=1;

    WITH eligible AS (
      SELECT r.residual_vector_id,r.representation_id,g.model_profile_id,v.prompt_group_id,t.text_artifact_id,g.split,g.length_band,p.stratum,r.embedding,
        ROW_NUMBER() OVER(PARTITION BY r.representation_id,g.generation_id ORDER BY r.residual_vector_id) rn
      FROM dbo.generations g JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
      JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
      JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
      JOIN dbo.residual_vectors r ON r.generation_id=g.generation_id AND r.embedding_profile_id='qwen3-embedding-0.6b'
      WHERE g.campaign_id=@campaign AND g.split='train' AND JSON_VALUE(d.config_json,'$.key') IN('det','nat-0','nat-1')
        AND v.carrier_id<>'structured-v1' AND g.truncated=0 AND g.reference_token_count>=16 AND t.multi_source_exact_duplicate=0)
    INSERT dbo.search_residual_train(vector_id,representation_id,model_profile_id,prompt_group_id,text_artifact_id,split,length_band,stratum,corpus_manifest_hash,frozen_at,embedding)
    SELECT residual_vector_id,representation_id,model_profile_id,prompt_group_id,text_artifact_id,split,length_band,stratum,@hash,@frozen,embedding FROM eligible WHERE rn=1;

    WITH eligible AS (
      SELECT f.fingerprint_vector_id,g.model_profile_id,v.prompt_group_id,t.text_artifact_id,g.split,g.length_band,p.stratum,f.embedding,
        ROW_NUMBER() OVER(PARTITION BY g.generation_id ORDER BY f.fingerprint_vector_id) rn
      FROM dbo.generations g JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
      JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
      JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
      JOIN dbo.fingerprint_vectors f ON f.generation_id=g.generation_id AND f.representation_id='fingerprint64-v1'
      WHERE g.campaign_id=@campaign AND g.split='train' AND JSON_VALUE(d.config_json,'$.key') IN('det','nat-0','nat-1')
        AND v.carrier_id<>'structured-v1' AND g.truncated=0 AND g.reference_token_count>=16 AND t.multi_source_exact_duplicate=0)
    INSERT dbo.search_fingerprint_train(vector_id,model_profile_id,prompt_group_id,text_artifact_id,split,length_band,stratum,corpus_manifest_hash,frozen_at,embedding)
    SELECT fingerprint_vector_id,model_profile_id,prompt_group_id,text_artifact_id,split,length_band,stratum,@hash,@frozen,embedding FROM eligible WHERE rn=1;

    WITH eligible AS (
      SELECT l.likelihood_profile_vector_id,g.model_profile_id,v.prompt_group_id,t.text_artifact_id,g.split,g.length_band,p.stratum,l.embedding
      FROM dbo.generations g JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
      JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
      JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
      JOIN dbo.likelihood_profile_vectors l ON l.generation_id=g.generation_id AND l.representation_id='likelihood-profile8-v1'
      WHERE g.campaign_id=@campaign AND g.split='train' AND JSON_VALUE(d.config_json,'$.key') IN('det','nat-0','nat-1')
        AND v.carrier_id<>'structured-v1' AND g.truncated=0 AND g.reference_token_count>=16 AND t.multi_source_exact_duplicate=0)
    INSERT dbo.search_likelihood_train(vector_id,model_profile_id,prompt_group_id,text_artifact_id,split,length_band,stratum,corpus_manifest_hash,frozen_at,embedding)
    SELECT likelihood_profile_vector_id,model_profile_id,prompt_group_id,text_artifact_id,split,length_band,stratum,@hash,@frozen,embedding FROM eligible;`);
  const counts: Record<string, number> = {};
  for (const table of tables) {
    const result = await pool.request().query<{ count: number; hashes: number }>(`SELECT COUNT(*) count,COUNT(DISTINCT corpus_manifest_hash) hashes FROM dbo.${table};`);
    if (result.recordset[0]!.hashes > 1) throw new Error(`STOP_DATA: ${table} contains multiple corpus hashes`);
    counts[table] = result.recordset[0]!.count;
  }
  const sanity = await pool.request().query<{ semantic: number | null; style: number | null; fingerprint: number | null }>(`
    SELECT (SELECT TOP(1) VECTOR_DISTANCE('cosine',embedding,embedding) FROM dbo.search_semantic_train) semantic,
      (SELECT TOP(1) VECTOR_DISTANCE('cosine',embedding,embedding) FROM dbo.search_style_train) style,
      (SELECT TOP(1) VECTOR_DISTANCE('cosine',embedding,embedding) FROM dbo.search_fingerprint_train) fingerprint;`);
  if (Object.values(sanity.recordset[0] ?? {}).some((value) => value !== null && Math.abs(Number(value)) > 1e-5)) throw new Error("STOP_DATA: vector self-distance sanity failed");
  const manifest = { ...corpusBase, corpusHash, frozenAt: frozenAt.toISOString(), counts, selfDistance: sanity.recordset[0] };
  await writeFile(`${runDirectory}/manifests/vector-corpora.json`, `${JSON.stringify({ ...manifest, manifestHash: hashJson(manifest) }, null, 2)}\n`);
  await appendExperimentLog(`MP-6 exact search corpora frozen; hash=${corpusHash}; counts=${JSON.stringify(counts)}.`);
  console.log(JSON.stringify(manifest, null, 2));
} finally { await pool.close(); }
