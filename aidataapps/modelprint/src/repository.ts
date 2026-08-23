import sql from "mssql";
import type { AppConfig } from "./config.js";

export interface ExactNeighbor {
  vectorId: number;
  modelProfileId: string;
  promptGroupId: string;
  textArtifactId: number;
  distance: number;
}

const searchSpaces = {
  semantic1024: { table: "dbo.search_semantic_train", dimensions: 1024, discriminator: "embedding_profile_id", defaultSubspace: "qwen3-embedding-0.6b", viewColumn: "text_view_id" },
  residual1024: { table: "dbo.search_residual_train", dimensions: 1024, discriminator: "representation_id", defaultSubspace: "residual-ridge-v1", viewColumn: null },
  style512: { table: "dbo.search_style_train", dimensions: 512, discriminator: null, defaultSubspace: null, viewColumn: "text_view_id" },
  fingerprint64: { table: "dbo.search_fingerprint_train", dimensions: 64, discriminator: null, defaultSubspace: null, viewColumn: null },
  likelihood8: { table: "dbo.search_likelihood_train", dimensions: 8, discriminator: null, defaultSubspace: null, viewColumn: null },
} as const;

export class SqlServerRepository {
  private constructor(readonly pool: sql.ConnectionPool) {}

  static async connect(config: AppConfig["database"], database = config.database): Promise<SqlServerRepository> {
    const pool = await new sql.ConnectionPool({
      server: config.server,
      port: config.port,
      user: config.user,
      password: config.password,
      database,
      options: { encrypt: false, trustServerCertificate: true },
      pool: { min: 0, max: 20, idleTimeoutMillis: 60_000 },
      requestTimeout: 600_000,
    }).connect();
    return new SqlServerRepository(pool);
  }

  async exactNeighbors(
    space: keyof typeof searchSpaces,
    embedding: number[],
    options: { k: number; candidateK?: number; excludePromptGroupId?: string; subspace?: string; textView?: string } = { k: 10 },
  ): Promise<ExactNeighbor[]> {
    const definition = searchSpaces[space];
    if (embedding.length !== definition.dimensions || !embedding.every(Number.isFinite)) {
      throw new Error(`${space} requires a finite ${definition.dimensions}-dimensional vector`);
    }
    const k = Math.max(1, Math.min(options.k, 100));
    const candidateK = Math.max(k, Math.min(options.candidateK ?? k * 10, 2000));
    const result = await this.pool.request()
      .input("embedding", sql.NVarChar(sql.MAX), JSON.stringify(embedding))
      .input("k", sql.Int, k)
      .input("candidateK", sql.Int, candidateK)
      .input("excludePromptGroupId", sql.VarChar(120), options.excludePromptGroupId ?? null)
      .input("subspace", sql.VarChar(80), options.subspace ?? definition.defaultSubspace)
      .input("textView", sql.VarChar(80), options.textView ?? "raw-final-v1")
      .query<{
        vectorId: number; modelProfileId: string; promptGroupId: string; textArtifactId: number; distance: number;
      }>(`
        WITH candidates AS
        (
          SELECT TOP (@candidateK)
            vector_id AS vectorId,
            model_profile_id AS modelProfileId,
            prompt_group_id AS promptGroupId,
            text_artifact_id AS textArtifactId,
            VECTOR_DISTANCE('cosine', embedding, CAST(@embedding AS VECTOR(${definition.dimensions}))) AS distance
          FROM ${definition.table}
          WHERE (@excludePromptGroupId IS NULL OR prompt_group_id <> @excludePromptGroupId)
            AND ${definition.discriminator ? `${definition.discriminator}=@subspace` : "1=1"}
            AND ${definition.viewColumn ? `${definition.viewColumn}=@textView` : "1=1"}
          ORDER BY distance, vector_id
        ),
        deduped AS
        (
          SELECT *,
            ROW_NUMBER() OVER (PARTITION BY promptGroupId ORDER BY distance, vectorId) AS rnPrompt,
            ROW_NUMBER() OVER (PARTITION BY textArtifactId ORDER BY distance, vectorId) AS rnText
          FROM candidates
        )
        SELECT TOP (@k) vectorId, modelProfileId, promptGroupId, textArtifactId, distance
        FROM deduped
        WHERE rnPrompt = 1 AND rnText = 1
        ORDER BY distance, vectorId;
      `);
    return result.recordset.map((row) => ({ ...row, distance: Number(row.distance) }));
  }

  async exactVote(
    space: keyof typeof searchSpaces,
    embedding: number[],
    options: { k: number; tau: number; excludePromptGroupId?: string; subspace?: string; textView?: string },
  ): Promise<Array<{ modelProfileId: string; voteShare: number; neighborCount: number; independentGroups: number; nearestDistance: number }>> {
    const definition = searchSpaces[space];
    if (embedding.length !== definition.dimensions || !embedding.every(Number.isFinite)) {
      throw new Error(`${space} requires a finite ${definition.dimensions}-dimensional vector`);
    }
    const safeK = Math.max(1, Math.min(options.k, 100));
    const result = await this.pool.request()
      .input("embedding", sql.NVarChar(sql.MAX), JSON.stringify(embedding))
      .input("k", sql.Int, safeK)
      .input("candidateK", sql.Int, Math.min(safeK * 10, 2000))
      .input("tau", sql.Float, options.tau)
      .input("excludePromptGroupId", sql.VarChar(120), options.excludePromptGroupId ?? null)
      .input("subspace", sql.VarChar(80), options.subspace ?? definition.defaultSubspace)
      .input("textView", sql.VarChar(80), options.textView ?? "raw-final-v1")
      .query<{
      modelProfileId: string; voteShare: number; neighborCount: number; independentGroups: number; nearestDistance: number;
    }>(`
      WITH candidates AS
      (
        SELECT TOP (@candidateK)
          vector_id,
          model_profile_id,
          prompt_group_id,
          text_artifact_id,
          VECTOR_DISTANCE('cosine', embedding, CAST(@embedding AS VECTOR(${definition.dimensions}))) AS distance
        FROM ${definition.table}
        WHERE (@excludePromptGroupId IS NULL OR prompt_group_id <> @excludePromptGroupId)
          AND ${definition.discriminator ? `${definition.discriminator}=@subspace` : "1=1"}
          AND ${definition.viewColumn ? `${definition.viewColumn}=@textView` : "1=1"}
        ORDER BY distance, vector_id
      ),
      ranked AS
      (
        SELECT *,
          ROW_NUMBER() OVER (PARTITION BY prompt_group_id ORDER BY distance, vector_id) AS rn_prompt,
          ROW_NUMBER() OVER (PARTITION BY text_artifact_id ORDER BY distance, vector_id) AS rn_text
        FROM candidates
      ),
      nn AS
      (
        SELECT TOP (@k) * FROM ranked
        WHERE rn_prompt=1 AND rn_text=1
        ORDER BY distance, vector_id
      ),
      totals AS
      (
        SELECT model_profile_id,
          SUM(EXP(-distance / @tau)) AS weight,
          COUNT(*) AS neighbor_count,
          COUNT(DISTINCT prompt_group_id) AS independent_groups,
          MIN(distance) AS nearest_distance
        FROM nn GROUP BY model_profile_id
      )
      SELECT model_profile_id AS modelProfileId,
        weight / SUM(weight) OVER () AS voteShare,
        neighbor_count AS neighborCount,
        independent_groups AS independentGroups,
        nearest_distance AS nearestDistance
      FROM totals ORDER BY voteShare DESC, model_profile_id;
    `);
    return result.recordset.map((row) => ({ ...row, voteShare: Number(row.voteShare), nearestDistance: Number(row.nearestDistance) }));
  }

  async ready(): Promise<void> { await this.pool.request().query("SELECT 1 AS ready;"); }
  async latestCalibration(): Promise<Record<string, unknown> | null> {
    const result = await this.pool.request().query<Record<string, unknown>>("SELECT TOP (1) calibration_model_id,method,calibration_manifest_hash,created_at FROM dbo.calibration_models ORDER BY calibration_model_id DESC;");
    return result.recordset[0] ?? null;
  }
  async evaluations(): Promise<Record<string, unknown>[]> {
    const result = await this.pool.request().query<Record<string, unknown>>("SELECT prediction_run_id,run_id,suite,representation_id,method,created_at FROM dbo.prediction_runs ORDER BY prediction_run_id DESC;");
    return result.recordset;
  }
  async evaluation(id: number): Promise<Record<string, unknown> | null> {
    const result = await this.pool.request().input("id", sql.BigInt, id).query<Record<string, unknown>>(`
      SELECT r.prediction_run_id,r.run_id,r.suite,r.representation_id,r.method,r.config_json,r.created_at,
        COUNT(p.generation_id) AS prediction_count,SUM(CASE WHEN p.decision='attributed' THEN 1 ELSE 0 END) AS attributed_count
      FROM dbo.prediction_runs r LEFT JOIN dbo.predictions p ON p.prediction_run_id=r.prediction_run_id
      WHERE r.prediction_run_id=@id GROUP BY r.prediction_run_id,r.run_id,r.suite,r.representation_id,r.method,r.config_json,r.created_at;`);
    return result.recordset[0] ?? null;
  }
  async campaigns(id?: number): Promise<Record<string, unknown>[]> {
    const request=this.pool.request().input("id",sql.BigInt,id ?? null);
    const result=await request.query<Record<string,unknown>>(`SELECT c.campaign_id,c.campaign_name,c.tier,c.campaign_hash,MAX(f.freeze_hash) freeze_hash,COUNT(j.generation_job_id) expected_jobs,c.status,c.created_at,
      SUM(CASE WHEN j.status='complete' THEN 1 ELSE 0 END) complete_jobs,SUM(CASE WHEN j.status='failed' THEN 1 ELSE 0 END) failed_jobs
      FROM dbo.campaigns c LEFT JOIN dbo.campaign_freezes f ON f.campaign_id=c.campaign_id LEFT JOIN dbo.generation_jobs j ON j.campaign_id=c.campaign_id WHERE (@id IS NULL OR c.campaign_id=@id)
      GROUP BY c.campaign_id,c.campaign_name,c.tier,c.campaign_hash,c.status,c.created_at ORDER BY c.campaign_id DESC;`);
    return result.recordset;
  }
  async indexStatus(): Promise<Record<string, unknown>> {
    const indexes=await this.pool.request().query<Record<string,unknown>>(`SELECT OBJECT_NAME(i.object_id) table_name,i.name index_name,i.type_desc,
      JSON_VALUE(v.build_parameters,'$.Version') index_version,v.vector_index_type,v.distance_metric,v.build_parameters
      FROM sys.vector_indexes v JOIN sys.indexes i ON i.object_id=v.object_id AND i.index_id=v.index_id ORDER BY table_name,index_name;`);
    const counts=await this.pool.request().query<Record<string,unknown>>(`SELECT 'search_semantic_train' table_name,COUNT_BIG(*) rows FROM dbo.search_semantic_train UNION ALL
      SELECT 'search_segment_train',COUNT_BIG(*) FROM dbo.search_segment_train UNION ALL SELECT 'search_style_train',COUNT_BIG(*) FROM dbo.search_style_train UNION ALL
      SELECT 'search_fingerprint_train',COUNT_BIG(*) FROM dbo.search_fingerprint_train UNION ALL SELECT 'search_residual_train',COUNT_BIG(*) FROM dbo.search_residual_train;`);
    return { indexes:indexes.recordset,corpora:counts.recordset,defaultMode:"exact",reason:"ANN is enabled only after retained benchmark and plan evidence pass" };
  }
  async cluster(id: number): Promise<Record<string, unknown> | null> {
    const run=await this.pool.request().input("id",sql.BigInt,id).query<Record<string,unknown>>(`SELECT cluster_run_id,run_id,representation_id,algorithm,labels_hidden,config_json,created_at FROM dbo.cluster_runs WHERE cluster_run_id=@id;`);
    if (!run.recordset[0]) return null;
    const assignments=await this.pool.request().input("id",sql.BigInt,id).query<Record<string,unknown>>(`SELECT TOP(5000) a.generation_id,a.cluster_label,p.x,p.y,p.method
      FROM dbo.cluster_assignments a LEFT JOIN dbo.projection_coordinates p ON p.cluster_run_id=a.cluster_run_id AND p.generation_id=a.generation_id WHERE a.cluster_run_id=@id ORDER BY a.generation_id;`);
    return {...run.recordset[0],assignments:assignments.recordset};
  }
  async knownGenerationNeighbors(id: number,k=20): Promise<Record<string, unknown> | null> {
    const result=await this.pool.request().input("id",sql.BigInt,id).input("k",sql.Int,Math.max(1,Math.min(k,50))).query<Record<string,unknown>>(`WITH query_row AS
      (SELECT TOP(1) g.generation_id,v.prompt_group_id,s.embedding FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
       JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
       JOIN dbo.semantic_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.embedding_profile_id='qwen3-embedding-0.6b' AND s.representation_id='whole-raw-final-v1' WHERE g.generation_id=@id), candidates AS
      (SELECT TOP(500) c.vector_id,c.model_profile_id,c.prompt_group_id,c.text_artifact_id,VECTOR_DISTANCE('cosine',c.embedding,q.embedding) distance FROM dbo.search_semantic_train c CROSS JOIN query_row q
       WHERE c.embedding_profile_id='qwen3-embedding-0.6b' AND c.text_view_id='raw-final-v1' AND c.prompt_group_id<>q.prompt_group_id ORDER BY distance,c.vector_id), ranked AS
      (SELECT *,ROW_NUMBER() OVER(PARTITION BY prompt_group_id ORDER BY distance,vector_id) rp,ROW_NUMBER() OVER(PARTITION BY text_artifact_id ORDER BY distance,vector_id) rt FROM candidates)
      SELECT TOP(@k) vector_id,model_profile_id,prompt_group_id,text_artifact_id,distance FROM ranked WHERE rp=1 AND rt=1 ORDER BY distance,vector_id;`);
    return result.recordset.length ? {generationId:id,actualSearchMode:"exact",neighbors:result.recordset} : null;
  }
  async fieldGuide(): Promise<Record<string, unknown>[]> {
    const result=await this.pool.request().query<Record<string,unknown>>(`WITH ranked AS (SELECT s.model_profile_id,d.phrase,s.log_odds,s.z_score,
      ROW_NUMBER() OVER(PARTITION BY s.model_profile_id ORDER BY ABS(s.z_score) DESC,d.phrase) rank FROM dbo.model_phrase_stats s JOIN dbo.phrase_dictionary d ON d.phrase_id=s.phrase_id)
      SELECT model_profile_id,phrase,log_odds,z_score,rank FROM ranked WHERE rank<=15 ORDER BY model_profile_id,rank;`);
    return result.recordset;
  }
  async close(): Promise<void> { await this.pool.close(); }
}
