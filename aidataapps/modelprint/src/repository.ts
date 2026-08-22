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
  semantic1024: { table: "dbo.search_semantic_train", dimensions: 1024 },
  style512: { table: "dbo.search_style_train", dimensions: 512 },
  fingerprint64: { table: "dbo.search_fingerprint_train", dimensions: 64 },
  likelihood8: { table: "dbo.search_likelihood_train", dimensions: 8 },
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
    options: { k: number; candidateK?: number; excludePromptGroupId?: string } = { k: 10 },
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
          WHERE @excludePromptGroupId IS NULL OR prompt_group_id <> @excludePromptGroupId
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
    k: number,
    tau: number,
    excludePromptGroupId?: string,
  ): Promise<Array<{ modelProfileId: string; voteShare: number; neighborCount: number; independentGroups: number; nearestDistance: number }>> {
    const definition = searchSpaces[space];
    if (embedding.length !== definition.dimensions || !embedding.every(Number.isFinite)) {
      throw new Error(`${space} requires a finite ${definition.dimensions}-dimensional vector`);
    }
    const safeK = Math.max(1, Math.min(k, 100));
    const result = await this.pool.request()
      .input("embedding", sql.NVarChar(sql.MAX), JSON.stringify(embedding))
      .input("k", sql.Int, safeK)
      .input("candidateK", sql.Int, Math.min(safeK * 10, 2000))
      .input("tau", sql.Float, tau)
      .input("excludePromptGroupId", sql.VarChar(120), excludePromptGroupId ?? null)
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
        WHERE @excludePromptGroupId IS NULL OR prompt_group_id <> @excludePromptGroupId
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
  async close(): Promise<void> { await this.pool.close(); }
}
