import sql from "mssql";
import type { AppConfig } from "./config.js";
import type {
  CreateWorkOrderInput,
  CreatedWorkOrder,
  KnowledgeHit,
  KnowledgeRepository,
} from "./types.js";

export class SqlServerRepository implements KnowledgeRepository {
  private constructor(private readonly pool: sql.ConnectionPool) {}

  static async connect(
    config: AppConfig["database"],
  ): Promise<SqlServerRepository> {
    const pool = await new sql.ConnectionPool({
      server: config.server,
      port: config.port,
      user: config.user,
      password: config.password,
      database: config.database,
      options: {
        encrypt: false,
        trustServerCertificate: true,
      },
      pool: { min: 0, max: 10, idleTimeoutMillis: 30_000 },
    }).connect();
    return new SqlServerRepository(pool);
  }

  async search(embedding: number[], topK: number): Promise<KnowledgeHit[]> {
    const result = await this.pool
      .request()
      .input("embedding", sql.NVarChar(sql.MAX), JSON.stringify(embedding))
      .input("topK", sql.Int, topK)
      .query<KnowledgeHit>(`
        SELECT TOP (@topK)
          chunk_id AS chunkId,
          document_id AS documentId,
          document_title AS documentTitle,
          heading,
          content,
          VECTOR_DISTANCE('cosine', embedding, CAST(@embedding AS VECTOR(1024))) AS distance
        FROM dbo.knowledge_chunks
        ORDER BY distance ASC, chunk_id ASC;
      `);
    return result.recordset.map((row) => ({ ...row, distance: Number(row.distance) }));
  }

  async createWorkOrder(
    input: CreateWorkOrderInput,
  ): Promise<CreatedWorkOrder> {
    const transaction = new sql.Transaction(this.pool);
    await transaction.begin();
    try {
      const asset = await new sql.Request(transaction)
        .input("assetTag", sql.VarChar(32), input.assetTag)
        .query<{ assetTag: string }>(`
          SELECT asset_tag AS assetTag
          FROM dbo.assets WITH (UPDLOCK, HOLDLOCK)
          WHERE asset_tag = @assetTag;
        `);
      if (asset.recordset.length === 0) {
        throw new Error(`Unknown asset tag: ${input.assetTag}`);
      }

      const inserted = await new sql.Request(transaction)
        .input("assetTag", sql.VarChar(32), input.assetTag)
        .input("title", sql.NVarChar(160), input.title)
        .input("description", sql.NVarChar(sql.MAX), input.description)
        .input("priority", sql.VarChar(16), input.priority)
        .query<{ id: number; createdAt: Date }>(`
          INSERT dbo.work_orders (asset_tag, title, description, priority)
          OUTPUT INSERTED.work_order_id AS id, INSERTED.created_at AS createdAt
          VALUES (@assetTag, @title, @description, @priority);
        `);
      await transaction.commit();
      const row = inserted.recordset[0];
      if (!row) throw new Error("SQL Server did not return the created work order");
      return {
        id: row.id,
        assetTag: input.assetTag,
        title: input.title,
        description: input.description,
        priority: input.priority,
        status: "open",
        createdAt: row.createdAt.toISOString(),
      };
    } catch (error) {
      try {
        await transaction.rollback();
      } catch {
        // SQL Server can abort a transaction before the driver sees the error.
      }
      throw error;
    }
  }

  async ready(): Promise<void> {
    await this.pool.request().query("SELECT 1 AS ready;");
  }

  async close(): Promise<void> {
    await this.pool.close();
  }
}
