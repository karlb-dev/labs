import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import sql from "mssql";
import { z } from "zod";
import { loadConfig } from "../src/config.js";
import { VllmGateway } from "../src/inference.js";

const assetsSchema = z.array(
  z.object({
    assetTag: z.string(),
    model: z.string(),
    depot: z.string(),
    status: z.enum(["active", "inspection_due", "out_of_service"]),
    odometerKm: z.number().int(),
    lastServiceAt: z.string().datetime(),
    notes: z.string().nullable(),
  }),
);

const knowledgeSchema = z.array(
  z.object({
    documentId: z.string(),
    title: z.string(),
    sourceUri: z.string(),
    revision: z.string(),
    effectiveAt: z.string(),
    chunks: z.array(
      z.object({
        chunkId: z.string(),
        heading: z.string(),
        content: z.string(),
      }),
    ),
  }),
);

const here = new URL(".", import.meta.url);
const config = loadConfig();
const assets = assetsSchema.parse(
  JSON.parse(
    await readFile(new URL("../data/assets.json", here), "utf8"),
  ) as unknown,
);
const documents = knowledgeSchema.parse(
  JSON.parse(
    await readFile(new URL("../data/knowledge.json", here), "utf8"),
  ) as unknown,
);
const schemaSql = await readFile(new URL("../db/schema.sql", here), "utf8");

const connectionOptions = {
  server: config.database.server,
  port: config.database.port,
  user: config.database.user,
  password: config.database.password,
  options: { encrypt: false, trustServerCertificate: true },
};

const master = await new sql.ConnectionPool({
  ...connectionOptions,
  database: "master",
}).connect();
try {
  const database = config.database.database;
  await master.request().query(`
    IF DB_ID(N'${database}') IS NULL
      EXEC(N'CREATE DATABASE [${database}]');
  `);
} finally {
  await master.close();
}

const pool = await new sql.ConnectionPool({
  ...connectionOptions,
  database: config.database.database,
}).connect();

try {
  for (const batch of schemaSql.split(/^\s*GO\s*$/gim)) {
    if (batch.trim()) await pool.request().batch(batch);
  }

  await pool.request().batch(`
    DELETE FROM dbo.work_orders;
    DELETE FROM dbo.knowledge_chunks;
    DELETE FROM dbo.knowledge_documents;
    DELETE FROM dbo.assets;
  `);

  for (const asset of assets) {
    await pool
      .request()
      .input("assetTag", sql.VarChar(32), asset.assetTag)
      .input("model", sql.NVarChar(80), asset.model)
      .input("depot", sql.NVarChar(80), asset.depot)
      .input("status", sql.VarChar(24), asset.status)
      .input("odometerKm", sql.Int, asset.odometerKm)
      .input("lastServiceAt", sql.DateTime2, new Date(asset.lastServiceAt))
      .input("notes", sql.NVarChar(500), asset.notes)
      .query(`
        INSERT dbo.assets
          (asset_tag, model, depot, status, odometer_km, last_service_at, notes)
        VALUES
          (@assetTag, @model, @depot, @status, @odometerKm, @lastServiceAt, @notes);
      `);
  }

  for (const document of documents) {
    await pool
      .request()
      .input("documentId", sql.VarChar(80), document.documentId)
      .input("title", sql.NVarChar(200), document.title)
      .input("sourceUri", sql.NVarChar(300), document.sourceUri)
      .input("revision", sql.VarChar(32), document.revision)
      .input("effectiveAt", sql.Date, new Date(`${document.effectiveAt}T00:00:00Z`))
      .query(`
        INSERT dbo.knowledge_documents
          (document_id, title, source_uri, revision, effective_at)
        VALUES
          (@documentId, @title, @sourceUri, @revision, @effectiveAt);
      `);
  }

  const chunks = documents.flatMap((document) =>
    document.chunks.map((chunk) => ({ ...chunk, document })),
  );
  const gateway = new VllmGateway(config.inference);
  const embeddings = await gateway.embed(
    chunks.map(
      ({ document, heading, content }) =>
        `${document.title}\n${heading}\n${content}`,
    ),
  );

  for (const [index, chunk] of chunks.entries()) {
    const embedding = embeddings[index];
    if (!embedding) throw new Error(`Missing embedding for ${chunk.chunkId}`);
    const digest = createHash("sha256").update(chunk.content).digest("hex");
    await pool
      .request()
      .input("chunkId", sql.VarChar(100), chunk.chunkId)
      .input("documentId", sql.VarChar(80), chunk.document.documentId)
      .input("documentTitle", sql.NVarChar(200), chunk.document.title)
      .input("heading", sql.NVarChar(200), chunk.heading)
      .input("content", sql.NVarChar(sql.MAX), chunk.content)
      .input("contentSha256", sql.Char(64), digest)
      .input("embedding", sql.NVarChar(sql.MAX), JSON.stringify(embedding))
      .query(`
        INSERT dbo.knowledge_chunks
          (chunk_id, document_id, document_title, heading, content, content_sha256, embedding)
        VALUES
          (@chunkId, @documentId, @documentTitle, @heading, @content, @contentSha256,
           CAST(@embedding AS VECTOR(1024)));
      `);
  }

  const verification = await pool.request().query<{
    assets: number;
    documents: number;
    chunks: number;
    productVersion: string;
  }>(`
    SELECT
      (SELECT COUNT(*) FROM dbo.assets) AS assets,
      (SELECT COUNT(*) FROM dbo.knowledge_documents) AS documents,
      (SELECT COUNT(*) FROM dbo.knowledge_chunks) AS chunks,
      CONVERT(VARCHAR(64), SERVERPROPERTY('ProductVersion')) AS productVersion;
  `);
  console.log(JSON.stringify(verification.recordset[0], null, 2));
} finally {
  await pool.close();
}
