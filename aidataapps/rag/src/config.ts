import dotenv from "dotenv";
import { z } from "zod";
import { resolveModelProfile, type ModelProfile } from "./models.js";

dotenv.config({ quiet: true });

const environmentSchema = z.object({
  HOST: z.string().default("127.0.0.1"),
  PORT: z.coerce.number().int().min(1).max(65535).default(3000),
  LOG_LEVEL: z.string().default("info"),
  SQLSERVER_HOST: z.string().default("127.0.0.1"),
  SQLSERVER_PORT: z.coerce.number().int().min(1).max(65535).default(1433),
  MSSQL_SA_PASSWORD: z.string().min(8),
  MSSQL_DATABASE: z.string().regex(/^[A-Za-z][A-Za-z0-9_]{0,63}$/).default("RagLab"),
  MODEL_PROFILE: z.string().default("qwen-smoke"),
  CHAT_BASE_URL: z.string().url().default("http://127.0.0.1:8000/v1"),
  EMBEDDING_BASE_URL: z.string().url().default("http://127.0.0.1:8001/v1"),
  EMBEDDING_MODEL: z.string().default("Qwen/Qwen3-Embedding-0.6B"),
  EMBEDDING_DIMENSIONS: z.coerce.number().int().min(1024).max(1024).default(1024),
  TOP_K: z.coerce.number().int().min(1).max(20).default(4),
});

export interface AppConfig {
  host: string;
  port: number;
  logLevel: string;
  database: {
    server: string;
    port: number;
    user: string;
    password: string;
    database: string;
  };
  inference: {
    chatBaseUrl: string;
    embeddingBaseUrl: string;
    embeddingModel: string;
    embeddingDimensions: number;
    model: ModelProfile;
  };
  topK: number;
}

export function loadConfig(
  values: NodeJS.ProcessEnv = process.env,
): AppConfig {
  const env = environmentSchema.parse(values);
  return {
    host: env.HOST,
    port: env.PORT,
    logLevel: env.LOG_LEVEL,
    database: {
      server: env.SQLSERVER_HOST,
      port: env.SQLSERVER_PORT,
      user: "sa",
      password: env.MSSQL_SA_PASSWORD,
      database: env.MSSQL_DATABASE,
    },
    inference: {
      chatBaseUrl: env.CHAT_BASE_URL.replace(/\/$/, ""),
      embeddingBaseUrl: env.EMBEDDING_BASE_URL.replace(/\/$/, ""),
      embeddingModel: env.EMBEDDING_MODEL,
      embeddingDimensions: env.EMBEDDING_DIMENSIONS,
      model: resolveModelProfile(env.MODEL_PROFILE),
    },
    topK: env.TOP_K,
  };
}
