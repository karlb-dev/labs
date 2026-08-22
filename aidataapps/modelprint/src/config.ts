import dotenv from "dotenv";
import { z } from "zod";

dotenv.config({ quiet: true });

const schema = z.object({
  HOST: z.string().default("127.0.0.1"),
  PORT: z.coerce.number().int().min(1).max(65535).default(3001),
  LOG_LEVEL: z.string().default("info"),
  SQLSERVER_HOST: z.string().default("127.0.0.1"),
  SQLSERVER_PORT: z.coerce.number().int().min(1).max(65535).default(1433),
  MSSQL_SA_PASSWORD: z.string().min(8),
  MSSQL_DATABASE: z.string().regex(/^[A-Za-z][A-Za-z0-9_]{0,63}$/).default("ModelPrint"),
  CHAT_BASE_URL: z.string().url().default("http://127.0.0.1:8000/v1"),
  EMBEDDING_BASE_URL: z.string().url().default("http://127.0.0.1:8001/v1"),
  SECOND_EMBEDDING_BASE_URL: z.string().url().default("http://127.0.0.1:8002/v1"),
});

export type AppConfig = ReturnType<typeof loadConfig>;

export function loadConfig(values: NodeJS.ProcessEnv = process.env) {
  const env = schema.parse(values);
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
      qwenEmbeddingBaseUrl: env.EMBEDDING_BASE_URL.replace(/\/$/, ""),
      bgeEmbeddingBaseUrl: env.SECOND_EMBEDDING_BASE_URL.replace(/\/$/, ""),
    },
  };
}
