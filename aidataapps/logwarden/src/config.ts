import dotenv from "dotenv";
import { z } from "zod";

dotenv.config({ quiet: true });

const databaseName = z.string().regex(/^[A-Za-z][A-Za-z0-9_]{0,63}$/);

const schema = z.object({
  HOST: z.string().default("127.0.0.1"),
  PORT: z.coerce.number().int().min(1).max(65535).default(3003),
  LOG_LEVEL: z.string().default("info"),
  SQLSERVER_HOST: z.string().default("127.0.0.1"),
  SQLSERVER_PORT: z.coerce.number().int().min(1).max(65535).default(1434),
  MSSQL_SA_PASSWORD: z.string().min(12),
  LW_LAB_PASSWORD: z.string().min(12),
  LW_AGENT_PASSWORD: z.string().min(12),
  CONTROL_DATABASE: databaseName.default("LogWardenControl"),
  WORKLOAD_DATABASE: databaseName.default("LogWardenWorkload"),
  CHAT_BASE_URL: z.string().url().default("http://127.0.0.1:8010/v1"),
  EMBEDDING_BASE_URL: z.string().url().default("http://127.0.0.1:8011/v1"),
  SECOND_EMBEDDING_BASE_URL: z.string().url().default("http://127.0.0.1:8012/v1"),
  RUNS_MIRROR: z.string().default("/content/drive/MyDrive/aidataapps/lab03/runs"),
  LOGWARDEN_TELEMETRY_SAMPLE_SECONDS: z.coerce.number().int().min(1).max(300).default(5),
});

export type AppConfig = ReturnType<typeof loadConfig>;

export function loadConfig(values: NodeJS.ProcessEnv = process.env) {
  const env = schema.parse(values);
  const baseDatabase = {
    server: env.SQLSERVER_HOST,
    port: env.SQLSERVER_PORT,
    options: { encrypt: false, trustServerCertificate: true, appName: "LogWarden" },
  } as const;
  return {
    host: env.HOST,
    port: env.PORT,
    logLevel: env.LOG_LEVEL,
    databases: {
      controlName: env.CONTROL_DATABASE,
      workloadName: env.WORKLOAD_DATABASE,
      admin: { ...baseDatabase, user: "sa", password: env.MSSQL_SA_PASSWORD },
      lab: { ...baseDatabase, user: "lw_lab", password: env.LW_LAB_PASSWORD },
      agent: { ...baseDatabase, user: "lw_agent", password: env.LW_AGENT_PASSWORD },
    },
    inference: {
      chatBaseUrl: env.CHAT_BASE_URL.replace(/\/$/, ""),
      qwenEmbeddingBaseUrl: env.EMBEDDING_BASE_URL.replace(/\/$/, ""),
      bgeEmbeddingBaseUrl: env.SECOND_EMBEDDING_BASE_URL.replace(/\/$/, ""),
    },
    runsMirror: env.RUNS_MIRROR,
    telemetrySampleSeconds: env.LOGWARDEN_TELEMETRY_SAMPLE_SECONDS,
  };
}
