import dotenv from "dotenv";
import { z } from "zod";

dotenv.config({ quiet: true });

const databaseName = z.string().regex(/^[A-Za-z][A-Za-z0-9_]{0,63}$/);

const schema = z.object({
  SQLSERVER_HOST: z.string().default("127.0.0.1"),
  SQLSERVER_PORT: z.coerce.number().int().min(1).max(65535).default(1435),
  MSSQL_SA_PASSWORD: z.string().min(12),
  GT_LAB_PASSWORD: z.string().min(12).optional(),
  GT_SERVICE_PASSWORD: z.string().min(12).optional(),
  CONTROL_DATABASE: databaseName.default("GhostTypeControl"),
  WORKLOAD_DATABASE: databaseName.default("GhostTypeWorkloads"),
  GATEWAY_PORT: z.coerce.number().int().default(8030),
  SCRIPTDOM_PORT: z.coerce.number().int().default(8031),
  CHAT_BASE_URL: z.string().url().default("http://127.0.0.1:8020/v1"),
  VLM_BASE_URL: z.string().url().default("http://127.0.0.1:8021/v1"),
  EMBEDDING_BASE_URL: z.string().url().default("http://127.0.0.1:8010/v1"),
  RUNS_MIRROR: z.string().default("/Users/karl/aidataapps/lab04/runs"),
});

export function loadConfig(values: NodeJS.ProcessEnv = process.env) {
  const env = schema.parse(values);
  const base = {
    server: env.SQLSERVER_HOST,
    port: env.SQLSERVER_PORT,
    options: { encrypt: false, trustServerCertificate: true },
    requestTimeout: 600_000,
  } as const;
  return {
    databases: {
      controlName: env.CONTROL_DATABASE,
      workloadName: env.WORKLOAD_DATABASE,
      admin: { ...base, user: "sa", password: env.MSSQL_SA_PASSWORD },
      lab: env.GT_LAB_PASSWORD ? { ...base, user: "gt_lab", password: env.GT_LAB_PASSWORD } : undefined,
      service: env.GT_SERVICE_PASSWORD ? { ...base, user: "gt_service", password: env.GT_SERVICE_PASSWORD } : undefined,
    },
    inference: {
      chatBaseUrl: env.CHAT_BASE_URL.replace(/\/$/, ""),
      vlmBaseUrl: env.VLM_BASE_URL.replace(/\/$/, ""),
      embeddingBaseUrl: env.EMBEDDING_BASE_URL.replace(/\/$/, ""),
    },
    ports: { gateway: env.GATEWAY_PORT, scriptdom: env.SCRIPTDOM_PORT },
    runsMirror: env.RUNS_MIRROR,
  };
}
