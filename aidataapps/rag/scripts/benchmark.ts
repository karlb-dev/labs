import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { z } from "zod";
import { RagAgent } from "../src/agent.js";
import { buildApp } from "../src/app.js";
import { loadConfig } from "../src/config.js";
import { VllmGateway } from "../src/inference.js";
import { SqlServerRepository } from "../src/repository.js";

const casesSchema = z.array(
  z.object({
    id: z.string(),
    query: z.string(),
    expectedKind: z.enum(["answer", "action"]),
    requiredCitation: z.string(),
  }),
);

function valueAfter(flag: string): string | undefined {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function diagnostic(command: string, args: string[]): string | null {
  try {
    return execFileSync(command, args, { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

function percentile(values: number[], fraction: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))] ?? null;
}

const profile = valueAfter("--profile") ?? process.env.MODEL_PROFILE ?? "qwen-smoke";
const repeats = Number(valueAfter("--repeats") ?? "1");
if (!Number.isInteger(repeats) || repeats < 1 || repeats > 20) {
  throw new Error("--repeats must be an integer from 1 to 20");
}
process.env.MODEL_PROFILE = profile;

const config = loadConfig();
const cases = casesSchema.parse(
  JSON.parse(
    await readFile(new URL("../data/benchmark-cases.json", import.meta.url), "utf8"),
  ) as unknown,
);
const repository = await SqlServerRepository.connect(config.database);
const gateway = new VllmGateway(config.inference);
const agent = new RagAgent(repository, gateway, {
  topK: config.topK,
  modelProfile: config.inference.model.key,
});
const app = buildApp({ agent });

const startedAt = new Date();
const stamp = startedAt.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
const runDirectory = new URL(`../runs/benchmark-${profile}-${stamp}/`, import.meta.url);
await mkdir(runDirectory, { recursive: true });

const rows: Array<{
  caseId: string;
  repeat: number;
  latencyMs: number;
  statusCode: number;
  responseKind: string | null;
  citations: string[];
  response: Record<string, unknown>;
  passed: boolean;
  error: string | null;
}> = [];

try {
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    for (const testCase of cases) {
      const before = performance.now();
      const response = await app.inject({
        method: "POST",
        url: "/api/query",
        payload: { query: testCase.query, allowActions: false },
      });
      const latencyMs = performance.now() - before;
      let body: Record<string, unknown> = {};
      try {
        body = response.json<Record<string, unknown>>();
      } catch {
        // The row below records the invalid response as a failure.
      }
      const responseKind = typeof body.kind === "string" ? body.kind : null;
      const citations = Array.isArray(body.citations)
        ? body.citations.flatMap((citation) => {
            if (
              citation &&
              typeof citation === "object" &&
              "chunkId" in citation &&
              typeof citation.chunkId === "string"
            ) {
              return [citation.chunkId];
            }
            return [];
          })
        : [];
      const passed =
        response.statusCode === 200 &&
        responseKind === testCase.expectedKind &&
        citations.includes(testCase.requiredCitation);
      rows.push({
        caseId: testCase.id,
        repeat,
        latencyMs: Number(latencyMs.toFixed(2)),
        statusCode: response.statusCode,
        responseKind,
        citations,
        response: body,
        passed,
        error: passed ? null : JSON.stringify(body).slice(0, 2000),
      });
    }
  }
} finally {
  await app.close();
  await repository.close();
}

const passed = rows.filter((row) => row.passed).length;
const latencies = rows.filter((row) => row.passed).map((row) => row.latencyMs);
const run = {
  schemaVersion: 1,
  startedAt: startedAt.toISOString(),
  finishedAt: new Date().toISOString(),
  model: config.inference.model,
  embeddingModel: config.inference.embeddingModel,
  embeddingDimensions: config.inference.embeddingDimensions,
  topK: config.topK,
  repeats,
  gitCommit: diagnostic("git", ["rev-parse", "HEAD"]),
  nodeVersion: process.version,
  gpu: diagnostic("nvidia-smi", ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
  chatContainer: diagnostic("docker", [
    "inspect",
    "--format",
    "{{.Image}}",
    `${process.env.COMPOSE_PROJECT_NAME ?? "aidataapps-rag"}-chat`,
  ]),
};
const summary = {
  profile,
  passed,
  total: rows.length,
  passRate: rows.length ? passed / rows.length : 0,
  latencyMs: {
    p50: percentile(latencies, 0.5),
    p95: percentile(latencies, 0.95),
    min: latencies.length ? Math.min(...latencies) : null,
    max: latencies.length ? Math.max(...latencies) : null,
  },
};
const markdown = `# RAG benchmark: ${profile}\n\n- Passed: ${passed}/${rows.length}\n- p50 latency: ${summary.latencyMs.p50 ?? "n/a"} ms\n- p95 latency: ${summary.latencyMs.p95 ?? "n/a"} ms\n- Model: \`${config.inference.model.modelId}@${config.inference.model.revision}\`\n- Embeddings: \`${config.inference.embeddingModel}\`\n\nThis is a plumbing and grounded-response benchmark, not a general model-quality leaderboard.\n`;

await Promise.all([
  writeFile(new URL("run.json", runDirectory), `${JSON.stringify(run, null, 2)}\n`),
  writeFile(new URL("results.jsonl", runDirectory), `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`),
  writeFile(new URL("summary.json", runDirectory), `${JSON.stringify(summary, null, 2)}\n`),
  writeFile(new URL("summary.md", runDirectory), markdown),
]);
console.log(JSON.stringify({ runDirectory: runDirectory.pathname, ...summary }, null, 2));
if (passed !== rows.length) process.exitCode = 1;
