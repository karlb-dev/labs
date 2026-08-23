import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import Fastify from "fastify";
import { z } from "zod";
import type { AppConfig } from "./config.js";
import { VllmGateway } from "./inference.js";
import { maskNames } from "./prompt-bank.js";
import { SqlServerRepository } from "./repository.js";
import { styleVector } from "./style.js";

const identifySchema = z.object({ text: z.string().min(1).max(200_000), prompt: z.string().max(100_000).optional(),
  textView: z.enum(["raw-final-v1", "name-masked-v1"]).default("name-masked-v1"), searchMode: z.literal("exact").default("exact"), k: z.number().int().min(1).max(50).default(20) });
const compareSchema = z.object({ left: z.string().min(1).max(200_000), right: z.string().min(1).max(200_000) });
const groupSchema = z.object({ texts: z.array(z.string().min(1).max(200_000)).min(2).max(50), threshold: z.number().min(0).max(2).default(0.25) });
const publicRoot = fileURLToPath(new URL("../public/", import.meta.url));

export async function buildApp(config: AppConfig) {
  const app = Fastify({ logger: { level: config.logLevel } });
  const repository = await SqlServerRepository.connect(config.database); const gateway = new VllmGateway();
  app.addHook("onClose", async () => repository.close());
  app.get("/health", async () => { await repository.ready(); return { status: "ok", database: config.database.database }; });
  app.get("/", async (_request, reply) => reply.type("text/html; charset=utf-8").send(await readFile(`${publicRoot}/index.html`, "utf8")));
  app.get("/app.js", async (_request, reply) => reply.type("text/javascript; charset=utf-8").send(await readFile(`${publicRoot}/app.js`, "utf8")));

  app.post("/api/identify", async (request, reply) => {
    const parsed = identifySchema.safeParse(request.body); if (!parsed.success) return reply.code(400).send({ error: parsed.error.flatten() });
    const input = parsed.data; const text = input.textView === "name-masked-v1" ? maskNames(input.text) : input.text;
    const words = text.trim().split(/\s+/).filter(Boolean).length;
    if (words < 16) return { decision: "insufficient_text", claimScope: "closed-set-served-profile", actualSearchMode: "exact", reason: "fewer-than-16-whitespace-tokens", evidence: [] };
    const semantic = (await gateway.embed(config.inference.qwenEmbeddingBaseUrl, "Qwen/Qwen3-Embedding-0.6B", 1024, [text]))[0]!;
    const [semanticVote, styleVote, semanticNeighbors, styleNeighbors] = await Promise.all([
      repository.exactVote("semantic1024", semantic, { k: input.k, tau: 0.10, textView: input.textView }),
      repository.exactVote("style512", styleVector(text), { k: input.k, tau: 0.10, textView: input.textView }),
      repository.exactNeighbors("semantic1024", semantic, { k: Math.min(input.k, 10), textView: input.textView }),
      repository.exactNeighbors("style512", styleVector(text), { k: Math.min(input.k, 10), textView: input.textView }),
    ]);
    const hasCalibration = await repository.latestCalibration();
    return { decision: hasCalibration ? "ambiguous" : "ambiguous", calibrated: false,
      reason: hasCalibration ? "calibrated-hybrid-loading-not-yet-enabled" : "no-completed-calibration-model",
      claimScope: "closed-set-served-profile", actualSearchMode: "exact", requestedSearchMode: input.searchMode, textView: input.textView,
      channels: { semanticVote, styleVote }, evidence: { semanticNeighbors, styleNeighbors }, caveat: "Neighbor vote shares are retrieval features, not probabilities." };
  });

  app.post("/api/compare", async (request, reply) => {
    const parsed = compareSchema.safeParse(request.body); if (!parsed.success) return reply.code(400).send({ error: parsed.error.flatten() });
    const embeddings = await gateway.embed(config.inference.qwenEmbeddingBaseUrl, "Qwen/Qwen3-Embedding-0.6B", 1024, [parsed.data.left, parsed.data.right]);
    const cosineDistance = (left: number[], right: number[]) => 1 - left.reduce((sum, value, index) => sum + value * right[index]!, 0) /
      Math.max(Math.sqrt(left.reduce((sum, value) => sum + value * value, 0) * right.reduce((sum, value) => sum + value * value, 0)), Number.EPSILON);
    return { decision: "ambiguous", calibrated: false, semanticDistance: cosineDistance(embeddings[0]!, embeddings[1]!),
      styleDistance: cosineDistance(styleVector(parsed.data.left), styleVector(parsed.data.right)), actualSearchMode: "exact",
      caveat: "Distance alone is not a calibrated same-source probability." };
  });

  app.post("/api/group", async (request, reply) => {
    const parsed = groupSchema.safeParse(request.body); if (!parsed.success) return reply.code(400).send({ error: parsed.error.flatten() });
    const vectors = await gateway.embed(config.inference.qwenEmbeddingBaseUrl, "Qwen/Qwen3-Embedding-0.6B", 1024, parsed.data.texts);
    const parent = vectors.map((_, index) => index); const find = (x: number): number => parent[x] === x ? x : (parent[x] = find(parent[x]!));
    const join = (a: number, b: number) => { const left = find(a); const right = find(b); if (left !== right) parent[right] = left; };
    const distances: Array<{ left: number; right: number; distance: number }> = [];
    for (let left = 0; left < vectors.length; left += 1) for (let right = left + 1; right < vectors.length; right += 1) {
      const distance = 1 - vectors[left]!.reduce((sum, value, index) => sum + value * vectors[right]![index]!, 0); distances.push({ left, right, distance });
      if (distance <= parsed.data.threshold) join(left, right);
    }
    const labelMap = new Map<number, number>(); const assignments = parent.map((_, index) => { const root = find(index); if (!labelMap.has(root)) labelMap.set(root, labelMap.size); return labelMap.get(root)!; });
    return { assignments, distances, calibrated: false, method: "semantic-threshold-explorer", actualSearchMode: "exact",
      caveat: "Explorer groups are descriptive until the pairwise calibration gate passes." };
  });

  app.get("/api/evaluations", async () => repository.evaluations());
  app.get("/api/evaluations/:id", async (request, reply) => {
    const id = Number.parseInt((request.params as { id: string }).id, 10); if (!Number.isInteger(id)) return reply.code(400).send({ error: "invalid id" });
    const row = await repository.evaluation(id); return row ?? reply.code(404).send({ error: "not found" });
  });
  app.get("/api/campaigns/:id", async (request, reply) => {
    const id=Number.parseInt((request.params as {id:string}).id,10);if (!Number.isInteger(id)) return reply.code(400).send({error:"invalid id"});
    const rows=await repository.campaigns(id);return rows[0] ?? reply.code(404).send({error:"not found"});
  });
  app.get("/api/index/status",async()=>repository.indexStatus());
  app.get("/api/clusters/:id",async(request,reply)=>{
    const id=Number.parseInt((request.params as {id:string}).id,10);if (!Number.isInteger(id)) return reply.code(400).send({error:"invalid id"});
    return await repository.cluster(id) ?? reply.code(404).send({error:"not found"});
  });
  app.get("/api/generations/:id/neighbors",async(request,reply)=>{
    const id=Number.parseInt((request.params as {id:string}).id,10);if (!Number.isInteger(id)) return reply.code(400).send({error:"invalid id"});
    return await repository.knownGenerationNeighbors(id) ?? reply.code(404).send({error:"generation or frozen vector not found"});
  });
  app.get("/api/field-guide",async()=>({scope:"frequency-evidence-not-signatures",phrases:await repository.fieldGuide()}));
  return app;
}
