import { z } from "zod";
import {
  prioritySchema,
  type Citation,
  type InferenceGateway,
  type KnowledgeHit,
  type KnowledgeRepository,
  type QueryAgent,
  type QueryRequest,
  type QueryResponse,
} from "./types.js";

const answerPlanSchema = z.object({
  kind: z.literal("answer"),
  answer: z.string().min(1),
  citations: z.array(z.string()).min(1),
});

const actionPlanSchema = z.object({
  kind: z.literal("action"),
  message: z.string().min(1),
  citations: z.array(z.string()).min(1),
  action: z.object({
    name: z.literal("create_work_order"),
    arguments: z.object({
      assetTag: z.string().regex(/^[A-Z]{2,8}-\d{2,8}$/),
      title: z.string().min(3).max(160),
      description: z.string().min(3).max(4000),
      priority: prioritySchema,
    }),
  }),
});

const modelPlanSchema = z.discriminatedUnion("kind", [
  answerPlanSchema,
  actionPlanSchema,
]);
export type ModelPlan = z.infer<typeof modelPlanSchema>;

const systemPrompt = `You are the grounded service assistant for Northstar Bikes, a fictional shared e-bike fleet.
Use only the supplied knowledge chunks for factual maintenance claims. Never invent an asset, policy, or procedure.

Choose exactly one response shape:
1. For a question, return {"kind":"answer","answer":"...","citations":["chunk-id"]}.
2. Only when the user explicitly asks to create, open, or schedule a work order, return {"kind":"action","message":"...","citations":["chunk-id"],"action":{"name":"create_work_order","arguments":{"assetTag":"NB-000","title":"...","description":"...","priority":"low|normal|high|urgent"}}}.

Citation values must be chunk IDs from the supplied context. Return JSON only.`;

export function parseModelPlan(raw: string): ModelPlan {
  const unfenced = raw
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "");
  const start = unfenced.indexOf("{");
  const end = unfenced.lastIndexOf("}");
  if (start < 0 || end <= start) {
    throw new Error("Model response did not contain a JSON object");
  }
  return modelPlanSchema.parse(JSON.parse(unfenced.slice(start, end + 1)) as unknown);
}

export class RagAgent implements QueryAgent {
  constructor(
    private readonly repository: KnowledgeRepository,
    private readonly inference: InferenceGateway,
    private readonly options: { topK: number; modelProfile: string },
  ) {}

  async query(request: QueryRequest): Promise<QueryResponse> {
    const [embedding] = await this.inference.embed([request.query]);
    if (!embedding) throw new Error("Embedding service returned no query vector");
    const hits = await this.repository.search(embedding, this.options.topK);
    if (hits.length === 0) throw new Error("No knowledge chunks are available");

    const prompt = this.buildPrompt(request.query, hits);
    const plan = parseModelPlan(await this.inference.complete(systemPrompt, prompt));
    const citations = this.resolveCitations(plan.citations, hits);
    if (citations.length === 0) {
      throw new Error("Model response contained no valid citations");
    }

    if (plan.kind === "answer") {
      return {
        kind: "answer",
        answer: plan.answer,
        citations,
        modelProfile: this.options.modelProfile,
      };
    }

    if (!request.allowActions) {
      return {
        kind: "action",
        message: plan.message,
        citations,
        modelProfile: this.options.modelProfile,
        action: {
          name: plan.action.name,
          arguments: plan.action.arguments,
          status: "proposed",
        },
      };
    }

    const result = await this.repository.createWorkOrder(plan.action.arguments);
    return {
      kind: "action",
      message: plan.message,
      citations,
      modelProfile: this.options.modelProfile,
      action: {
        name: plan.action.name,
        arguments: plan.action.arguments,
        status: "executed",
        result,
      },
    };
  }

  async ready(): Promise<void> {
    await Promise.all([this.repository.ready(), this.inference.ready()]);
  }

  private buildPrompt(query: string, hits: KnowledgeHit[]): string {
    const context = hits
      .map(
        (hit) =>
          `<chunk id="${hit.chunkId}" document="${hit.documentTitle}" heading="${hit.heading}">\n${hit.content}\n</chunk>`,
      )
      .join("\n\n");
    return `Knowledge context:\n${context}\n\nUser request:\n${query}`;
  }

  private resolveCitations(ids: string[], hits: KnowledgeHit[]): Citation[] {
    const byId = new Map(hits.map((hit) => [hit.chunkId, hit]));
    return [...new Set(ids)].flatMap((id) => {
      const hit = byId.get(id);
      return hit
        ? [
            {
              chunkId: hit.chunkId,
              documentTitle: hit.documentTitle,
              heading: hit.heading,
              distance: hit.distance,
            },
          ]
        : [];
    });
  }
}
