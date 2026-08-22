import { z } from "zod";

export const prioritySchema = z.enum(["low", "normal", "high", "urgent"]);
export type WorkOrderPriority = z.infer<typeof prioritySchema>;

export interface KnowledgeHit {
  chunkId: string;
  documentId: string;
  documentTitle: string;
  heading: string;
  content: string;
  distance: number;
}

export interface Citation {
  chunkId: string;
  documentTitle: string;
  heading: string;
  distance: number;
}

export interface CreateWorkOrderInput {
  assetTag: string;
  title: string;
  description: string;
  priority: WorkOrderPriority;
}

export interface CreatedWorkOrder extends CreateWorkOrderInput {
  id: number;
  status: "open";
  createdAt: string;
}

export const queryRequestSchema = z.object({
  query: z.string().trim().min(3).max(4000),
  allowActions: z.boolean().default(false),
});
export type QueryRequest = z.infer<typeof queryRequestSchema>;

export type QueryResponse =
  | {
      kind: "answer";
      answer: string;
      citations: Citation[];
      modelProfile: string;
    }
  | {
      kind: "action";
      message: string;
      citations: Citation[];
      modelProfile: string;
      action: {
        name: "create_work_order";
        arguments: CreateWorkOrderInput;
        status: "proposed" | "executed";
        result?: CreatedWorkOrder;
      };
    };

export interface KnowledgeRepository {
  search(embedding: number[], topK: number): Promise<KnowledgeHit[]>;
  createWorkOrder(input: CreateWorkOrderInput): Promise<CreatedWorkOrder>;
  ready(): Promise<void>;
  close(): Promise<void>;
}

export interface InferenceGateway {
  embed(input: string[]): Promise<number[][]>;
  complete(systemPrompt: string, userPrompt: string): Promise<string>;
  ready(): Promise<void>;
}

export interface QueryAgent {
  query(request: QueryRequest): Promise<QueryResponse>;
  ready(): Promise<void>;
}
