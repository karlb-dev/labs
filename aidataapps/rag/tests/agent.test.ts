import { describe, expect, it, vi } from "vitest";
import { RagAgent } from "../src/agent.js";
import type {
  CreateWorkOrderInput,
  InferenceGateway,
  KnowledgeHit,
  KnowledgeRepository,
} from "../src/types.js";

const hits: KnowledgeHit[] = [
  {
    chunkId: "comet-motor-rain-cutout",
    documentId: "manual",
    documentTitle: "Comet S2 Service Manual",
    heading: "Motor assistance cuts out after rain",
    content: "Inspect and dry connector J4.",
    distance: 0.08,
  },
  {
    chunkId: "triage-priority-levels",
    documentId: "policy",
    documentTitle: "Fleet Service Triage Policy",
    heading: "Work-order priority levels",
    content: "Intermittent drive loss is high priority.",
    distance: 0.12,
  },
];

function harness(modelOutput: object) {
  const repository: KnowledgeRepository = {
    search: vi.fn().mockResolvedValue(hits),
    createWorkOrder: vi.fn().mockImplementation(async (input: CreateWorkOrderInput) => ({
      id: 42,
      ...input,
      status: "open" as const,
      createdAt: "2026-08-22T00:00:00.000Z",
    })),
    ready: vi.fn().mockResolvedValue(undefined),
    close: vi.fn().mockResolvedValue(undefined),
  };
  const inference: InferenceGateway = {
    embed: vi.fn().mockResolvedValue([[0.1, 0.2]]),
    complete: vi.fn().mockResolvedValue(JSON.stringify(modelOutput)),
    ready: vi.fn().mockResolvedValue(undefined),
  };
  return {
    repository,
    inference,
    agent: new RagAgent(repository, inference, {
      topK: 4,
      modelProfile: "test-model",
    }),
  };
}

describe("RagAgent", () => {
  it("returns grounded text with only valid citations", async () => {
    const { agent } = harness({
      kind: "answer",
      answer: "Inspect connector J4 and keep the wet harness de-energized.",
      citations: ["comet-motor-rain-cutout", "made-up-chunk"],
    });

    const result = await agent.query({ query: "What should I inspect?", allowActions: false });

    expect(result.kind).toBe("answer");
    expect(result.citations.map((citation) => citation.chunkId)).toEqual([
      "comet-motor-rain-cutout",
    ]);
  });

  it("proposes an action without mutating data when actions are disabled", async () => {
    const { agent, repository } = harness({
      kind: "action",
      message: "I can open the requested inspection.",
      citations: ["triage-priority-levels"],
      action: {
        name: "create_work_order",
        arguments: {
          assetTag: "NB-104",
          title: "Inspect intermittent drive loss",
          description: "Rider observed motor assistance loss after rain.",
          priority: "high",
        },
      },
    });

    const result = await agent.query({ query: "Create a work order", allowActions: false });

    expect(result.kind).toBe("action");
    if (result.kind !== "action") throw new Error("expected action");
    expect(result.action.status).toBe("proposed");
    expect(repository.createWorkOrder).not.toHaveBeenCalled();
  });

  it("executes an allowlisted action when explicitly enabled", async () => {
    const { agent, repository } = harness({
      kind: "action",
      message: "Opening the requested inspection.",
      citations: ["triage-priority-levels"],
      action: {
        name: "create_work_order",
        arguments: {
          assetTag: "NB-104",
          title: "Inspect intermittent drive loss",
          description: "Rider observed motor assistance loss after rain.",
          priority: "high",
        },
      },
    });

    const result = await agent.query({ query: "Create a work order", allowActions: true });

    expect(result.kind).toBe("action");
    if (result.kind !== "action") throw new Error("expected action");
    expect(result.action.status).toBe("executed");
    expect(result.action.result?.id).toBe(42);
    expect(repository.createWorkOrder).toHaveBeenCalledOnce();
  });

  it("rejects a response with no grounded citation", async () => {
    const { agent } = harness({
      kind: "answer",
      answer: "Unsupported claim",
      citations: ["not-retrieved"],
    });

    await expect(
      agent.query({ query: "Tell me something", allowActions: false }),
    ).rejects.toThrow("no valid citations");
  });
});
