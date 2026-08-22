import { z } from "zod";

const baseUrl = process.env.APP_BASE_URL ?? "http://127.0.0.1:3000";

async function query(body: { query: string; allowActions: boolean }): Promise<unknown> {
  const response = await fetch(`${baseUrl}/api/query`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(180_000),
  });
  if (!response.ok) {
    throw new Error(`Query failed with HTTP ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

const citationSchema = z.object({ chunkId: z.string() });
const answerSchema = z.object({
  kind: z.literal("answer"),
  answer: z.string().min(1),
  citations: z.array(citationSchema).min(1),
});
const actionSchema = z.object({
  kind: z.literal("action"),
  citations: z.array(citationSchema).min(1),
  action: z.object({
    name: z.literal("create_work_order"),
    status: z.literal("executed"),
    result: z.object({ id: z.number().int().positive(), assetTag: z.literal("NB-104") }),
  }),
});

const answer = answerSchema.parse(
  await query({
    query: "What should I inspect when a Comet S2 loses motor assistance after heavy rain?",
    allowActions: false,
  }),
);
const action = actionSchema.parse(
  await query({
    query:
      "Create a high-priority work order for NB-104 to inspect its intermittent motor assistance loss after rain. Do not claim the cause is confirmed.",
    allowActions: true,
  }),
);

console.log(
  JSON.stringify(
    {
      status: "passed",
      answerCitations: answer.citations.map(({ chunkId }) => chunkId),
      workOrderId: action.action.result.id,
    },
    null,
    2,
  ),
);
