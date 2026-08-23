import { z } from "zod";

export const incidentClasses = [
  "deadlock",
  "blocking",
  "transaction_log_full",
  "authentication_failure",
  "integrity_signal",
  "backup_restore_failure",
  "query_resource_pressure",
  "schema_data_error",
  "unknown_ambiguous",
  "benign_noise",
] as const;

export const severities = ["info", "low", "medium", "high", "critical"] as const;
export const actions = ["no_action", "run_tsg", "open_work_item", "escalate_to_human"] as const;

export const toolRequestSchema = z.object({
  kind: z.literal("tool_request"),
  tool: z.string().min(1).max(80),
  arguments: z.record(z.string(), z.unknown()),
}).strict();

export const decisionSchema = z.object({
  kind: z.literal("decision"),
  incidentClass: z.enum(incidentClasses),
  severity: z.enum(severities),
  action: z.enum(actions),
  actionArguments: z.record(z.string(), z.unknown()),
  citedChunkIds: z.array(z.string().min(1).max(140)).max(20),
  confidence: z.number().min(0).max(1),
  abstain: z.boolean(),
  correlationKey: z.string().min(1).max(160),
  summary: z.string().max(500),
  rationale: z.string().max(4000),
}).strict();

export const agentResponseSchema = z.discriminatedUnion("kind", [toolRequestSchema, decisionSchema]);
export type AgentResponse = z.infer<typeof agentResponseSchema>;
export type RepairKind = "none" | "fence_strip" | "leading_text_strip";
export type AgentResponseErrorClass = "empty_output" | "invalid_json" | "contract_schema";

export class AgentResponseParseError extends Error {
  constructor(readonly errorClass: AgentResponseErrorClass, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "AgentResponseParseError";
  }
}

export function parseAgentResponse(raw: string): { value: AgentResponse; repairKind: RepairKind; parsedText: string } {
  const trimmed = raw.trim();
  if (trimmed.length === 0) throw new AgentResponseParseError("empty_output", "Agent response is empty");
  let candidate = trimmed;
  let repairKind: RepairKind = "none";

  const fenced = trimmed.match(/^```(?:json)?\s*\n?([\s\S]*?)\n?```$/i);
  if (fenced?.[1] !== undefined) {
    candidate = fenced[1].trim();
    repairKind = "fence_strip";
  } else if (!trimmed.startsWith("{")) {
    const firstBrace = trimmed.indexOf("{");
    if (firstBrace < 0) throw new AgentResponseParseError("invalid_json", "Agent response does not contain a JSON object");
    candidate = trimmed.slice(firstBrace).trim();
    repairKind = "leading_text_strip";
  }

  let decoded: unknown;
  try {
    decoded = JSON.parse(candidate);
  } catch (error) {
    throw new AgentResponseParseError(
      "invalid_json",
      `Agent response is not valid JSON after ${repairKind}: ${(error as Error).message}`,
      { cause: error },
    );
  }
  const validated = agentResponseSchema.safeParse(decoded);
  if (!validated.success) {
    throw new AgentResponseParseError(
      "contract_schema",
      `Agent response violates the structured contract (${validated.error.issues.length} issue(s))`,
      { cause: validated.error },
    );
  }
  return { value: validated.data, repairKind, parsedText: candidate };
}

export const OPERATING_CONTRACT_VERSION = "logwarden-json-v1";

export function operatingContract(toolSchemas: unknown): string {
  return [
    `LogWarden operating contract ${OPERATING_CONTRACT_VERSION}.`,
    "Return exactly one JSON object and no other text.",
    'Request a tool as {"kind":"tool_request","tool":"...","arguments":{...}}.',
    'Finish as {"kind":"decision","incidentClass":"...","severity":"...","action":"...","actionArguments":{},"citedChunkIds":[],"confidence":0.0,"abstain":false,"correlationKey":"...","summary":"25 words maximum","rationale":"60 words maximum"}.',
    "For a decision, correlationKey, summary, and rationale are required top-level fields beside actionArguments; never put them inside actionArguments.",
    "Use actionArguments only for action-specific parameters, or {} when no parameters are needed.",
    `Incident classes: ${incidentClasses.join(", ")}.`,
    `Severities: ${severities.join(", ")}.`,
    `Actions: ${actions.join(", ")}.`,
    "Never invent a tool or cite a chunk that was not returned. Stop after a decision.",
    `Available tool schemas: ${JSON.stringify(toolSchemas)}`,
  ].join("\n");
}
