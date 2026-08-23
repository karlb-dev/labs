import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { canonicalJson, hashJson } from "./hash.js";
import { incidentClasses } from "./contracts.js";

const databaseName = z.string().regex(/^(?:LogWardenWorkload|LW_[A-Za-z0-9_]{1,60})$/i);
const maxRows = z.number().int().min(1).max(100).default(20);

const schemas = {
  runbook_search: z.object({
    query: z.string().trim().min(1).max(1000),
    topK: z.number().int().min(1).max(20).default(5),
    corpusId: z.string().regex(/^[A-Za-z][A-Za-z0-9_-]{0,79}$/).default("primary-v1"),
  }).strict(),
  get_recent_incident_counts: z.object({
    incidentClass: z.enum(incidentClasses).nullable().default(null),
    windowMinutes: z.number().int().min(1).max(10080).default(60),
  }).strict(),
  get_blocking_snapshot: z.object({ databaseName, maxRows }).strict(),
  get_log_space: z.object({ databaseName }).strict(),
  get_active_transactions: z.object({ databaseName, maxRows }).strict(),
  get_backup_history: z.object({ databaseName, maxRows }).strict(),
  get_deadlock_graph: z.object({ maxRows: z.number().int().min(1).max(20).default(5) }).strict(),
} as const;

export type ToolName = keyof typeof schemas;
export const toolNames = Object.freeze(Object.keys(schemas).sort() as ToolName[]);

const registryToolSchema = z.object({
  procedure: z.string().regex(/^(?:agent|kb|ops)\.[A-Za-z][A-Za-z0-9_]*$/),
  modes: z.array(z.enum(["replay", "live"])).min(1),
  mutates: z.literal(false),
  tier: z.literal(1),
  arguments: z.array(z.string()),
}).strict();
const registrySchema = z.object({
  schemaVersion: z.literal(1),
  registryVersion: z.literal("tools-v1"),
  frozen: z.boolean(),
  tools: z.record(z.string(), registryToolSchema),
}).strict();
export type ToolRegistry = z.infer<typeof registrySchema>;

const defaultPath = fileURLToPath(new URL("../config/tools.json", import.meta.url));

export function loadToolRegistry(path = defaultPath): ToolRegistry {
  const registry = registrySchema.parse(JSON.parse(readFileSync(path, "utf8")));
  const expected = [...toolNames];
  const actual = Object.keys(registry.tools).sort();
  if (canonicalJson(expected) !== canonicalJson(actual)) throw new Error(`Tool registry/schema mismatch: expected ${expected.join(",")}; got ${actual.join(",")}`);
  for (const name of expected as ToolName[]) {
    const declared = registry.tools[name]!;
    const shape = new Set(Object.keys(schemas[name].shape));
    if (declared.arguments.some((argument) => !shape.has(argument)) || declared.arguments.length !== shape.size) {
      throw new Error(`Tool registry argument drift for ${name}`);
    }
  }
  return registry;
}

export function isToolName(value: string): value is ToolName {
  return Object.hasOwn(schemas, value);
}

export function promptToolSchemas(
  registry = loadToolRegistry(),
  allowed: readonly ToolName[] = toolNames,
): Array<{ name: ToolName; arguments: string[]; modes: string[]; mutates: false }> {
  const allowedSet = new Set(allowed);
  return toolNames
    .filter((name) => allowedSet.has(name))
    .map((name) => ({
      name,
      arguments: [...registry.tools[name]!.arguments],
      modes: [...registry.tools[name]!.modes],
      mutates: false as const,
    }));
}

export function canonicalToolArguments(name: ToolName, input: unknown): Record<string, unknown> {
  const parsed = schemas[name].parse(input) as Record<string, unknown>;
  const normalized = { ...parsed };
  for (const key of ["databaseName", "corpusId", "incidentClass"] as const) {
    const value = normalized[key];
    if (typeof value === "string") normalized[key] = value.toLowerCase();
  }
  if (typeof normalized.query === "string") normalized.query = normalized.query.replace(/\s+/g, " ").trim();
  return JSON.parse(canonicalJson(normalized)) as Record<string, unknown>;
}

export function canonicalToolArgumentsSha256(name: ToolName, input: unknown): string {
  return hashJson(canonicalToolArguments(name, input));
}

export function toolRegistrySha256(registry = loadToolRegistry()): string {
  return hashJson(registry);
}
