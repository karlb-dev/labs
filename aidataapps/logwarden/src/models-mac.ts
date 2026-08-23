import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { z } from "zod";

// Mac-plane serving registry (Azure Foundry Local). Deliberately separate
// from the frozen vLLM registry in src/models.ts: mac profiles are pinned by
// Foundry variant id instead of image digest and are never campaign targets.

const macProfileSchema = z.object({
  description: z.string(),
  family: z.string(),
  foundryAlias: z.string().min(1),
  foundryVariantId: z.string(),
  executionProvider: z.string().min(1),
  contextLength: z.number().int().positive(),
  fileSizeMb: z.number().int().positive(),
  role: z.enum(["plumbing", "agent"]),
});

const macEmbeddingSchema = z.object({
  foundryAlias: z.string().min(1),
  foundryVariantId: z.string(),
  baseModelId: z.string().min(1),
  expectedDimensions: z.number().int().positive(),
  note: z.string(),
});

const macRegistrySchema = z.object({
  schemaVersion: z.literal(1),
  platform: z.literal("mac-foundry-local"),
  runtime: z.object({
    tool: z.literal("azure-foundry-local"),
    cliVersionTested: z.string().min(1),
    openAiCompatible: z.literal(true),
  }),
  comparability: z.string().min(1),
  defaultProfile: z.string(),
  smokeProfile: z.string(),
  profiles: z.record(z.string(), macProfileSchema),
  embeddingProfiles: z.record(z.string(), macEmbeddingSchema),
});

export type MacModelProfile = z.infer<typeof macProfileSchema> & { key: string };
export type MacEmbeddingProfile = z.infer<typeof macEmbeddingSchema> & { key: string };
export type MacModelRegistry = z.infer<typeof macRegistrySchema>;

const defaultPath = fileURLToPath(new URL("../config/models.mac.json", import.meta.url));

export function loadMacModelRegistry(path = defaultPath): MacModelRegistry {
  const registry = macRegistrySchema.parse(JSON.parse(readFileSync(path, "utf8")));
  for (const key of [registry.defaultProfile, registry.smokeProfile]) {
    if (!registry.profiles[key]) throw new Error(`Mac registry references missing profile: ${key}`);
  }
  return registry;
}

export function resolveMacProfile(key: string, registry = loadMacModelRegistry()): MacModelProfile {
  const chat = registry.profiles[key];
  if (chat) return { key, ...chat };
  throw new Error(`Unknown mac profile ${JSON.stringify(key)}`);
}

export function resolveMacEmbeddingProfile(key: string, registry = loadMacModelRegistry()): MacEmbeddingProfile {
  const embedding = registry.embeddingProfiles[key];
  if (embedding) return { key, ...embedding };
  throw new Error(`Unknown mac embedding profile ${JSON.stringify(key)}`);
}
