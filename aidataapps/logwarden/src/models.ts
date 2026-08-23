import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { z } from "zod";

const profileSchema = z.object({
  description: z.string(),
  family: z.string(),
  modelId: z.string().min(1),
  revision: z.string().length(40),
  vllmSourceTag: z.string().min(1),
  vllmImage: z.string().regex(/@sha256:[0-9a-f]{64}$/),
  imageProvenance: z.enum(["registry-pull", "local-build"]),
  maxModelLen: z.number().int().positive(),
  gpuMemoryUtilization: z.number().positive().max(1),
  maxNumSeqs: z.number().int().positive(),
  chatTemplateKwargs: z.record(z.string(), z.unknown()),
  campaignArgs: z.array(z.string()),
});

const embeddingSchema = z.object({
  modelId: z.string(),
  revision: z.string().length(40),
  dimensions: z.number().int().positive(),
  vllmImage: z.string().regex(/@sha256:[0-9a-f]{64}$/),
});

const registrySchema = z.object({
  schemaVersion: z.literal(2),
  syncedFrom: z.string(),
  sourceCommit: z.string().length(40),
  defaultProfile: z.string(),
  targetProfiles: z.array(z.string()).length(4),
  profiles: z.record(z.string(), profileSchema),
  embeddingProfiles: z.record(z.string(), embeddingSchema),
});

export type ModelProfile = z.infer<typeof profileSchema> & { key: string };
export type EmbeddingProfile = z.infer<typeof embeddingSchema> & { key: string };
export type ModelRegistry = z.infer<typeof registrySchema>;

const defaultPath = fileURLToPath(new URL("../config/models.json", import.meta.url));

export function loadModelRegistry(path = defaultPath): ModelRegistry {
  const registry = registrySchema.parse(JSON.parse(readFileSync(path, "utf8")));
  const known = new Set(Object.keys(registry.profiles));
  for (const key of [registry.defaultProfile, ...registry.targetProfiles]) {
    if (!known.has(key)) throw new Error(`Registry references missing profile: ${key}`);
  }
  if (new Set(registry.targetProfiles).size !== 4) {
    throw new Error("The target profile list must contain four unique profiles");
  }
  return registry;
}

export function resolveModelProfile(key: string, registry = loadModelRegistry()): ModelProfile {
  const profile = registry.profiles[key];
  if (!profile) throw new Error(`Unknown profile ${JSON.stringify(key)}`);
  return { key, ...profile };
}

export function resolveEmbeddingProfile(key: string, registry = loadModelRegistry()): EmbeddingProfile {
  const profile = registry.embeddingProfiles[key];
  if (!profile) throw new Error(`Unknown embedding profile ${JSON.stringify(key)}`);
  return { key, ...profile };
}
