import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { z } from "zod";

const modelProfileSchema = z.object({
  description: z.string(),
  family: z.string(),
  modelId: z.string().min(1),
  revision: z.string().min(7),
  vllmImage: z.string().min(1),
  maxModelLen: z.number().int().positive(),
  gpuMemoryUtilization: z.number().positive().max(1),
  systemRole: z.boolean(),
  chatTemplateKwargs: z.record(z.string(), z.unknown()),
  vllmArgs: z.array(z.string()),
});

const modelRegistrySchema = z.object({
  schemaVersion: z.literal(1),
  defaultProfile: z.string(),
  benchmarkProfiles: z.array(z.string()),
  profiles: z.record(z.string(), modelProfileSchema),
});

export type ModelProfile = z.infer<typeof modelProfileSchema> & { key: string };
export type ModelRegistry = z.infer<typeof modelRegistrySchema>;

const defaultRegistryPath = fileURLToPath(
  new URL("../config/models.json", import.meta.url),
);

export function loadModelRegistry(path = defaultRegistryPath): ModelRegistry {
  const registry = modelRegistrySchema.parse(
    JSON.parse(readFileSync(path, "utf8")) as unknown,
  );

  const known = new Set(Object.keys(registry.profiles));
  const referenced = [registry.defaultProfile, ...registry.benchmarkProfiles];
  for (const key of referenced) {
    if (!known.has(key)) {
      throw new Error(`Model registry references missing profile: ${key}`);
    }
  }
  return registry;
}

export function resolveModelProfile(
  key: string,
  registry = loadModelRegistry(),
): ModelProfile {
  const profile = registry.profiles[key];
  if (!profile) {
    throw new Error(
      `Unknown model profile ${JSON.stringify(key)}. Available: ${Object.keys(registry.profiles).join(", ")}`,
    );
  }
  return { key, ...profile };
}
