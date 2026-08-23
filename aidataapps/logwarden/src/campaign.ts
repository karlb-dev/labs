import { readFileSync } from "node:fs";
import { z } from "zod";
import { OPERATING_CONTRACT_VERSION, operatingContract } from "./contracts.js";
import { hashJson, sha256 } from "./hash.js";
import { loadModelRegistry } from "./models.js";
import { LAB_ROOT } from "./run.js";
import { loadRulesBaseline } from "./rules-baseline.js";
import { loadToolRegistry, promptToolSchemas, type ToolName } from "./tools.js";

const armSchema = z.object({
  armId: z.string().regex(/^[A-Za-z][A-Za-z0-9_.-]{0,79}$/),
  kind: z.enum(["agent", "baseline", "derived", "evaluator"]),
  modelRequired: z.boolean(),
  allowedTools: z.array(z.string().min(1).max(80)),
  retrievalMode: z.enum(["none", "lexical_fulltext", "vector_exact", "hybrid_rrf", "oracle_runbook"]),
  correlationMode: z.literal("frozen_packet"),
  execution: z.string().min(1).max(120),
}).strict();

const armRegistrySchema = z.object({
  schemaVersion: z.literal(1),
  policyVersion: z.string().min(1),
  packetVersion: z.string().min(1),
  decodeConfigId: z.string().min(1),
  arms: z.array(armSchema).min(1),
}).strict();

const decodeConfigSchema = z.object({
  temperature: z.literal(0), top_p: z.literal(1), top_k: z.literal(0), min_p: z.literal(0),
  repetition_penalty: z.literal(1), presence_penalty: z.literal(0), frequency_penalty: z.literal(0),
  seed: z.number().int(), max_tokens: z.number().int().min(256).max(4096), n: z.literal(1),
  stop: z.array(z.string()).max(20), stream: z.literal(false), transport: z.literal("structured_json"),
}).strict();

const decodeRegistrySchema = z.object({
  schemaVersion: z.literal(1),
  configs: z.record(z.string(), decodeConfigSchema),
}).strict();

const campaignSchema = z.object({
  schemaVersion: z.literal(1), campaignId: z.string().min(1), tier: z.literal("standard"),
  scenarioCatalog: z.string().min(1), runbookCorpus: z.string().min(1), agentArms: z.string().min(1),
  decodeConfigs: z.string().min(1), rulesBaseline: z.string().min(1), embeddingProfile: z.string().min(1),
  targetProfiles: z.array(z.string()).length(4), qualityRoles: z.array(z.string()).min(1),
  mandatoryInferenceArms: z.array(z.string()).min(1),
  retrievalArm: z.object({ armId: z.string(), eligibility: z.literal("expected_runbooks_nonempty") }).strict(),
  derivedArms: z.array(z.string()), baselines: z.array(z.string()).min(1),
  workerCount: z.number().int().min(1).max(64), oneResidentChatModel: z.literal(true),
  agentBudget: z.object({
    maxModelTurns: z.number().int().positive(), maxToolCalls: z.number().int().nonnegative(),
    maxSameToolCalls: z.number().int().nonnegative(), maxToolResultChars: z.number().int().positive(),
    maxTotalToolResultChars: z.number().int().positive(), maxWallTimeSeconds: z.number().int().positive(),
  }).strict(),
  controls: z.object({
    subsetEpisodes: z.number().int().positive(), roles: z.array(z.string()).min(1),
    selection: z.literal("hash-stratified-family-regime-v1"), inferenceArm: z.string(),
  }).strict(),
  statistics: z.object({
    unit: z.literal("episode"), resamplingGroup: z.literal("scenario_group_id"), paired: z.literal(true),
    bootstrapReplicates: z.number().int().min(1000), permutationReplicates: z.number().int().min(1000),
    alpha: z.number().positive().max(0.1), multipleComparison: z.literal("Holm"),
    primaryHypothesisFamilySize: z.number().int().positive(), powerTarget: z.number().positive().max(1),
    powerEffectAbsolute: z.number().positive().max(1), powerBaseAccuracy: z.number().positive().max(1),
  }).strict(),
  safety: z.object({ liveRemediation: z.literal(false), writeTools: z.literal(false), confirmedViolationTolerance: z.literal(0) }).strict(),
}).strict();

export type AgentArmRegistry = z.infer<typeof armRegistrySchema>;
export type AgentArmDefinition = z.infer<typeof armSchema>;
export type DecodeRegistry = z.infer<typeof decodeRegistrySchema>;
export type StandardCampaignConfig = z.infer<typeof campaignSchema>;

export interface FrozenArmIdentity {
  armId: string;
  kind: AgentArmDefinition["kind"];
  modelRequired: boolean;
  promptSha256: string;
  toolRegistrySha256: string;
  policySha256: string;
  contractSha256: string;
  packetVersion: string;
  retrievalMode: AgentArmDefinition["retrievalMode"];
  correlationMode: "frozen_packet";
  decodeConfigId: string;
  execution: string;
}

export function loadAgentArmRegistry(path = `${LAB_ROOT}/config/agent-arms.json`): AgentArmRegistry {
  const registry = armRegistrySchema.parse(JSON.parse(readFileSync(path, "utf8")));
  const tools = loadToolRegistry();
  const seen = new Set<string>();
  for (const arm of registry.arms) {
    if (seen.has(arm.armId)) throw new Error(`Duplicate agent arm ${arm.armId}`);
    seen.add(arm.armId);
    for (const tool of arm.allowedTools) if (!(tool in tools.tools)) throw new Error(`Arm ${arm.armId} references unknown tool ${tool}`);
    if (!arm.modelRequired && arm.kind === "agent") throw new Error(`Agent arm ${arm.armId} must require a model`);
  }
  return registry;
}

export function loadDecodeRegistry(path = `${LAB_ROOT}/config/decode.json`): DecodeRegistry {
  return decodeRegistrySchema.parse(JSON.parse(readFileSync(path, "utf8")));
}

export function loadStandardCampaign(path = `${LAB_ROOT}/config/campaigns/standard.json`): StandardCampaignConfig {
  const campaign = campaignSchema.parse(JSON.parse(readFileSync(path, "utf8")));
  const models = loadModelRegistry();
  if (hashJson(campaign.targetProfiles) !== hashJson(models.targetProfiles)) throw new Error("Campaign target profile order drifts from the pinned model registry");
  if (!(campaign.embeddingProfile in models.embeddingProfiles)) throw new Error(`Campaign embedding profile is unknown: ${campaign.embeddingProfile}`);
  const arms = loadAgentArmRegistry(resolveLabPath(campaign.agentArms));
  const armIds = new Set(arms.arms.map((arm) => arm.armId));
  for (const arm of [...campaign.mandatoryInferenceArms, campaign.retrievalArm.armId, ...campaign.derivedArms, ...campaign.baselines, campaign.controls.inferenceArm]) {
    if (!armIds.has(arm)) throw new Error(`Campaign references missing arm ${arm}`);
  }
  const decodes = loadDecodeRegistry(resolveLabPath(campaign.decodeConfigs));
  if (!(arms.decodeConfigId in decodes.configs)) throw new Error(`Arm registry decode config is missing: ${arms.decodeConfigId}`);
  return campaign;
}

export function frozenArmIdentities(registry = loadAgentArmRegistry()): FrozenArmIdentity[] {
  const tools = loadToolRegistry();
  const rules = loadRulesBaseline();
  return registry.arms.map((arm) => {
    const allowedTools = arm.allowedTools as ToolName[];
    const promptSchemas = promptToolSchemas(tools, allowedTools);
    const promptTemplate = arm.kind === "agent" || arm.kind === "derived"
      ? operatingContract(promptSchemas)
      : `${arm.armId} has no model prompt`;
    const policy = arm.armId === rules.baselineId
      ? { policyVersion: registry.policyVersion, rulesetSha256: hashJson(rules), execution: arm.execution }
      : { policyVersion: registry.policyVersion, noRemediation: true, unknownRequiresAbstention: true, execution: arm.execution };
    return {
      armId: arm.armId, kind: arm.kind, modelRequired: arm.modelRequired,
      promptSha256: sha256(promptTemplate), toolRegistrySha256: hashJson(promptSchemas),
      policySha256: hashJson(policy), contractSha256: hashJson({ version: OPERATING_CONTRACT_VERSION, kind: arm.kind }),
      packetVersion: registry.packetVersion, retrievalMode: arm.retrievalMode,
      correlationMode: arm.correlationMode, decodeConfigId: registry.decodeConfigId, execution: arm.execution,
    };
  });
}

export function campaignInputManifest(campaign = loadStandardCampaign()) {
  const files = {
    scenarioCatalog: campaign.scenarioCatalog, runbookCorpus: campaign.runbookCorpus,
    agentArms: campaign.agentArms, decodeConfigs: campaign.decodeConfigs, rulesBaseline: campaign.rulesBaseline,
  };
  return {
    schemaVersion: 1,
    campaign: { config: campaign, sha256: hashJson(campaign) },
    files: Object.fromEntries(Object.entries(files).map(([key, path]) => {
      const parsed = JSON.parse(readFileSync(resolveLabPath(path), "utf8")) as unknown;
      return [key, { path, sha256: hashJson(parsed) }];
    })),
    arms: frozenArmIdentities(),
  };
}

function resolveLabPath(path: string): string {
  return path.startsWith("/") ? path : `${LAB_ROOT}/${path}`;
}
