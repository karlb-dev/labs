import type { FrozenArmIdentity } from "./campaign.js";

const identityFields: ReadonlyArray<keyof FrozenArmIdentity> = [
  "armId",
  "kind",
  "modelRequired",
  "promptSha256",
  "toolRegistrySha256",
  "policySha256",
  "contractSha256",
  "packetVersion",
  "retrievalMode",
  "correlationMode",
  "decodeConfigId",
  "execution",
];

export function differingArmIdentityFields(
  prior: FrozenArmIdentity,
  replacement: FrozenArmIdentity,
): Array<keyof FrozenArmIdentity> {
  return identityFields.filter((field) => prior[field] !== replacement[field]);
}

export function assertAuditableNonModelMetadataDrift(
  prior: FrozenArmIdentity,
  replacement: FrozenArmIdentity,
): Array<keyof FrozenArmIdentity> {
  const differences = differingArmIdentityFields(prior, replacement);
  if (differences.length === 0) throw new Error(`Arm ${replacement.armId} has no identity drift`);
  if (prior.armId !== replacement.armId) throw new Error("Arm identity synchronization cannot rename an arm");
  if (prior.modelRequired || replacement.modelRequired) {
    throw new Error(`Arm ${replacement.armId} is model-backed; metadata-only synchronization is forbidden`);
  }
  if (!(["baseline", "evaluator"] as const).includes(replacement.kind as "baseline" | "evaluator") || prior.kind !== replacement.kind) {
    throw new Error(`Arm ${replacement.armId} is not a stable non-model baseline/evaluator`);
  }
  const allowed = new Set<keyof FrozenArmIdentity>(["contractSha256"]);
  if (replacement.kind === "baseline" && replacement.retrievalMode !== "none") {
    allowed.add("toolRegistrySha256");
  }
  const forbidden = differences.filter((field) => !allowed.has(field));
  if (forbidden.length > 0) {
    throw new Error(`Arm ${replacement.armId} has execution-relevant drift: ${forbidden.join(",")}`);
  }
  return differences;
}
