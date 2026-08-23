import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  assertAuditableNonModelMetadataDrift,
  differingArmIdentityFields,
} from "../src/arm-identity-sync.js";
import type { FrozenArmIdentity } from "../src/campaign.js";

const base: FrozenArmIdentity = {
  armId: "B2-hybrid-v1",
  kind: "baseline",
  modelRequired: false,
  promptSha256: "a".repeat(64),
  toolRegistrySha256: "b".repeat(64),
  policySha256: "c".repeat(64),
  contractSha256: "d".repeat(64),
  packetVersion: "incident-packet-v1",
  retrievalMode: "hybrid_rrf",
  correlationMode: "frozen_packet",
  decodeConfigId: "primary-json-v3",
  execution: "config/baselines/derived-v1.json#retrievalOnly",
};

describe("pre-freeze arm identity synchronization", () => {
  it("allows only irrelevant contract/tool prompt metadata on a non-model retrieval baseline", () => {
    const replacement = { ...base, toolRegistrySha256: "e".repeat(64), contractSha256: "f".repeat(64) };
    expect(differingArmIdentityFields(base, replacement)).toEqual(["toolRegistrySha256", "contractSha256"]);
    expect(assertAuditableNonModelMetadataDrift(base, replacement)).toEqual(["toolRegistrySha256", "contractSha256"]);
  });

  it("rejects policy or execution changes", () => {
    expect(() => assertAuditableNonModelMetadataDrift(base, { ...base, policySha256: "e".repeat(64) }))
      .toThrow(/execution-relevant drift/);
  });

  it("rejects model-backed arms", () => {
    const agent = { ...base, armId: "A-tools", kind: "agent" as const, modelRequired: true, contractSha256: "e".repeat(64) };
    expect(() => assertAuditableNonModelMetadataDrift({ ...agent, contractSha256: base.contractSha256 }, agent))
      .toThrow(/model-backed/);
  });

  it("does not allow tool metadata drift on a no-retrieval baseline", () => {
    const noRetrieval = { ...base, armId: "B0-majority-no-action-v1", retrievalMode: "none" as const };
    expect(() => assertAuditableNonModelMetadataDrift(noRetrieval, { ...noRetrieval, toolRegistrySha256: "e".repeat(64) }))
      .toThrow(/execution-relevant drift/);
  });

  it("scopes prediction guards through their owning campaign jobs", () => {
    const source = readFileSync(new URL("../scripts/sync-prefreeze-arm-identities.ts", import.meta.url), "utf8");
    expect(source).not.toContain("prediction.campaign_id");
    expect(source.match(/INNER JOIN control\.jobs job ON job\.job_id=prediction\.job_id/g)).toHaveLength(3);
  });
});
