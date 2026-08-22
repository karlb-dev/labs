import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { conformalCandidateSet, decide, weightedVote } from "../src/attribution.js";
import { hashJson } from "../src/hash.js";
import { loadModelRegistry, resolveModelProfile } from "../src/models.js";
import {
  assignSplit, loadCarriers, maskNames, relationCompletionToQuestion, renderCarrier,
  scanSelfName, scanTemplateResidue, stripTemplateResidue,
} from "../src/prompt-bank.js";
import { STYLE_SCHEMA_HASH, styleVector } from "../src/style.js";
import { decodeCell } from "../src/types.js";

describe("frozen generation contract", () => {
  it("emits every sampling field for every decode cell", () => {
    for (const key of ["det", "nat-0", "nat-1", "hv"] as const) {
      const cell = decodeCell(key, 600);
      expect(Object.keys(cell).sort()).toEqual([
        "frequency_penalty", "key", "max_tokens", "min_p", "n", "presence_penalty",
        "repetition_penalty", "seed", "stop", "temperature", "top_k", "top_p",
      ]);
      expect(cell).toMatchObject({ top_k: -1, min_p: 0, repetition_penalty: 1, presence_penalty: 0, frequency_penalty: 0, n: 1, stop: [] });
    }
  });

  it("keeps target identities and serving images immutable", () => {
    const registry = loadModelRegistry();
    expect(registry.targetProfiles).toHaveLength(4);
    expect(new Set(registry.targetProfiles)).toHaveLength(4);
    for (const key of registry.targetProfiles) {
      const profile = resolveModelProfile(key, registry);
      expect(profile.revision).toMatch(/^[0-9a-f]{40}$/);
      expect(profile.vllmImage).toMatch(/@sha256:[0-9a-f]{64}$/);
      expect(profile.campaignArgs.join(" ")).toContain("--generation-config vllm");
    }
  });
});

describe("prompt and leakage controls", () => {
  it("renders a single byte-stable user carrier", () => {
    const carrier = loadCarriers().get("explain-v1")!;
    const left = renderCarrier(carrier, "Why is the sky blue?");
    const right = renderCarrier(carrier, "Why is the sky blue?");
    expect(left).toBe(right);
    expect(left).not.toMatch(/^assistant\s*:/im);
  });

  it("converts completion probes to questions", () => {
    expect(relationCompletionToQuestion("The capital of France is")).toBe("What is the capital of France?");
  });

  it("finds residue, masks names idempotently, and strips residue", () => {
    const source = "<think>hidden</think>\nAssistant: I am Gemma by Google.";
    expect(scanTemplateResidue(source).templateResidueFound).toBe(true);
    expect(scanSelfName(source).selfNameFound).toBe(true);
    const masked = maskNames(source);
    expect(maskNames(masked)).toBe(masked);
    expect(masked).toContain("[MODEL]");
    expect(stripTemplateResidue(source)).not.toMatch(/<think>|Assistant:/i);
  });

  it("assigns one deterministic split and source-holds OASST", () => {
    expect(assignSplit("same", "core")).toBe(assignSplit("same", "core"));
    expect(assignSplit("any", "oasst1")).toBe("test_source_holdout");
  });

  it("keeps frozen tier groups nested", () => {
    const load = (tier: string) => JSON.parse(readFileSync(new URL(`../data/manifests/tier-${tier}.json`, import.meta.url), "utf8")) as { groupIds: string[] };
    const tiers = [load("smoke"), load("dev"), load("standard"), load("full")];
    for (let index = 0; index < tiers.length - 1; index += 1) {
      const next = new Set(tiers[index + 1]!.groupIds);
      expect(tiers[index]!.groupIds.every((id) => next.has(id))).toBe(true);
    }
  });
});

describe("representations and attribution math", () => {
  it("builds a deterministic finite 512-vector", () => {
    const text = "## Answer\n\n- First item\n- Second item\n\nHowever, this is only a test.";
    const vector = styleVector(text);
    expect(vector).toHaveLength(512);
    expect(vector.every(Number.isFinite)).toBe(true);
    expect(styleVector(text)).toEqual(vector);
    expect(STYLE_SCHEMA_HASH).toMatch(/^[0-9a-f]{64}$/);
    expect(hashJson(vector)).not.toBe(hashJson(styleVector("A short plain answer.")));
  });

  it("deduplicates group and text evidence before voting", () => {
    const vote = weightedVote([
      { model: "a", distance: 0.1, promptGroupId: "g1", textArtifactId: "t1" },
      { model: "a", distance: 0.11, promptGroupId: "g1", textArtifactId: "t2" },
      { model: "b", distance: 0.2, promptGroupId: "g2", textArtifactId: "t3" },
    ], 0.2);
    expect(vote.a! + vote.b!).toBeCloseTo(1);
    expect(vote.a).toBeGreaterThan(vote.b!);
  });

  it("returns conformal sets and bounded user decisions", () => {
    expect(conformalCandidateSet({ a: 0.9, b: 0.1 }, 0.2)).toEqual(["a"]);
    expect(decide({ referenceTokens: 8, supportedLanguage: true, novelty: 0, noveltyThreshold: 1, candidateSet: ["a"], topProbability: 0.9 })).toBe("insufficient_text");
    expect(decide({ referenceTokens: 80, supportedLanguage: true, novelty: 2, noveltyThreshold: 1, candidateSet: ["a"], topProbability: 0.9 })).toBe("unknown_or_out_of_distribution");
  });
});
