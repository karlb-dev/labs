import { campaignInputManifest, frozenArmIdentities, loadDecodeRegistry, loadStandardCampaign } from "../src/campaign.js";

describe("standard campaign registry", () => {
  it("cross-validates model, arm, tool, baseline, and decode references", () => {
    const campaign = loadStandardCampaign();
    expect(campaign.targetProfiles).toEqual(["muse-glimmer-30b", "gemma-4-31b", "olmo-3.1-32b-instruct", "qwen-3.8-27b"]);
    expect(frozenArmIdentities()).toHaveLength(10);
    expect(campaignInputManifest().arms.map((arm) => arm.armId)).toContain("B1-rules-v1");
  });

  it("pins every decode field that can change generation", () => {
    expect(loadDecodeRegistry().configs["primary-json-v3"]).toEqual({
      temperature: 0, top_p: 1, top_k: 0, min_p: 0, repetition_penalty: 1,
      presence_penalty: 0, frequency_penalty: 0, seed: 0, max_tokens: 900,
      n: 1, stop: [], stream: false, transport: "structured_json",
    });
  });
});
