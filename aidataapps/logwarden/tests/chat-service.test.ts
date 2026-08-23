import { chatContainerName, chatDockerRunArguments, chatServerArguments } from "../src/chat-service.js";
import { resolveModelProfile } from "../src/models.js";

describe("pinned chat service command", () => {
  const profile = resolveModelProfile("qwen-smoke");
  const runtime = {
    project: "aidataapps-logwarden",
    containerName: "aidataapps-logwarden-chat",
    port: 8010,
    nested: true,
    hfVolume: "aidataapps-logwarden-huggingface-cache",
    vllmVolume: "aidataapps-logwarden-vllm-cache",
    maxNumSeqs: 64,
    batchInvariant: false,
  };

  it("uses one stable Lab 3 container identity", () => {
    expect(chatContainerName(runtime.project)).toBe(runtime.containerName);
    expect(() => chatContainerName("bad project")).toThrow(/Invalid/);
  });

  it("serves the registry model ID expected by the agent gateway", () => {
    const args = chatServerArguments(profile, runtime);
    expect(args.slice(args.indexOf("--served-model-name") + 1, args.indexOf("--served-model-name") + 2)).toEqual([profile.modelId]);
    expect(args).toContain("--generation-config");
    expect(args).toContain("vllm");
    expect(args).toContain("--port");
    expect(args).toContain("8010");
  });

  it("uses CDI and host networking in nested Colab without publishing a port", () => {
    const args = chatDockerRunArguments(profile, runtime, "a".repeat(64), false);
    expect(args).toContain("nvidia.com/gpu=all");
    expect(args).toContain("host");
    expect(args).not.toContain("--publish");
    expect(args).toContain(profile.vllmImage);
  });
});
