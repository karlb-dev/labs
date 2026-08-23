import {
  chatContainerName,
  chatDockerRunArguments,
  chatServerArguments,
  fatalChatServiceLogSignatures,
  terminalChatContainerFailure,
} from "../src/chat-service.js";
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

  it("distinguishes a slow startup from a terminal container", () => {
    expect(terminalChatContainerFailure({ Status: "running", Running: true, ExitCode: 0 })).toBeNull();
    expect(terminalChatContainerFailure({ Status: "restarting", Running: false, ExitCode: 1 })).toBeNull();
    expect(terminalChatContainerFailure({
      Status: "exited",
      Running: false,
      OOMKilled: false,
      ExitCode: 1,
      Error: "",
      FinishedAt: "2026-08-23T11:24:10Z",
    })).toContain("status=exited running=false exitCode=1 oomKilled=false");
  });

  it("does not treat recoverable compiler warning stacks as fatal service failures", () => {
    const warning = [
      "[rank0]:W torch/_inductor/triton_bundler.py Failed to reload cubin file",
      "[rank0]:W torch/_inductor/triton_bundler.py Traceback (most recent call last):",
      "[rank0]:W RuntimeError: Cubin file saved by TritonBundler not found",
      "INFO Directly load the compiled graph(s) from the cache",
      "INFO Application startup complete.",
    ].join("\n");
    expect(fatalChatServiceLogSignatures(warning)).toEqual([]);
    expect(fatalChatServiceLogSignatures("torch.OutOfMemoryError: CUDA out of memory")).toEqual(["cuda_out_of_memory"]);
    expect(fatalChatServiceLogSignatures("RuntimeError: Engine core initialization failed")).toEqual(["engine_initialization_failed"]);
  });
});
