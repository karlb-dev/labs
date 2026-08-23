import type { ModelProfile } from "./models.js";

export interface ChatRuntimeIdentity {
  project: string;
  containerName: string;
  port: number;
  nested: boolean;
  hfVolume: string;
  vllmVolume: string;
  maxNumSeqs: number;
  batchInvariant: boolean;
}

export interface ChatContainerState {
  Status?: string;
  Running?: boolean;
  OOMKilled?: boolean;
  ExitCode?: number;
  Error?: string;
  StartedAt?: string;
  FinishedAt?: string;
}

export function terminalChatContainerFailure(state: ChatContainerState): string | null {
  const status = state.Status?.toLowerCase() ?? "unknown";
  if (status === "created" || status === "running" || status === "restarting") return null;
  return [
    `status=${status}`,
    `running=${String(state.Running ?? false)}`,
    `exitCode=${String(state.ExitCode ?? "unknown")}`,
    `oomKilled=${String(state.OOMKilled ?? false)}`,
    `error=${JSON.stringify(state.Error ?? "")}`,
    `finishedAt=${state.FinishedAt ?? "unknown"}`,
  ].join(" ");
}

export function chatContainerName(project: string): string {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$/.test(project)) throw new Error("Invalid Compose project name");
  return `${project}-chat`;
}

export function chatServerArguments(profile: ModelProfile, runtime: ChatRuntimeIdentity): string[] {
  if (!Number.isSafeInteger(runtime.port) || runtime.port < 1 || runtime.port > 65_535)
    throw new Error("Invalid chat service port");
  if (!Number.isSafeInteger(runtime.maxNumSeqs) || runtime.maxNumSeqs < 1)
    throw new Error("Invalid max-num-seqs override");
  return [
    "--model", profile.modelId,
    "--revision", profile.revision,
    "--tokenizer-revision", profile.revision,
    "--served-model-name", profile.modelId,
    "--max-model-len", String(profile.maxModelLen),
    "--gpu-memory-utilization", String(profile.gpuMemoryUtilization),
    "--max-num-seqs", String(runtime.maxNumSeqs),
    "--enable-log-requests",
    ...(runtime.nested ? ["--port", String(runtime.port)] : []),
    ...profile.campaignArgs,
  ];
}

export function chatDockerRunArguments(
  profile: ModelProfile,
  runtime: ChatRuntimeIdentity,
  profileHash: string,
  hasHfToken: boolean,
): string[] {
  const args = [
    "run", "--detach", "--name", runtime.containerName,
    "--label", "ai.labs.lab=logwarden",
    "--label", "ai.labs.role=chat",
    "--label", `ai.labs.model-profile=${profile.key}`,
    "--label", `ai.labs.model-profile-hash=${profileHash}`,
    "--label", `ai.labs.max-num-seqs=${runtime.maxNumSeqs}`,
  ];
  if (runtime.nested) {
    args.push(
      "--device", "nvidia.com/gpu=all",
      "--pid", "host",
      "--network", "host",
      "--ipc", "host",
      "--cgroupns", "host",
      "--security-opt", "seccomp=unconfined",
      "--security-opt", "apparmor=unconfined",
      "--mount", "type=bind,src=/proc,dst=/proc,readonly",
      "--mount", "type=bind,src=/sys/fs/cgroup,dst=/sys/fs/cgroup,readonly",
      "--env", "LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:/usr/local/nvidia/lib64",
    );
  } else {
    args.push("--gpus", "all", "--ipc", "host", "--publish", `${runtime.port}:8000`);
  }
  args.push(
    "--volume", `${runtime.hfVolume}:/root/.cache/huggingface`,
    "--volume", `${runtime.vllmVolume}:/root/.cache/vllm`,
    "--env", "VLLM_ENABLE_CUDA_COMPATIBILITY=1",
  );
  if (runtime.batchInvariant) {
    args.push("--env", "VLLM_BATCH_INVARIANT=1", "--label", "ai.labs.batch-invariant=1");
  }
  if (hasHfToken) args.push("--env", "HF_TOKEN");
  args.push(profile.vllmImage, ...chatServerArguments(profile, runtime));
  return args;
}
