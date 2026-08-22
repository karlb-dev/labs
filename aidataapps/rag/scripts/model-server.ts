import { spawnSync } from "node:child_process";
import { chmodSync, existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import dotenv from "dotenv";
import { loadModelRegistry, resolveModelProfile } from "../src/models.js";

dotenv.config({ quiet: true });

const project = process.env.COMPOSE_PROJECT_NAME ?? "aidataapps-rag";
const containerName = `${project}-chat`;
const nestedColab = process.env.CONTAINER_RUNTIME_PROFILE === "colab-rootless";
if (nestedColab && !process.env.DOCKER_HOST) {
  process.env.DOCKER_HOST = "unix:///run/user/1000/docker.sock";
}

function docker(args: string[], options: { quiet?: boolean } = {}): string {
  const result = spawnSync("docker", args, {
    encoding: "utf8",
    stdio: options.quiet ? "pipe" : ["inherit", "pipe", "pipe"],
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(
      `docker ${args[0] ?? ""} failed: ${(result.stderr || result.stdout).trim()}`,
    );
  }
  return result.stdout.trim();
}

function hasContainer(): boolean {
  const result = spawnSync("docker", ["container", "inspect", containerName], {
    stdio: "ignore",
  });
  return result.status === 0;
}

function currentContainerProfile(): string | null {
  if (!hasContainer()) return null;
  return (
    docker(
      [
        "inspect",
        "--format",
        '{{index .Config.Labels "ai.labs.model-profile"}}',
        containerName,
      ],
      { quiet: true },
    ) || null
  );
}

function removeContainer(): void {
  if (!hasContainer()) return;
  if (nestedColab) {
    // A vLLM worker in Colab's shared PID namespace can be slow to acknowledge
    // Docker's graceful stop. This container is stateless, so terminate it
    // directly instead of blocking profile switches indefinitely.
    docker(["kill", containerName]);
  } else {
    docker(["stop", "--time", "20", containerName]);
  }
  docker(["rm", containerName]);
}

function valueAfter(flag: string): string | undefined {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function persistModelProfile(profileKey: string): void {
  const envPath = ".env";
  if (!existsSync(envPath)) return;
  const current = readFileSync(envPath, "utf8");
  const next = /^MODEL_PROFILE=/m.test(current)
    ? current.replace(/^MODEL_PROFILE=.*$/m, `MODEL_PROFILE=${profileKey}`)
    : `${current.replace(/\n?$/, "\n")}MODEL_PROFILE=${profileKey}\n`;
  const temporaryPath = `.env.model-profile.${process.pid}.tmp`;
  writeFileSync(temporaryPath, next, { encoding: "utf8", mode: 0o600 });
  chmodSync(temporaryPath, 0o600);
  renameSync(temporaryPath, envPath);
}

const command = process.argv[2] ?? "list";
const registry = loadModelRegistry();

if (command === "list") {
  console.log("PROFILE\tFAMILY\tMODEL\tREVISION\tIMAGE");
  for (const [key, profile] of Object.entries(registry.profiles)) {
    console.log(
      `${key}\t${profile.family}\t${profile.modelId}\t${profile.revision.slice(0, 12)}\t${profile.vllmImage}`,
    );
  }
} else if (command === "benchmark-list") {
  console.log(registry.benchmarkProfiles.join("\n"));
} else if (command === "status") {
  if (!hasContainer()) {
    console.log("stopped");
  } else {
    console.log(
      docker(
        [
          "inspect",
          "--format",
          "{{.State.Status}} profile={{index .Config.Labels \"ai.labs.model-profile\"}} image={{.Image}}",
          containerName,
        ],
        { quiet: true },
      ),
    );
  }
} else if (command === "stop") {
  removeContainer();
} else if (command === "evict") {
  const profileKey = valueAfter("--profile") ?? process.env.MODEL_PROFILE ?? registry.defaultProfile;
  const profile = resolveModelProfile(profileKey, registry);
  if (currentContainerProfile() === profile.key) {
    removeContainer();
  }
  const args = ["run", "--rm"];
  if (nestedColab) {
    args.push(
      "--pid",
      "host",
      "--network",
      "host",
      "--ipc",
      "host",
      "--cgroupns",
      "host",
      "--security-opt",
      "seccomp=unconfined",
      "--security-opt",
      "apparmor=unconfined",
      "--mount",
      "type=bind,src=/proc,dst=/proc,readonly",
      "--mount",
      "type=bind,src=/sys/fs/cgroup,dst=/sys/fs/cgroup,readonly",
    );
  }
  args.push(
    "--volume",
    `${project}-huggingface-cache:/root/.cache/huggingface`,
    "--entrypoint",
    "hf",
    profile.vllmImage,
    "cache",
    "rm",
    `model/${profile.modelId}`,
    "--yes",
  );
  docker(args);
} else if (command === "start") {
  const profileKey = valueAfter("--profile") ?? process.env.MODEL_PROFILE ?? registry.defaultProfile;
  const replace = process.argv.includes("--replace");
  const profile = resolveModelProfile(profileKey, registry);
  if (hasContainer()) {
    if (!replace) {
      throw new Error(
        `${containerName} already exists. Pass --replace to switch profiles or run the stop command first.`,
      );
    }
    removeContainer();
  }

  const port = process.env.CHAT_PORT ?? "8000";
  const args = [
    "run",
    "--detach",
    "--name",
    containerName,
    "--label",
    `ai.labs.model-profile=${profile.key}`,
  ];
  if (nestedColab) {
    args.push(
      "--device",
      "nvidia.com/gpu=all",
      "--pid",
      "host",
      "--network",
      "host",
      "--ipc",
      "host",
      "--cgroupns",
      "host",
      "--security-opt",
      "seccomp=unconfined",
      "--security-opt",
      "apparmor=unconfined",
      "--mount",
      "type=bind,src=/proc,dst=/proc,readonly",
      "--mount",
      "type=bind,src=/sys/fs/cgroup,dst=/sys/fs/cgroup,readonly",
      "--env",
      "LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:/usr/local/nvidia/lib64",
    );
  } else {
    args.push("--gpus", "all", "--ipc", "host", "--publish", `${port}:8000`);
  }
  args.push(
    "--volume",
    `${project}-huggingface-cache:/root/.cache/huggingface`,
    "--volume",
    `${project}-vllm-cache:/root/.cache/vllm`,
    "--env",
    "VLLM_ENABLE_CUDA_COMPATIBILITY=1",
  );
  if (process.env.HF_TOKEN) {
    args.push("--env", `HF_TOKEN=${process.env.HF_TOKEN}`);
  }
  args.push(
    profile.vllmImage,
    "--model",
    profile.modelId,
    "--revision",
    profile.revision,
    "--served-model-name",
    profile.key,
    "--max-model-len",
    String(profile.maxModelLen),
    "--gpu-memory-utilization",
    String(profile.gpuMemoryUtilization),
    "--max-num-seqs",
    "8",
    ...(nestedColab ? ["--port", port] : []),
    ...profile.vllmArgs,
  );
  const id = docker(args, { quiet: true });
  persistModelProfile(profile.key);
  console.log(
    JSON.stringify(
      {
        containerId: id,
        containerName,
        profile: profile.key,
        modelId: profile.modelId,
        revision: profile.revision,
        image: profile.vllmImage,
        endpoint: `http://127.0.0.1:${port}/v1`,
      },
      null,
      2,
    ),
  );
} else {
  throw new Error(
    `Unknown command ${JSON.stringify(command)}. Use list, benchmark-list, start, stop, evict, or status.`,
  );
}
