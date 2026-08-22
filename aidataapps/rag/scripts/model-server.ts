import { spawnSync } from "node:child_process";
import dotenv from "dotenv";
import { loadModelRegistry, resolveModelProfile } from "../src/models.js";

dotenv.config({ quiet: true });

const project = process.env.COMPOSE_PROJECT_NAME ?? "aidataapps-rag";
const containerName = `${project}-chat`;

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

function removeContainer(): void {
  if (!hasContainer()) return;
  docker(["stop", "--time", "20", containerName]);
  docker(["rm", containerName]);
}

function valueAfter(flag: string): string | undefined {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] : undefined;
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
    "--gpus",
    "all",
    "--ipc",
    "host",
    "--publish",
    `${port}:8000`,
    "--volume",
    `${project}-huggingface-cache:/root/.cache/huggingface`,
    "--volume",
    `${project}-vllm-cache:/root/.cache/vllm`,
    "--env",
    "VLLM_ENABLE_CUDA_COMPATIBILITY=1",
  ];
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
    ...profile.vllmArgs,
  );
  const id = docker(args, { quiet: true });
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
    `Unknown command ${JSON.stringify(command)}. Use list, benchmark-list, start, stop, or status.`,
  );
}
