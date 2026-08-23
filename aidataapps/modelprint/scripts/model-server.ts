import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, renameSync, writeFileSync, chmodSync, statfsSync } from "node:fs";
import dotenv from "dotenv";
import { hashJson } from "../src/hash.js";
import { loadModelRegistry, resolveModelProfile } from "../src/models.js";
import { valueAfter } from "../src/run.js";

dotenv.config({ quiet: true });
const project = process.env.COMPOSE_PROJECT_NAME ?? "aidataapps-modelprint";
const nested = process.env.CONTAINER_RUNTIME_PROFILE === "colab-rootless";
if (nested && !process.env.DOCKER_HOST) process.env.DOCKER_HOST = "unix:///run/user/1000/docker.sock";

function docker(args: string[], quiet = false): string {
  const result = spawnSync("docker", args, { encoding: "utf8", stdio: quiet ? "pipe" : ["inherit", "pipe", "pipe"] });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`docker ${args[0] ?? ""} failed: ${(result.stderr || result.stdout).trim()}`);
  return result.stdout.trim();
}
function hasContainer(): boolean { return spawnSync("docker", ["container", "inspect", containerName], { stdio: "ignore" }).status === 0; }
function currentProfile(): string | null { return hasContainer() ? docker(["inspect", "--format", '{{index .Config.Labels "ai.labs.model-profile"}}', containerName], true) || null : null; }
function removeContainer() { if (!hasContainer()) return; docker(["rm", "--force", containerName]); }
function persistProfile(key: string) {
  if (!existsSync(".env")) return;
  const current = readFileSync(".env", "utf8");
  const next = /^MODEL_PROFILE=/m.test(current) ? current.replace(/^MODEL_PROFILE=.*$/m, `MODEL_PROFILE=${key}`) : `${current.replace(/\n?$/, "\n")}MODEL_PROFILE=${key}\n`;
  const temporary = `.env.model.${process.pid}.tmp`; writeFileSync(temporary, next, { mode: 0o600 }); chmodSync(temporary, 0o600); renameSync(temporary, ".env");
}

const registry = loadModelRegistry();
const command = process.argv[2] ?? "list";
const selectedKey = valueAfter("--profile") ?? process.env.MODEL_PROFILE ?? registry.defaultProfile;
const containerName = process.env.CHAT_CONTAINER_NAME ?? `${project}-chat-${selectedKey.replace(/[^a-z0-9_.-]+/gi, "-")}`;
if (command === "list") {
  console.log("PROFILE\tFAMILY\tMODEL\tREVISION\tIMAGE");
  for (const [key, profile] of Object.entries(registry.profiles)) console.log(`${key}\t${profile.family}\t${profile.modelId}\t${profile.revision.slice(0, 12)}\t${profile.vllmImage}`);
} else if (command === "target-list") {
  console.log(registry.targetProfiles.join("\n"));
} else if (command === "status") {
  console.log(hasContainer() ? docker(["inspect", "--format", "{{.State.Status}} profile={{index .Config.Labels \"ai.labs.model-profile\"}} image={{.Image}}", containerName], true) : "stopped");
} else if (command === "stop") {
  removeContainer();
} else if (command === "evict") {
  const key = selectedKey;
  const profile = resolveModelProfile(key, registry);
  if (currentProfile() === key) removeContainer();
  const args = ["run", "--rm"];
  if (nested) args.push("--pid", "host", "--network", "host", "--ipc", "host", "--cgroupns", "host", "--security-opt", "seccomp=unconfined", "--security-opt", "apparmor=unconfined", "--mount", "type=bind,src=/proc,dst=/proc,readonly", "--mount", "type=bind,src=/sys/fs/cgroup,dst=/sys/fs/cgroup,readonly");
  args.push("--volume", `${process.env.SHARED_HF_VOLUME ?? "aidataapps-rag-huggingface-cache"}:/root/.cache/huggingface`, "--entrypoint", "hf", profile.vllmImage, "cache", "rm", `model/${profile.modelId}`, "--yes");
  docker(args);
} else if (command === "start") {
  const key = selectedKey;
  const profile = resolveModelProfile(key, registry);
  const effectiveGpuMemoryUtilization = Number(process.env.CHAT_GPU_MEMORY_UTILIZATION ?? profile.gpuMemoryUtilization);
  if (!Number.isFinite(effectiveGpuMemoryUtilization) || effectiveGpuMemoryUtilization <= 0 || effectiveGpuMemoryUtilization >= 1) {
    throw new Error(`CHAT_GPU_MEMORY_UTILIZATION must be between 0 and 1; received ${process.env.CHAT_GPU_MEMORY_UTILIZATION}`);
  }
  const profileHash = hashJson(profile);
  const runtimeProfileHash = hashJson({ profileHash, effectiveGpuMemoryUtilization });
  const snapshot = JSON.parse(readFileSync("data/manifests/model-registry-snapshot.json", "utf8")) as { profiles?: Record<string, { hf?: { weightBytes?: number } }> };
  const weightBytes = snapshot.profiles?.[key]?.hf?.weightBytes ?? 0;
  const volume = process.env.SHARED_HF_VOLUME ?? "aidataapps-rag-huggingface-cache";
  const mountpoint = docker(["volume", "inspect", volume, "--format", "{{.Mountpoint}}"], true);
  const cacheDirectory = `${mountpoint}/hub/models--${profile.modelId.replaceAll("/", "--")}`;
  const freeBytes = statfsSync(mountpoint).bavail * statfsSync(mountpoint).bsize;
  if (!existsSync(cacheDirectory) && weightBytes > 0 && freeBytes < weightBytes * 1.5) {
    throw new Error(`STOP_BUDGET: ${key} requires ${(weightBytes * 1.5 / 1e9).toFixed(1)} GB free before download; ${(freeBytes / 1e9).toFixed(1)} GB is available`);
  }
  if (hasContainer()) {
    if (!process.argv.includes("--replace")) throw new Error(`${containerName} exists; pass --replace`);
    removeContainer();
  }
  const port = process.env.CHAT_PORT ?? "8000";
  const args = ["run", "--detach", "--name", containerName,
    "--label", `ai.labs.model-profile=${profile.key}`,
    "--label", `ai.labs.model-profile-hash=${profileHash}`,
    "--label", `ai.labs.runtime-profile-hash=${runtimeProfileHash}`,
    "--label", `ai.labs.gpu-memory-utilization=${effectiveGpuMemoryUtilization}`];
  if (nested) args.push("--device", "nvidia.com/gpu=all", "--pid", "host", "--network", "host", "--ipc", "host", "--cgroupns", "host", "--security-opt", "seccomp=unconfined", "--security-opt", "apparmor=unconfined", "--mount", "type=bind,src=/proc,dst=/proc,readonly", "--mount", "type=bind,src=/sys/fs/cgroup,dst=/sys/fs/cgroup,readonly", "--env", "LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:/usr/local/nvidia/lib64");
  else args.push("--gpus", "all", "--ipc", "host", "--publish", `${port}:8000`);
  args.push("--volume", `${process.env.SHARED_HF_VOLUME ?? "aidataapps-rag-huggingface-cache"}:/root/.cache/huggingface`,
    "--volume", `${process.env.SHARED_VLLM_VOLUME ?? "aidataapps-rag-vllm-cache"}:/root/.cache/vllm`,
    "--env", "VLLM_ENABLE_CUDA_COMPATIBILITY=1");
  const batchInvariant = process.env.VLLM_BATCH_INVARIANT ?? (key === "muse-glimmer-30b" ? "1" : undefined);
  if (batchInvariant) args.push("--env", `VLLM_BATCH_INVARIANT=${batchInvariant}`, "--label", `ai.labs.batch-invariant=${batchInvariant}`);
  if (process.env.HF_TOKEN) args.push("--env", `HF_TOKEN=${process.env.HF_TOKEN}`);
  const serverArgs = ["--model", profile.modelId, "--revision", profile.revision, "--tokenizer-revision", profile.revision,
    "--served-model-name", profile.key,
    "--max-model-len", String(profile.maxModelLen), "--gpu-memory-utilization", String(effectiveGpuMemoryUtilization),
    "--max-num-seqs", String(profile.maxNumSeqs), "--enable-log-requests", ...(nested ? ["--port", port] : []), ...profile.campaignArgs];
  args.push(profile.vllmImage, ...serverArgs);
  const containerId = docker(args, true); persistProfile(key);
  console.log(JSON.stringify({ containerId, containerName, profile: key, profileHash, runtimeProfileHash, modelId: profile.modelId,
    revision: profile.revision, image: profile.vllmImage, endpoint: `http://127.0.0.1:${port}/v1`, cachedBeforeStart: existsSync(cacheDirectory),
    weightBytes, freeBytesBeforeStart: freeBytes, configuredGpuMemoryUtilization: profile.gpuMemoryUtilization,
    effectiveGpuMemoryUtilization, batchInvariant: batchInvariant ?? "0" }, null, 2));
} else throw new Error(`Unknown model command ${command}`);
