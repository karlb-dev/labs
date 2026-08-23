import { spawnSync } from "node:child_process";
import { chmodSync, existsSync, readFileSync, renameSync, statfsSync, writeFileSync } from "node:fs";
import {
  chatContainerName,
  chatDockerRunArguments,
  type ChatRuntimeIdentity,
} from "../src/chat-service.js";
import { hashJson } from "../src/hash.js";
import { loadModelRegistry, resolveModelProfile } from "../src/models.js";
import { atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";

const registry = loadModelRegistry();
const command = process.argv[2] ?? "list";
const selectedKey = valueAfter("--profile") ?? process.env.MODEL_PROFILE ?? registry.defaultProfile;
const project = process.env.COMPOSE_PROJECT_NAME ?? "aidataapps-logwarden";
const nested = process.env.CONTAINER_RUNTIME_PROFILE === "colab-rootless";
if (nested && !process.env.DOCKER_HOST) process.env.DOCKER_HOST = "unix:///run/user/1000/docker.sock";
const configuredUrlPort = new URL(process.env.CHAT_BASE_URL ?? "http://127.0.0.1:8010/v1").port;
const port = Number(process.env.CHAT_PORT ?? (configuredUrlPort === "" ? "8010" : configuredUrlPort));
const maxNumSeqs = Number(process.env.LOGWARDEN_CHAT_MAX_NUM_SEQS ?? "64");
const containerName = process.env.CHAT_CONTAINER_NAME ?? chatContainerName(project);
const runtime: ChatRuntimeIdentity = {
  project,
  containerName,
  port,
  nested,
  hfVolume: process.env.LOGWARDEN_HF_VOLUME ?? `${project}-huggingface-cache`,
  vllmVolume: process.env.LOGWARDEN_VLLM_VOLUME ?? `${project}-vllm-cache`,
  maxNumSeqs,
  batchInvariant: process.env.VLLM_BATCH_INVARIANT === "1" || selectedKey === "muse-glimmer-30b",
};

if (command === "list") {
  console.log("PROFILE\tFAMILY\tMODEL\tREVISION\tIMAGE");
  for (const [key, profile] of Object.entries(registry.profiles))
    console.log(`${key}\t${profile.family}\t${profile.modelId}\t${profile.revision.slice(0, 12)}\t${profile.vllmImage}`);
} else if (command === "target-list") {
  console.log(registry.targetProfiles.join("\n"));
} else if (command === "status") {
  console.log(hasContainer() ? docker([
    "inspect", "--format",
    "{{.State.Status}} profile={{index .Config.Labels \"ai.labs.model-profile\"}} image={{.Config.Image}} started={{.State.StartedAt}}",
    containerName,
  ], true) : "stopped");
} else if (command === "logs") {
  if (!hasContainer()) throw new Error(`${containerName} does not exist`);
  console.log(docker(["logs", "--tail", valueAfter("--tail") ?? "500", containerName], true));
} else if (command === "stop") {
  await stopAndRetain();
} else if (command === "evict") {
  if (hasContainer()) throw new Error("Stop the chat container before evicting its exact profile cache");
  const profile = resolveModelProfile(selectedKey, registry);
  const args = ["run", "--rm"];
  if (nested) args.push(
    "--pid", "host", "--network", "host", "--ipc", "host", "--cgroupns", "host",
    "--security-opt", "seccomp=unconfined", "--security-opt", "apparmor=unconfined",
    "--mount", "type=bind,src=/proc,dst=/proc,readonly",
    "--mount", "type=bind,src=/sys/fs/cgroup,dst=/sys/fs/cgroup,readonly",
  );
  args.push("--volume", `${runtime.hfVolume}:/root/.cache/huggingface`, "--entrypoint", "hf",
    profile.vllmImage, "cache", "rm", `model/${profile.modelId}`, "--yes");
  docker(args);
} else if (command === "start") {
  const profile = resolveModelProfile(selectedKey, registry);
  if (hasContainer()) {
    if (!process.argv.includes("--replace")) throw new Error(`${containerName} exists; pass --replace`);
    await stopAndRetain();
  }
  refuseOtherChatContainers();
  ensureDiskBudget(profile.modelId);
  const profileHash = hashJson(profile);
  const args = chatDockerRunArguments(profile, runtime, profileHash, Boolean(process.env.HF_TOKEN));
  const containerId = docker(args, true);
  persistProfile(selectedKey);
  const receipt = {
    schemaVersion: 1,
    action: "start",
    startedAtUtc: new Date().toISOString(),
    containerId,
    containerName,
    profile: selectedKey,
    profileHash,
    modelId: profile.modelId,
    revision: profile.revision,
    image: profile.vllmImage,
    endpoint: `http://127.0.0.1:${port}/v1`,
    runtime,
    serverArguments: args.slice(args.indexOf(profile.vllmImage) + 1),
  };
  await retainReceipt(`chat-service-start-${safeName(selectedKey)}.json`, receipt);
  console.log(JSON.stringify(receipt, null, 2));
} else {
  throw new Error(`Unknown model command ${command}`);
}

function docker(args: string[], quiet = false): string {
  const result = spawnSync("docker", args, {
    encoding: "utf8",
    stdio: quiet ? "pipe" : ["inherit", "pipe", "pipe"],
    maxBuffer: 32 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`docker ${args[0] ?? ""} failed: ${(result.stderr || result.stdout).trim()}`);
  return result.stdout.trim();
}

function hasContainer(): boolean {
  return spawnSync("docker", ["container", "inspect", containerName], { stdio: "ignore" }).status === 0;
}

function refuseOtherChatContainers(): void {
  const names = docker([
    "ps", "-a", "--filter", "label=ai.labs.lab=logwarden", "--filter", "label=ai.labs.role=chat", "--format", "{{.Names}}",
  ], true).split("\n").filter(Boolean);
  const other = names.filter((name) => name !== containerName);
  if (other.length > 0) throw new Error(`Unexpected LogWarden chat containers exist: ${other.join(",")}`);
}

function ensureDiskBudget(modelId: string): void {
  const mountpoint = docker(["volume", "inspect", runtime.hfVolume, "--format", "{{.Mountpoint}}"], true);
  if (!mountpoint.startsWith("/")) throw new Error("Hugging Face volume mountpoint is not absolute");
  const cacheDirectory = `${mountpoint}/hub/models--${modelId.replaceAll("/", "--")}`;
  if (existsSync(cacheDirectory)) return;
  const disk = statfsSync(mountpoint);
  const freeBytes = Number(disk.bavail) * Number(disk.bsize);
  const minimumBytes = Number(process.env.LOGWARDEN_CHAT_MIN_FREE_GIB ?? "90") * 1024 ** 3;
  if (freeBytes < minimumBytes)
    throw new Error(`STOP_BUDGET: uncached model requires at least ${(minimumBytes / 1024 ** 3).toFixed(1)} GiB free; ${(freeBytes / 1024 ** 3).toFixed(1)} GiB is available`);
}

async function stopAndRetain(): Promise<void> {
  if (!hasContainer()) return;
  const inspected = JSON.parse(docker(["inspect", containerName], true))[0] as {
    Id?: string;
    Image?: string;
    Name?: string;
    Config?: { Image?: string; Labels?: Record<string, string>; Cmd?: string[] };
    State?: Record<string, unknown>;
  };
  const profile = inspected.Config?.Labels?.["ai.labs.model-profile"] ?? "unknown";
  const retained = {
    schemaVersion: 1,
    action: "stop",
    stoppedAtUtc: new Date().toISOString(),
    container: {
      id: inspected.Id,
      imageId: inspected.Image,
      name: inspected.Name,
      configuredImage: inspected.Config?.Image,
      labels: inspected.Config?.Labels,
      command: inspected.Config?.Cmd,
      state: inspected.State,
    },
    logs: docker(["logs", "--timestamps", containerName], true),
  };
  await retainReceipt(`chat-service-stop-${safeName(profile)}.json`, retained);
  docker(["rm", "--force", containerName]);
}

function persistProfile(key: string): void {
  if (!existsSync(".env")) return;
  const current = readFileSync(".env", "utf8");
  const next = /^MODEL_PROFILE=/m.test(current)
    ? current.replace(/^MODEL_PROFILE=.*$/m, `MODEL_PROFILE=${key}`)
    : `${current.replace(/\n?$/, "\n")}MODEL_PROFILE=${key}\n`;
  const temporary = `.env.model.${process.pid}.tmp`;
  writeFileSync(temporary, next, { mode: 0o600 });
  chmodSync(temporary, 0o600);
  renameSync(temporary, ".env");
}

async function retainReceipt(name: string, value: unknown): Promise<void> {
  try {
    const runDirectory = resolveRunDirectory();
    await atomicWrite(`${runDirectory}/environment/${name}`, `${JSON.stringify(value, null, 2)}\n`, 0o600);
  } catch (error) {
    if (!/No run selected/.test((error as Error).message)) throw error;
  }
}

function safeName(value: string): string {
  return value.replace(/[^a-z0-9_.-]+/gi, "-").toLowerCase();
}
