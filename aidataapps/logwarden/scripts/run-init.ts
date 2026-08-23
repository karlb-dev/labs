import { execFileSync } from "node:child_process";
import { copyFile, readFile, writeFile } from "node:fs/promises";
import { relative } from "node:path";
import { hashFile, hashJson } from "../src/hash.js";
import { FileTelemetryJournal } from "../src/telemetry.js";
import { LAB_ROOT, RUNS_ROOT, appendExperimentLog, atomicWrite, ensureRunLayout, valueAfter } from "../src/run.js";

function diagnostic(command: string, args: string[], timeout = 30_000): string | null {
  try {
    return execFileSync(command, args, { encoding: "utf8", timeout }).trim();
  } catch {
    return null;
  }
}

const campaign = valueAfter("--campaign") ?? "smoke";
if (!new Set(["smoke", "dev", "standard", "full"]).has(campaign)) {
  throw new Error(`Invalid campaign ${campaign}`);
}
const branch = diagnostic("git", ["branch", "--show-current"]);
if (branch !== "aidataapps-logwarden") {
  throw new Error(`Refusing Lab 3 run initialization on branch ${JSON.stringify(branch)}`);
}
const predecessorDiff = diagnostic("git", [
  "diff",
  "--name-only",
  "origin/aidataapps-modelprint...HEAD",
  "--",
  "aidataapps/rag",
  "aidataapps/modelprint",
  "interpretability",
]);
if (predecessorDiff) throw new Error(`Read-only predecessor paths changed:\n${predecessorDiff}`);

const startedAt = new Date();
const stamp = startedAt.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
const runId = `logwarden-${campaign}-${stamp}`;
const runDirectory = `${RUNS_ROOT}/${runId}`;
await ensureRunLayout(runDirectory);

const dockerHost = process.env.DOCKER_HOST ?? "unix:///run/user/1000/docker.sock";
const inputs = {
  specSha256: await hashFile(`${LAB_ROOT}/docs/SPEC.md`),
  addendumSha256: await hashFile(`${LAB_ROOT}/docs/SPEC_ADDENDUM.md`),
  modelRegistrySha256: await hashFile(`${LAB_ROOT}/config/models.json`),
};
const environment = {
  schemaVersion: 1,
  runId,
  campaign,
  startedAt: startedAt.toISOString(),
  status: "initialized",
  inputs,
  git: {
    commit: diagnostic("git", ["rev-parse", "HEAD"]),
    branch,
    upstream: diagnostic("git", ["rev-parse", "--abbrev-ref", "@{upstream}"]),
    baseCommit: "88ea443092cd27226272776e6c6fcf8fbce329de",
    status: diagnostic("git", ["status", "--short"]),
    predecessorDiff: predecessorDiff ?? "",
  },
  node: { version: process.version, npm: diagnostic("npm", ["--version"]) },
  python: { version: diagnostic("python3", ["--version"]) },
  gpu: {
    inventory: diagnostic("nvidia-smi", ["--query-gpu=name,uuid,driver_version,memory.total,memory.free", "--format=csv,noheader"]),
    processes: diagnostic("nvidia-smi", ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
  },
  docker: {
    host: dockerHost,
    version: diagnostic("docker", ["--host", dockerHost, "version", "--format", "{{.Server.Version}}"]),
    containers: diagnostic("docker", ["--host", dockerHost, "ps", "--format", "{{json .}}"]),
    images: diagnostic("docker", ["--host", dockerHost, "image", "ls", "--digests", "--format", "{{json .}}"]),
  },
  disk: {
    local: diagnostic("df", ["-h", "/content"]),
    drive: diagnostic("df", ["-h", "/content/drive/MyDrive"]),
  },
};
const run = { ...environment, runManifestHash: hashJson(environment) };

await Promise.all([
  writeFile(`${runDirectory}/run.json`, `${JSON.stringify(run, null, 2)}\n`),
  writeFile(`${runDirectory}/environment/git.json`, `${JSON.stringify(environment.git, null, 2)}\n`),
  writeFile(`${runDirectory}/environment/node.json`, `${JSON.stringify(environment.node, null, 2)}\n`),
  writeFile(`${runDirectory}/environment/python.json`, `${JSON.stringify(environment.python, null, 2)}\n`),
  writeFile(`${runDirectory}/environment/gpu.json`, `${JSON.stringify(environment.gpu, null, 2)}\n`),
  writeFile(`${runDirectory}/environment/containers.json`, `${JSON.stringify(environment.docker, null, 2)}\n`),
  copyFile(`${LAB_ROOT}/config/models.json`, `${runDirectory}/manifests/models.json`),
  copyFile(`${LAB_ROOT}/EXPERIMENT_LOG.md`, `${runDirectory}/EXPERIMENT_LOG.md`),
  writeFile(
    `${runDirectory}/RESUME.md`,
    [
      `# Resume ${runId}`,
      "",
      "```bash",
      `cd ${LAB_ROOT}`,
      "source scripts/runtime-env.sh",
      `export LOGWARDEN_RUN_DIR=${relative(LAB_ROOT, runDirectory)}`,
      "npm run db:setup",
      "npm run doctor",
      "npm run check",
      "```",
      "",
      "No scientific campaign is frozen at this foundation checkpoint.",
      "",
    ].join("\n"),
  ),
]);
await atomicWrite(`${LAB_ROOT}/.current-run`, `${relative(LAB_ROOT, runDirectory)}\n`);
const journal = await FileTelemetryJournal.open(`${runDirectory}/telemetry/journal.jsonl`, runId);
await journal.record("point", "run.initialized", {}, { campaign, runManifestHash: run.runManifestHash, ...inputs });
await journal.flush();
await appendExperimentLog(`LW-0 run initialized: ${runId}; manifest=${run.runManifestHash}.`);

console.log(JSON.stringify({ runId, runDirectory, runManifestHash: run.runManifestHash, inputs }, null, 2));
