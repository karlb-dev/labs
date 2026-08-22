import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { relative } from "node:path";
import { hashJson } from "../src/hash.js";
import { LAB_ROOT, RUNS_ROOT, ensureRunLayout, valueAfter } from "../src/run.js";

function diagnostic(command: string, args: string[]): string | null {
  try { return execFileSync(command, args, { encoding: "utf8", timeout: 30_000 }).trim(); } catch { return null; }
}

const campaign = valueAfter("--campaign") ?? "full";
if (!new Set(["smoke", "dev", "standard", "full"]).has(campaign)) throw new Error(`Invalid campaign ${campaign}`);
const startedAt = new Date();
const stamp = startedAt.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
const runId = `modelprint-${campaign}-${stamp}`;
const runDirectory = `${RUNS_ROOT}/${runId}`;
await ensureRunLayout(runDirectory);
const dockerHost = process.env.DOCKER_HOST ?? "unix:///run/user/1000/docker.sock";
const environment = {
  schemaVersion: 1,
  runId,
  campaign,
  startedAt: startedAt.toISOString(),
  status: "initialized",
  git: { commit: diagnostic("git", ["rev-parse", "HEAD"]), branch: diagnostic("git", ["branch", "--show-current"]), status: diagnostic("git", ["status", "--short"]) },
  node: { version: process.version, npm: diagnostic("npm", ["--version"]) },
  python: { version: diagnostic("python3", ["--version"]), accessPath: "pymssql-or-TypeScript-export" },
  gpu: diagnostic("nvidia-smi", ["--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"]),
  docker: { host: dockerHost, version: diagnostic("docker", ["--host", dockerHost, "version", "--format", "{{.Server.Version}}"]), containers: diagnostic("docker", ["--host", dockerHost, "ps", "--format", "{{json .}}"]), images: diagnostic("docker", ["--host", dockerHost, "image", "ls", "--digests", "--format", "{{json .}}"]), },
  disk: diagnostic("df", ["-h", "/content"]),
};
const run = { ...environment, runManifestHash: hashJson(environment) };
await Promise.all([
  writeFile(`${runDirectory}/run.json`, `${JSON.stringify(run, null, 2)}\n`),
  writeFile(`${runDirectory}/environment/git.json`, `${JSON.stringify(environment.git, null, 2)}\n`),
  writeFile(`${runDirectory}/environment/node.json`, `${JSON.stringify(environment.node, null, 2)}\n`),
  writeFile(`${runDirectory}/environment/python.json`, `${JSON.stringify(environment.python, null, 2)}\n`),
  writeFile(`${runDirectory}/environment/gpu.json`, `${JSON.stringify({ raw: environment.gpu }, null, 2)}\n`),
  writeFile(`${runDirectory}/environment/containers.json`, `${JSON.stringify(environment.docker, null, 2)}\n`),
  writeFile(`${runDirectory}/RESUME.md`, `# Resume ${runId}\n\n\`\`\`bash\ncd ${LAB_ROOT}\nexport MODELPRINT_RUN_DIR=${relative(LAB_ROOT, runDirectory)}\nnpm run doctor -- --run "$MODELPRINT_RUN_DIR"\n\`\`\`\n`),
  writeFile(`${LAB_ROOT}/.current-run`, `${relative(LAB_ROOT, runDirectory)}\n`),
]);
console.log(JSON.stringify({ runId, runDirectory }, null, 2));
