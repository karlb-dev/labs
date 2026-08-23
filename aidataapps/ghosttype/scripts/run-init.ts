import { execFileSync } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { relative } from "node:path";
import { hashJson } from "../src/hash.js";
import { LAB_ROOT, RUNS_ROOT, atomicWrite, ensureRunLayout, valueAfter } from "../src/run.js";

function diagnostic(command: string, args: string[]): string | null {
  try { return execFileSync(command, args, { encoding: "utf8", timeout: 30_000 }).trim(); } catch { return null; }
}

const campaign = valueAfter("--campaign") ?? "cpu-dev";
if (!new Set(["cpu-dev", "mac-quality", "mac-serving", "gpu-quality", "gpu-serving"]).has(campaign)) {
  throw new Error(`Invalid campaign ${campaign}`);
}
const branch = diagnostic("git", ["branch", "--show-current"]);
if (branch !== "aidataapps-ghosttype") throw new Error(`Refusing Lab 4 run init on branch ${JSON.stringify(branch)}`);

const startedAt = new Date();
const stamp = startedAt.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
const runId = `ghosttype-${campaign}-${stamp}`;
const runDirectory = `${RUNS_ROOT}/${runId}`;
await ensureRunLayout(runDirectory);

const environment = {
  schemaVersion: 1,
  runId,
  campaign,
  startedAt: startedAt.toISOString(),
  git: { commit: diagnostic("git", ["rev-parse", "HEAD"]), branch, status: diagnostic("git", ["status", "--short"]) },
  host: { platform: process.platform, arch: process.arch, node: process.version, uname: diagnostic("uname", ["-a"]) },
  emulatedSql: process.platform === "darwin",
};
const run = { ...environment, runManifestHash: hashJson(environment) };
await writeFile(`${runDirectory}/run.json`, `${JSON.stringify(run, null, 2)}\n`);
await atomicWrite(`${LAB_ROOT}/.current-run`, `${relative(LAB_ROOT, runDirectory)}\n`);
console.log(JSON.stringify({ runId, runDirectory, runManifestHash: run.runManifestHash }, null, 2));
