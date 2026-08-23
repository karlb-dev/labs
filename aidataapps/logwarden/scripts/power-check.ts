import { execFile as execFileCallback } from "node:child_process";
import { readFile } from "node:fs/promises";
import { promisify } from "node:util";
import { hashJson } from "../src/hash.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";

const execFile = promisify(execFileCallback);
const runDirectory = resolveRunDirectory();
const outputPath = `${runDirectory}/metrics/power-analysis.json`;
const temporaryPath = `${runDirectory}/metrics/power-analysis.raw.json`;
const arguments_ = [
  "analysis/statistics.py",
  "--catalog", valueAfter("--catalog") ?? "config/scenarios/standard-v1.json",
  "--output", temporaryPath,
  "--seed", valueAfter("--seed") ?? "590003",
  "--simulations", valueAfter("--simulations") ?? "300",
  "--bootstrap-replicates", valueAfter("--bootstrap-replicates") ?? "1000",
  "--holm-family", valueAfter("--holm-family") ?? "10",
];
const devRows = valueAfter("--dev-rows");
if (devRows !== undefined) arguments_.push("--dev-rows", devRows);
const { stderr } = await execFile("python3", arguments_, { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
if (stderr.trim().length > 0) throw new Error(`Power analysis wrote stderr: ${stderr.trim()}`);
const result = JSON.parse(await readFile(temporaryPath, "utf8")) as Record<string, unknown>;
const body = {
  schemaVersion: 1,
  generatedAtUtc: new Date().toISOString(),
  implementation: "analysis/statistics.py",
  result,
};
const receipt = { ...body, receiptSha256: hashJson(body) };
await atomicWrite(outputPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
await appendExperimentLog(`Paired grouped power simulation completed with ${String(result.disposition)} disposition; receipt ${receipt.receiptSha256}.`);
console.log(JSON.stringify({ outputPath, disposition: result.disposition, receiptSha256: receipt.receiptSha256 }, null, 2));
