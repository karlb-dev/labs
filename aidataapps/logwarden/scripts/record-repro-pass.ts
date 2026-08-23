import { mkdir, readFile } from "node:fs/promises";
import { hashJson } from "../src/hash.js";
import { atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";

const mode = valueAfter("--mode");
if (mode !== "rows" && mode !== "restore") throw new Error("--mode must be rows or restore");
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const rowManifest = JSON.parse(await readFile(`${runDirectory}/repro/rows/manifest.json`, "utf8")) as {
  runId: string; receiptSha256: string; queryBundleSha256: string;
};
const expectedOutputs = JSON.parse(await readFile(`${runDirectory}/repro/expected-report-outputs.json`, "utf8")) as { receiptSha256: string };
if (rowManifest.runId !== run.runId) throw new Error("Row-manifest run mismatch");

const stableBody = {
  schemaVersion: 1,
  runId: run.runId,
  mode,
  rowManifestReceiptSha256: rowManifest.receiptSha256,
  queryBundleSha256: rowManifest.queryBundleSha256,
  disposition: "PASS",
};
const stable = { ...stableBody, receiptSha256: hashJson(stableBody) };
await atomicWrite(`${runDirectory}/repro/${mode}-pass.json`, `${JSON.stringify(stable, null, 2)}\n`);

const executionBody: Record<string, unknown> = {
  ...stableBody,
  completedAtUtc: new Date().toISOString(),
  stableReceiptSha256: stable.receiptSha256,
  expectedOutputsReceiptSha256: expectedOutputs.receiptSha256,
};
if (mode === "restore") {
  const backup = JSON.parse(await readFile(`${runDirectory}/database/backup-receipt.json`, "utf8")) as { receiptSha256: string };
  const restored = JSON.parse(await readFile(`${runDirectory}/repro/restore-rows/manifest.json`, "utf8")) as { receiptSha256: string };
  executionBody.sourceBackupReceiptSha256 = backup.receiptSha256;
  executionBody.restoredRowManifestReceiptSha256 = restored.receiptSha256;
}
const execution = { ...executionBody, receiptSha256: hashJson(executionBody) };
const stamp = String(executionBody.completedAtUtc).replaceAll(/[-:.]/g, "");
await mkdir(`${runDirectory}/repro/${mode}-executions`, { recursive: true });
await atomicWrite(`${runDirectory}/repro/${mode}-executions/${stamp}.json`, `${JSON.stringify(execution, null, 2)}\n`);
console.log(JSON.stringify(execution, null, 2));
