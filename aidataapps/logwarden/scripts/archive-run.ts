import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readdir, stat, writeFile } from "node:fs/promises";
import { basename, relative } from "node:path";
import { resolveRunDirectory } from "../src/run.js";

async function listFiles(root: string): Promise<string[]> {
  const output: string[] = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = `${root}/${entry.name}`;
    if (entry.isDirectory()) output.push(...await listFiles(path));
    else if (entry.isFile() && !["ARTIFACT_INVENTORY.json", "MIRROR_RECEIPT.json"].includes(entry.name)) output.push(path);
  }
  return output;
}

async function digest(path: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    const stream = createReadStream(path);
    stream.on("data", (value) => hash.update(value));
    stream.on("error", reject);
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

const runDirectory = resolveRunDirectory();
const runId = basename(runDirectory);
const paths = (await listFiles(runDirectory)).sort();
const artifacts = [];
for (const path of paths) {
  const info = await stat(path);
  const relativePath = relative(runDirectory, path);
  artifacts.push({ path: relativePath, kind: relativePath.split("/")[0] ?? "root", bytes: info.size, sha256: await digest(path) });
}
const inventory = { schemaVersion: 1, runId, createdAt: new Date().toISOString(), artifacts };
await writeFile(`${runDirectory}/ARTIFACT_INVENTORY.json`, `${JSON.stringify(inventory, null, 2)}\n`);
console.log(JSON.stringify({ runId, inventory: `${runDirectory}/ARTIFACT_INVENTORY.json`, artifacts: artifacts.length }, null, 2));
