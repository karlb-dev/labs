import { copyFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { hashFile } from "../src/hash.js";
import { loadModelRegistry } from "../src/models.js";

const source = fileURLToPath(new URL("../../modelprint/config/models.json", import.meta.url));
const target = fileURLToPath(new URL("../config/models.json", import.meta.url));
const sourceHash = await hashFile(source);
const targetHash = await hashFile(target);
if (sourceHash !== targetHash) {
  if (!process.argv.includes("--write")) {
    throw new Error(`Model registry drift: source=${sourceHash} target=${targetHash}; inspect then pass --write before freeze`);
  }
  await copyFile(source, target);
}
const registry = loadModelRegistry(target);
console.log(JSON.stringify({ source, target, sourceHash, targetHash: await hashFile(target), sourceCommit: registry.sourceCommit, targets: registry.targetProfiles }, null, 2));
