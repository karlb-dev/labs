import { readdir, readFile, stat } from "node:fs/promises";
import { relative, resolve } from "node:path";
import { canonicalJson, hashFile, hashJson } from "../src/hash.js";
import { resolveRunDirectory, valueAfter } from "../src/run.js";

interface Artifact {
  path: string;
  kind: "query" | "evidence";
  rows: number | null;
  bytes: number;
  sha256: string;
  querySha256?: string;
  sourcePath?: string;
}

interface Manifest {
  schemaVersion: number;
  runId: string;
  controlDatabaseRole: string;
  queryBundleSha256: string;
  artifacts: Artifact[];
  receiptSha256: string;
}

const runDirectory = resolveRunDirectory();
const expectedDirectory = resolve(valueAfter("--expected") ?? `${runDirectory}/repro/rows`);
const actualDirectory = resolve(valueAfter("--actual") ?? expectedDirectory);
for (const directory of [expectedDirectory, actualDirectory]) {
  if (!directory.startsWith(`${resolve(`${runDirectory}/repro`)}/`)) throw new Error(`Row bundle is outside run/repro: ${directory}`);
}

const expected = await verifyBundle(expectedDirectory);
const actual = actualDirectory === expectedDirectory ? expected : await verifyBundle(actualDirectory);
if (canonicalJson(actual) !== canonicalJson(expected)) throw new Error("Restored row manifest is not byte-identical to the retained row manifest");

const body = {
  schemaVersion: 1,
  runId: expected.runId,
  expectedDirectory: relative(runDirectory, expectedDirectory),
  actualDirectory: relative(runDirectory, actualDirectory),
  manifestReceiptSha256: expected.receiptSha256,
  queryBundleSha256: expected.queryBundleSha256,
  queryArtifacts: expected.artifacts.filter((item) => item.kind === "query").length,
  evidenceArtifacts: expected.artifacts.filter((item) => item.kind === "evidence").length,
  rows: expected.artifacts.reduce((total, item) => total + (item.rows ?? 0), 0),
  disposition: "PASS",
};
console.log(JSON.stringify({ ...body, receiptSha256: hashJson(body) }, null, 2));

async function verifyBundle(directory: string): Promise<Manifest> {
  const manifest = JSON.parse(await readFile(`${directory}/manifest.json`, "utf8")) as Manifest;
  const received = manifest.receiptSha256;
  const body = { ...manifest };
  delete (body as Partial<Manifest>).receiptSha256;
  if (hashJson(body) !== received) throw new Error(`Manifest receipt hash drift: ${directory}`);
  const declared = new Set(["manifest.json"]);
  for (const artifact of manifest.artifacts) {
    const path = resolve(directory, artifact.path);
    if (!path.startsWith(`${directory}/`) || relative(directory, path).startsWith("..")) throw new Error(`Unsafe artifact path: ${artifact.path}`);
    const information = await stat(path);
    if (information.size !== artifact.bytes || await hashFile(path) !== artifact.sha256) throw new Error(`Artifact drift: ${artifact.path}`);
    declared.add(artifact.path);
  }
  const observed = new Set(await walk(directory));
  if (canonicalJson([...observed].sort()) !== canonicalJson([...declared].sort())) {
    throw new Error(`Manifest/file inventory mismatch: ${directory}`);
  }
  return manifest;
}

async function walk(root: string, current = root): Promise<string[]> {
  const output: string[] = [];
  for (const entry of await readdir(current, { withFileTypes: true })) {
    const path = `${current}/${entry.name}`;
    if (entry.isDirectory()) output.push(...await walk(root, path));
    else if (entry.isFile()) output.push(relative(root, path));
    else throw new Error(`Unsupported row-bundle filesystem entry: ${path}`);
  }
  return output;
}
