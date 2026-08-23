import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal } from "../src/telemetry-ingest.js";
import { discoverTelemetryJournalPaths } from "../src/telemetry.js";

interface RunManifest { runId: string }

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const explicitPaths = repeatedArguments("--journal").map((path) => resolve(path));
const journalPaths = explicitPaths.length > 0 ? explicitPaths : await discoverTelemetryJournalPaths(runDirectory);
const batchSize = integerArgument("--batch-size") ?? 100;
const failAfterCommittedBatches = integerArgument("--fail-after-batches");
if (failAfterCommittedBatches !== undefined && journalPaths.length !== 1) {
  throw new Error("Injected ingestion failure requires exactly one --journal path");
}

const pool = await connect(config.databases.lab, config.databases.controlName);
const results = [];
try {
  for (const journalPath of journalPaths) {
    results.push(await ingestTelemetryJournal(pool, run.runId, journalPath, {
      batchSize,
      ...(failAfterCommittedBatches === undefined ? {} : { failAfterCommittedBatches }),
    }));
  }
} finally {
  await pool.close();
}

const receiptBody = {
  schemaVersion: 2,
  runId: run.runId,
  journalCount: results.length,
  validatedRecords: results.reduce((sum, result) => sum + result.validatedRecords, 0),
  insertedRecords: results.reduce((sum, result) => sum + result.insertedRecords, 0),
  duplicateRecords: results.reduce((sum, result) => sum + result.duplicateRecords, 0),
  journals: results,
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/telemetry/journal-ingestion.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));

function repeatedArguments(name: string): string[] {
  const values: string[] = [];
  for (let index = 0; index < process.argv.length; index += 1) {
    if (process.argv[index] === name && process.argv[index + 1] !== undefined) values.push(process.argv[index + 1]!);
  }
  return values;
}

function integerArgument(name: string): number | undefined {
  const value = repeatedArguments(name).at(-1);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}
