import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal } from "../src/telemetry-ingest.js";
import { readAndValidateTelemetryJournal } from "../src/telemetry.js";

interface SamplerOutput {
  sampledAtUtc: string;
  epochId: string;
  phase: string;
  receiptSha256: string;
}

const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const config = loadConfig();
const samplerScript = fileURLToPath(new URL("./telemetry-sampler.ts", import.meta.url));
const gateStartedAtUtc = new Date().toISOString();
const first = await runSampler(samplerScript);
const second = await runSampler(samplerScript);
if (first.epochId === second.epochId) throw new Error("Sampler restart reused its process epoch");

const pool = await connect(config.databases.lab, config.databases.controlName);
try {
  const epochs = [first.epochId, second.epochId];
  const journalResults = [];
  const replayResults = [];
  for (const epoch of epochs) {
    const path = `${runDirectory}/telemetry/journals/systems-sampler-${epoch}.jsonl`;
    const lines = await readAndValidateTelemetryJournal(path, run.runId);
    const rootStarts = lines.filter(({ record }) => record.eventKind === "span_start" && record.name === "sampler.sample");
    const rootEnds = lines.filter(({ record }) => record.eventKind === "span_end" && record.name === "sampler.sample");
    if (rootStarts.length !== 1 || rootEnds.length !== 1 || rootStarts[0]!.record.spanId !== rootEnds[0]!.record.spanId) {
      throw new Error(`Sampler journal did not close exactly one sample span: ${path}`);
    }
    journalResults.push(await ingestTelemetryJournal(pool, run.runId, path, { batchSize: 2 }));
    replayResults.push(await ingestTelemetryJournal(pool, run.runId, path, { batchSize: 3 }));
  }
  const observed = await pool.request()
    .input("run", sql.VarChar(120), run.runId)
    .input("epoch1", sql.UniqueIdentifier, first.epochId)
    .input("epoch2", sql.UniqueIdentifier, second.epochId)
    .input("started", sql.DateTime2(7), new Date(gateStartedAtUtc))
    .input("phase", sql.VarChar(40), "restart-gate")
    .query<{ queue_rows: number; queue_keys: number; queue_epochs: number; raw_snapshots: number; raw_snapshot_keys: number; closed_traces: number; open_traces: number }>(`
      SELECT
        (SELECT COUNT(*) FROM telemetry.queue_samples WHERE run_id=@run AND sampler_epoch_id IN (@epoch1,@epoch2)) AS queue_rows,
        (SELECT COUNT(DISTINCT sample_key) FROM telemetry.queue_samples WHERE run_id=@run AND sampler_epoch_id IN (@epoch1,@epoch2)) AS queue_keys,
        (SELECT COUNT(DISTINCT sampler_epoch_id) FROM telemetry.queue_samples WHERE run_id=@run AND sampler_epoch_id IN (@epoch1,@epoch2)) AS queue_epochs,
        (SELECT COUNT(*) FROM telemetry.raw_metric_snapshots WHERE run_id=@run AND phase=@phase AND observed_at_utc>=@started) AS raw_snapshots,
        (SELECT COUNT(DISTINCT snapshot_key) FROM telemetry.raw_metric_snapshots WHERE run_id=@run AND phase=@phase AND observed_at_utc>=@started) AS raw_snapshot_keys,
        (SELECT COUNT(*) FROM telemetry.traces WHERE run_id=@run AND started_at_utc>=@started AND finished_at_utc IS NOT NULL AND JSON_VALUE(root_attributes_json,'$.phase')=@phase) AS closed_traces,
        (SELECT COUNT(*) FROM telemetry.traces WHERE run_id=@run AND started_at_utc>=@started AND finished_at_utc IS NULL AND JSON_VALUE(root_attributes_json,'$.phase')=@phase) AS open_traces;
    `);
  const row = observed.recordset[0]!;
  if (
    Number(row.queue_rows) !== 2 || Number(row.queue_keys) !== 2 || Number(row.queue_epochs) !== 2 ||
    Number(row.raw_snapshots) !== 6 || Number(row.raw_snapshot_keys) !== 6 ||
    Number(row.closed_traces) !== 2 || Number(row.open_traces) !== 0 ||
    replayResults.some((result) => result.insertedRecords !== 0 || result.duplicateRecords !== result.validatedRecords)
  ) throw new Error(`Sampler restart reconciliation failed: ${JSON.stringify(row)}`);

  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    gateStartedAtUtc,
    samplerProcesses: [first, second],
    journalResults,
    replayResults,
    observed: row,
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(`${runDirectory}/telemetry/sampler-restart-gate.json`, `${JSON.stringify(receipt, null, 2)}\n`);
  console.log(JSON.stringify(receipt, null, 2));
} finally {
  await pool.close();
}

async function runSampler(script: string): Promise<SamplerOutput> {
  const stdout = await new Promise<string>((resolve, reject) => {
    execFile(process.execPath, ["--import", "tsx", script, "--phase", "restart-gate"], {
      cwd: fileURLToPath(new URL("../", import.meta.url)),
      env: process.env,
      encoding: "utf8",
      maxBuffer: 4 * 1024 * 1024,
      timeout: 60_000,
    }, (error, output, stderr) => error === null ? resolve(output) : reject(new Error(`Sampler child failed: ${stderr.trim() || error.message}`)));
  });
  return JSON.parse(stdout) as SamplerOutput;
}
