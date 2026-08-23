import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal, readAndValidateTelemetryJournal } from "../src/telemetry.js";

const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const config = loadConfig();
const created = await createComponentTelemetryJournal(runDirectory, run.runId, "crash-replay-gate");
const span = await created.journal.startSpan("gate.crash_replay", {}, { gate: "telemetry-ingestion-v1" });
for (let ordinal = 0; ordinal < 4; ordinal += 1) {
  await created.journal.record("point", "gate.crash_replay.fixture", { traceId: span.traceId, parentSpanId: span.spanId }, { ordinal });
}
await created.journal.endSpan(span, "success", { fixtureRecords: 4 });
await created.journal.flush();
const lines = await readAndValidateTelemetryJournal(created.path, run.runId);

const pool = await connect(config.databases.lab, config.databases.controlName);
let injectedFailure = "";
let recovery;
let idempotentReplay;
try {
  try {
    await ingestTelemetryJournal(pool, run.runId, created.path, { batchSize: 2, failAfterCommittedBatches: 1 });
    throw new Error("Injected ingestion crash did not fire");
  } catch (error) {
    injectedFailure = (error as Error).message;
    if (!injectedFailure.startsWith("INJECTED_TELEMETRY_INGESTION_CRASH_AFTER_BATCH_")) throw error;
  }
  recovery = await ingestTelemetryJournal(pool, run.runId, created.path, { batchSize: 2 });
  idempotentReplay = await ingestTelemetryJournal(pool, run.runId, created.path, { batchSize: 3 });
  const verification = await pool.request()
    .input("run", sql.VarChar(120), run.runId)
    .input("epoch", sql.UniqueIdentifier, created.processEpochId)
    .input("path", sql.Char(64), recovery.journalPathSha256)
    .input("trace", sql.Char(32), span.traceId)
    .query<{ rows_seen: number; unique_events: number; max_sequence: number; cursor_sequence: number; cursor_hash: string; open_spans: number; open_traces: number }>(`
      SELECT
        (SELECT COUNT(*) FROM telemetry.journal_events WHERE run_id=@run AND process_epoch_id=@epoch) AS rows_seen,
        (SELECT COUNT(DISTINCT event_id) FROM telemetry.journal_events WHERE run_id=@run AND process_epoch_id=@epoch) AS unique_events,
        (SELECT MAX(sequence) FROM telemetry.journal_events WHERE run_id=@run AND process_epoch_id=@epoch) AS max_sequence,
        (SELECT committed_sequence FROM telemetry.journal_cursors WHERE run_id=@run AND process_epoch_id=@epoch AND journal_path_sha256=@path) AS cursor_sequence,
        (SELECT committed_record_sha256 FROM telemetry.journal_cursors WHERE run_id=@run AND process_epoch_id=@epoch AND journal_path_sha256=@path) AS cursor_hash,
        (SELECT COUNT(*) FROM telemetry.spans WHERE trace_id=@trace AND finished_at_utc IS NULL) AS open_spans,
        (SELECT COUNT(*) FROM telemetry.traces WHERE trace_id=@trace AND finished_at_utc IS NULL) AS open_traces;
    `);
  const observed = verification.recordset[0]!;
  const expectedTerminalHash = lines.at(-1)!.record.payloadSha256;
  if (
    Number(observed.rows_seen) !== lines.length || Number(observed.unique_events) !== lines.length ||
    Number(observed.max_sequence) !== lines.length || Number(observed.cursor_sequence) !== lines.length ||
    observed.cursor_hash !== expectedTerminalHash || Number(observed.open_spans) !== 0 || Number(observed.open_traces) !== 0 ||
    idempotentReplay.insertedRecords !== 0 || idempotentReplay.duplicateRecords !== lines.length
  ) throw new Error(`Crash/replay reconciliation failed: ${JSON.stringify(observed)}`);

  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    journalPath: created.path,
    processEpochId: created.processEpochId,
    injectedFailure,
    fixtureRecords: lines.length,
    recovery,
    idempotentReplay,
    observed,
    expectedTerminalHash,
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(`${runDirectory}/telemetry/crash-replay-gate.json`, `${JSON.stringify(receipt, null, 2)}\n`);
  console.log(JSON.stringify(receipt, null, 2));
} finally {
  await pool.close();
}
