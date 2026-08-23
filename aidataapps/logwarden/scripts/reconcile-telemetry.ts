import { readFile } from "node:fs/promises";
import { relative, resolve } from "node:path";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashFile, hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { discoverTelemetryJournalPaths, readAndValidateTelemetryJournal, type TelemetryRecord } from "../src/telemetry.js";

interface LocalRecord {
  journalPath: string;
  journalPathSha256: string;
  relativePath: string;
  byteOffsetAfter: number;
  record: TelemetryRecord;
}

const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const paths = await discoverTelemetryJournalPaths(runDirectory);
const local: LocalRecord[] = [];
const journalSummaries = [];
for (const path of paths) {
  const lines = await readAndValidateTelemetryJournal(path, run.runId);
  const relativePath = relative(runDirectory, path);
  const pathSha256 = sha256(path);
  const fileSha256 = await hashFile(path);
  local.push(...lines.map((line) => ({ journalPath: path, journalPathSha256: pathSha256, relativePath, ...line })));
  journalSummaries.push({
    relativePath,
    journalPathSha256: pathSha256,
    fileSha256,
    recordCount: lines.length,
    processEpochIds: [...new Set(lines.map(({ record }) => record.processEpochId))].sort(),
    terminalRecordSha256: lines.at(-1)?.record.payloadSha256 ?? null,
  });
}
assertUnique(local.map(({ record }) => record.eventId), "local event ID");
assertUnique(local.map(({ record }) => record.payloadSha256), "local record hash");

const rawArtifactChecks = [];
for (const { record } of local.filter(({ record }) => record.name === "model.request.durable" || record.name === "model.response.durable")) {
  const path = record.attributes.path;
  const expectedHash = record.attributes.sha256;
  const expectedBytes = record.attributes.bytes;
  if (typeof path !== "string" || typeof expectedHash !== "string" || typeof expectedBytes !== "number") {
    throw new Error(`Raw artifact telemetry lacks path/hash/bytes: ${record.eventId}`);
  }
  const absolute = resolve(path);
  const rawRelative = relative(runDirectory, absolute);
  if (rawRelative.startsWith("..") || rawRelative.startsWith("/")) throw new Error(`Raw artifact escapes run directory: ${absolute}`);
  const body = await readFile(absolute);
  const observedHash = sha256(body);
  if (body.length !== expectedBytes || observedHash !== expectedHash) throw new Error(`Raw artifact mismatch: ${rawRelative}`);
  rawArtifactChecks.push({ relativePath: rawRelative, bytes: body.length, sha256: observedHash });
}

const expectedSpanStarts = new Map<string, LocalRecord>();
const expectedSpanEnds = new Map<string, LocalRecord>();
for (const record of local) {
  if (record.record.traceId === undefined || record.record.spanId === undefined) continue;
  const key = `${record.record.traceId}:${record.record.spanId}`;
  if (record.record.eventKind === "span_start") {
    if (expectedSpanStarts.has(key)) throw new Error(`Duplicate local span start: ${key}`);
    expectedSpanStarts.set(key, record);
  }
  if (record.record.eventKind === "span_end") {
    if (expectedSpanEnds.has(key)) throw new Error(`Duplicate local span end: ${key}`);
    expectedSpanEnds.set(key, record);
  }
}
for (const key of expectedSpanStarts.keys()) if (!expectedSpanEnds.has(key)) throw new Error(`Unclosed local span: ${key}`);
for (const key of expectedSpanEnds.keys()) if (!expectedSpanStarts.has(key)) throw new Error(`Span end lacks local start: ${key}`);

const config = loadConfig();
const pool = await connect(config.databases.lab, config.databases.controlName);
try {
  const sqlEvents = await pool.request()
    .input("run", sql.VarChar(120), run.runId)
    .query<{ event_id: string; process_epoch_id: string; sequence: string; record_sha256: string }>(`
      SELECT CONVERT(varchar(36),event_id) AS event_id,
             CONVERT(varchar(36),process_epoch_id) AS process_epoch_id,
             CONVERT(varchar(30),sequence) AS sequence,record_sha256
      FROM telemetry.journal_events WHERE run_id=@run;
    `);
  const localByEvent = new Map(local.map((entry) => [entry.record.eventId.toLowerCase(), entry]));
  const sqlByEvent = new Map(sqlEvents.recordset.map((entry) => [entry.event_id.toLowerCase(), entry]));
  const missingSqlEvents = [...localByEvent.keys()].filter((key) => !sqlByEvent.has(key));
  const extraSqlEvents = [...sqlByEvent.keys()].filter((key) => !localByEvent.has(key));
  const mismatchedSqlEvents = [...localByEvent].filter(([key, entry]) => {
    const observed = sqlByEvent.get(key);
    return observed !== undefined && (observed.record_sha256 !== entry.record.payloadSha256 || observed.process_epoch_id.toLowerCase() !== entry.record.processEpochId.toLowerCase() || Number(observed.sequence) !== entry.record.sequence);
  }).map(([key]) => key);
  if (missingSqlEvents.length > 0 || extraSqlEvents.length > 0 || mismatchedSqlEvents.length > 0) {
    throw new Error(`Journal/SQL event mismatch: ${JSON.stringify({ missingSqlEvents, extraSqlEvents, mismatchedSqlEvents })}`);
  }

  const cursors = await pool.request()
    .input("run", sql.VarChar(120), run.runId)
    .query<{ journal_path_sha256: string; process_epoch_id: string; committed_sequence: string; committed_record_sha256: string; journal_byte_offset: string }>(`
      SELECT journal_path_sha256,CONVERT(varchar(36),process_epoch_id) AS process_epoch_id,
             CONVERT(varchar(30),committed_sequence) AS committed_sequence,
             committed_record_sha256,CONVERT(varchar(30),journal_byte_offset) AS journal_byte_offset
      FROM telemetry.journal_cursors WHERE run_id=@run;
    `);
  const cursorMap = new Map(cursors.recordset.map((cursor) => [`${cursor.journal_path_sha256}:${cursor.process_epoch_id.toLowerCase()}`, cursor]));
  const expectedCursorGroups = new Map<string, LocalRecord[]>();
  for (const entry of local) {
    const key = `${entry.journalPathSha256}:${entry.record.processEpochId.toLowerCase()}`;
    expectedCursorGroups.set(key, [...(expectedCursorGroups.get(key) ?? []), entry]);
  }
  for (const [key, entries] of expectedCursorGroups) {
    const terminal = [...entries].sort((left, right) => left.record.sequence - right.record.sequence).at(-1)!;
    const cursor = cursorMap.get(key);
    if (cursor === undefined || Number(cursor.committed_sequence) !== terminal.record.sequence || cursor.committed_record_sha256 !== terminal.record.payloadSha256 || Number(cursor.journal_byte_offset) !== terminal.byteOffsetAfter) {
      throw new Error(`Journal cursor mismatch: ${key}`);
    }
  }
  const extraCursors = [...cursorMap.keys()].filter((key) => !expectedCursorGroups.has(key));
  if (extraCursors.length > 0) throw new Error(`SQL has cursors without retained journals: ${extraCursors.join(",")}`);

  const observed = await pool.request()
    .input("run", sql.VarChar(120), run.runId)
    .query<Record<string, number | string | null>>(`
      SELECT
        (SELECT COUNT(*) FROM telemetry.traces WHERE run_id=@run) AS trace_count,
        (SELECT COUNT(*) FROM telemetry.traces WHERE run_id=@run AND finished_at_utc IS NULL) AS open_trace_count,
        (SELECT COUNT(*) FROM telemetry.spans AS s INNER JOIN telemetry.traces AS t ON t.trace_id=s.trace_id WHERE t.run_id=@run) AS span_count,
        (SELECT COUNT(*) FROM telemetry.spans AS s INNER JOIN telemetry.traces AS t ON t.trace_id=s.trace_id WHERE t.run_id=@run AND s.finished_at_utc IS NULL) AS open_span_count,
        (SELECT COUNT(*) FROM control.jobs WHERE run_kind='fake_gateway_gate') AS gate_job_count,
        (SELECT COUNT(*) FROM control.jobs WHERE run_kind='fake_gateway_gate' AND status IN ('pending','running')) AS active_gate_jobs,
        (SELECT COUNT(*) FROM ops.work_items AS w INNER JOIN control.jobs AS j ON j.job_id=w.job_id WHERE j.run_kind='fake_gateway_gate' AND w.status IN ('pending','leased','packet_loaded','model_requested','tool_requested','tool_completed','decision_received','validated','persisted','proposed_action_recorded')) AS active_gate_work_items,
        (SELECT COUNT(*) FROM agent.agent_runs WHERE run_id=@run) AS agent_run_count,
        (SELECT COUNT(*) FROM agent.agent_runs WHERE run_id=@run AND finished_at_utc IS NULL) AS open_agent_runs,
        (SELECT COUNT(*) FROM agent.model_requests AS q INNER JOIN agent.turns AS t ON t.turn_id=q.turn_id INNER JOIN agent.agent_runs AS r ON r.agent_run_id=t.agent_run_id WHERE r.run_id=@run) AS model_request_count,
        (SELECT COUNT(*) FROM agent.model_responses AS p INNER JOIN agent.model_requests AS q ON q.model_request_id=p.model_request_id INNER JOIN agent.turns AS t ON t.turn_id=q.turn_id INNER JOIN agent.agent_runs AS r ON r.agent_run_id=t.agent_run_id WHERE r.run_id=@run) AS model_response_count,
        (SELECT COUNT(*) FROM agent.model_requests AS q INNER JOIN agent.turns AS t ON t.turn_id=q.turn_id INNER JOIN agent.agent_runs AS r ON r.agent_run_id=t.agent_run_id WHERE r.run_id=@run AND LOWER(CONVERT(varchar(64),HASHBYTES('SHA2_256',q.request_body),2))<>q.request_body_sha256) AS bad_request_hashes,
        (SELECT COUNT(*) FROM agent.model_responses AS p INNER JOIN agent.model_requests AS q ON q.model_request_id=p.model_request_id INNER JOIN agent.turns AS t ON t.turn_id=q.turn_id INNER JOIN agent.agent_runs AS r ON r.agent_run_id=t.agent_run_id WHERE r.run_id=@run AND LOWER(CONVERT(varchar(64),HASHBYTES('SHA2_256',p.response_body),2))<>p.response_body_sha256) AS bad_response_hashes,
        (SELECT COUNT(*) FROM telemetry.metric_samples) AS metric_sample_count,
        (SELECT COUNT(DISTINCT sample_key) FROM telemetry.metric_samples) AS metric_sample_keys,
        (SELECT COUNT(*) FROM telemetry.raw_metric_snapshots WHERE run_id=@run) AS raw_metric_snapshot_count,
        (SELECT COUNT(DISTINCT snapshot_key) FROM telemetry.raw_metric_snapshots WHERE run_id=@run) AS raw_metric_snapshot_keys;
    `);
  const row = observed.recordset[0]!;
  const expectedTraceIds = new Set([...expectedSpanStarts.values()].filter(({ record }) => record.parentSpanId === undefined).map(({ record }) => record.traceId));
  if (
    Number(row.trace_count) !== expectedTraceIds.size || Number(row.span_count) !== expectedSpanStarts.size ||
    Number(row.open_trace_count) !== 0 || Number(row.open_span_count) !== 0 ||
    Number(row.active_gate_jobs) !== 0 || Number(row.active_gate_work_items) !== 0 || Number(row.open_agent_runs) !== 0 ||
    Number(row.model_request_count) !== Number(row.model_response_count) ||
    Number(row.bad_request_hashes) !== 0 || Number(row.bad_response_hashes) !== 0 ||
    Number(row.metric_sample_count) !== Number(row.metric_sample_keys) ||
    Number(row.raw_metric_snapshot_count) !== Number(row.raw_metric_snapshot_keys)
  ) throw new Error(`Global telemetry reconciliation failed: ${JSON.stringify(row)}`);

  const inputSet = {
    runId: run.runId,
    journals: journalSummaries,
    sqlJournalEventCount: sqlEvents.recordset.length,
    cursorCount: cursors.recordset.length,
    rawArtifacts: rawArtifactChecks,
  };
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    inputSetSha256: hashJson(inputSet),
    journals: journalSummaries,
    journalRecords: local.length,
    rawArtifacts: rawArtifactChecks,
    expectedTraceCount: expectedTraceIds.size,
    expectedSpanCount: expectedSpanStarts.size,
    observed: row,
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(`${runDirectory}/telemetry/reconciliation.json`, `${JSON.stringify(receipt, null, 2)}\n`);
  console.log(JSON.stringify({ runId: run.runId, journalCount: paths.length, journalRecords: local.length, rawArtifacts: rawArtifactChecks.length, disposition: "PASS", inputSetSha256: receipt.inputSetSha256, receiptSha256: receipt.receiptSha256 }, null, 2));
} finally {
  await pool.close();
}

function assertUnique(values: string[], kind: string): void {
  const seen = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) throw new Error(`Duplicate ${kind}: ${value}`);
    seen.add(value);
  }
}
