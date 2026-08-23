import sql from "mssql";
import { canonicalJson, sha256 } from "./hash.js";
import { readAndValidateTelemetryJournal, type TelemetryRecord } from "./telemetry.js";

export interface TelemetryIngestionOptions {
  batchSize?: number;
  failAfterCommittedBatches?: number;
}

export interface TelemetryIngestionResult {
  journalPath: string;
  journalPathSha256: string;
  validatedRecords: number;
  insertedRecords: number;
  duplicateRecords: number;
  committedBatches: number;
  processEpochIds: string[];
  terminalRecordSha256: string | null;
}

export async function ingestTelemetryJournal(
  pool: sql.ConnectionPool,
  runId: string,
  journalPath: string,
  options: TelemetryIngestionOptions = {},
): Promise<TelemetryIngestionResult> {
  const batchSize = options.batchSize ?? 100;
  if (!Number.isSafeInteger(batchSize) || batchSize < 1 || batchSize > 10_000) {
    throw new Error(`Invalid telemetry ingestion batch size: ${batchSize}`);
  }
  const failAfter = options.failAfterCommittedBatches;
  if (failAfter !== undefined && (!Number.isSafeInteger(failAfter) || failAfter < 1)) {
    throw new Error(`Invalid injected failure batch count: ${failAfter}`);
  }
  const lines = await readAndValidateTelemetryJournal(journalPath, runId);
  const pathSha256 = sha256(journalPath);
  let inserted = 0;
  let committedBatches = 0;

  for (let offset = 0; offset < lines.length; offset += batchSize) {
    const batch = lines.slice(offset, offset + batchSize);
    const transaction = new sql.Transaction(pool);
    let batchInserted = 0;
    await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
    try {
      for (const line of batch) {
        const record = line.record;
        const exists = await new sql.Request(transaction)
          .input("event", sql.UniqueIdentifier, record.eventId)
          .query<{ present: number }>("SELECT CONVERT(int,CASE WHEN EXISTS(SELECT 1 FROM telemetry.journal_events WHERE event_id=@event) THEN 1 ELSE 0 END) AS present;");
        if (exists.recordset[0]?.present !== 1) {
          await insertJournalRecord(transaction, record);
          await projectSpan(transaction, record);
          batchInserted += 1;
        }
        await upsertCursor(transaction, runId, pathSha256, record, line.byteOffsetAfter);
      }
      await transaction.commit();
    } catch (error) {
      await transaction.rollback().catch(() => undefined);
      throw error;
    }
    inserted += batchInserted;
    committedBatches += 1;
    if (failAfter === committedBatches) {
      throw new Error(`INJECTED_TELEMETRY_INGESTION_CRASH_AFTER_BATCH_${committedBatches}`);
    }
  }

  return {
    journalPath,
    journalPathSha256: pathSha256,
    validatedRecords: lines.length,
    insertedRecords: inserted,
    duplicateRecords: lines.length - inserted,
    committedBatches,
    processEpochIds: [...new Set(lines.map(({ record }) => record.processEpochId))].sort(),
    terminalRecordSha256: lines.at(-1)?.record.payloadSha256 ?? null,
  };
}

async function insertJournalRecord(transaction: sql.Transaction, record: TelemetryRecord): Promise<void> {
  await new sql.Request(transaction)
    .input("run", sql.VarChar(120), record.runId)
    .input("epoch", sql.UniqueIdentifier, record.processEpochId)
    .input("sequence", sql.BigInt, record.sequence)
    .input("event", sql.UniqueIdentifier, record.eventId)
    .input("stream", sql.VarChar(40), semanticStream(record.name))
    .input("kind", sql.VarChar(32), record.eventKind)
    .input("name", sql.VarChar(160), record.name)
    .input("job", sql.BigInt, record.jobId ?? null)
    .input("episode", sql.VarChar(120), record.episodeId ?? null)
    .input("attempt", sql.Int, record.attemptId ?? null)
    .input("trace", sql.Char(32), record.traceId ?? null)
    .input("span", sql.Char(16), record.spanId ?? null)
    .input("parent", sql.Char(16), record.parentSpanId ?? null)
    .input("at", sql.DateTime2(7), new Date(record.atUtc))
    .input("monotonic", sql.Decimal(20, 3), record.monotonicMs)
    .input("duration", sql.Decimal(18, 3), record.durationMs ?? null)
    .input("status", sql.VarChar(40), record.status ?? null)
    .input("previous", sql.Char(64), record.previousRecordSha256)
    .input("hash", sql.Char(64), record.payloadSha256)
    .input("json", sql.NVarChar(sql.MAX), canonicalJson(record))
    .query(`
      INSERT telemetry.journal_events
        (run_id,process_epoch_id,sequence,event_id,semantic_stream,event_kind,event_name,
         job_id,episode_id,attempt_id,trace_id,span_id,parent_span_id,event_at_utc,
         monotonic_ms,duration_ms,status,previous_record_sha256,record_sha256,record_json)
      VALUES(@run,@epoch,@sequence,@event,@stream,@kind,@name,@job,@episode,@attempt,
        @trace,@span,@parent,@at,@monotonic,@duration,@status,@previous,@hash,@json);
    `);
}

async function upsertCursor(
  transaction: sql.Transaction,
  runId: string,
  pathSha256: string,
  record: TelemetryRecord,
  byteOffsetAfter: number,
): Promise<void> {
  await new sql.Request(transaction)
    .input("run", sql.VarChar(120), runId)
    .input("path", sql.Char(64), pathSha256)
    .input("epoch", sql.UniqueIdentifier, record.processEpochId)
    .input("sequence", sql.BigInt, record.sequence)
    .input("hash", sql.Char(64), record.payloadSha256)
    .input("offset", sql.BigInt, byteOffsetAfter)
    .query(`
      IF EXISTS (SELECT 1 FROM telemetry.journal_cursors WHERE run_id=@run AND journal_path_sha256=@path AND process_epoch_id=@epoch)
        UPDATE telemetry.journal_cursors
        SET committed_record_sha256=CASE WHEN committed_sequence<=@sequence THEN @hash ELSE committed_record_sha256 END,
            committed_sequence=CASE WHEN committed_sequence<@sequence THEN @sequence ELSE committed_sequence END,
            journal_byte_offset=CASE WHEN journal_byte_offset<@offset THEN @offset ELSE journal_byte_offset END,
            updated_at_utc=SYSUTCDATETIME()
        WHERE run_id=@run AND journal_path_sha256=@path AND process_epoch_id=@epoch;
      ELSE
        INSERT telemetry.journal_cursors
          (run_id,journal_path_sha256,process_epoch_id,committed_sequence,committed_record_sha256,journal_byte_offset)
        VALUES(@run,@path,@epoch,@sequence,@hash,@offset);
    `);
}

async function projectSpan(transaction: sql.Transaction, record: TelemetryRecord): Promise<void> {
  if (record.traceId === undefined || record.spanId === undefined || !record.eventKind.startsWith("span_")) return;
  const request = new sql.Request(transaction)
    .input("trace", sql.Char(32), record.traceId)
    .input("span", sql.Char(16), record.spanId)
    .input("parent", sql.Char(16), record.parentSpanId ?? null)
    .input("run", sql.VarChar(120), record.runId)
    .input("job", sql.BigInt, record.jobId ?? null)
    .input("episode", sql.VarChar(120), record.episodeId ?? null)
    .input("name", sql.VarChar(160), record.name)
    .input("at", sql.DateTime2(7), new Date(record.atUtc))
    .input("monotonic", sql.Decimal(20, 3), record.monotonicMs)
    .input("duration", sql.Decimal(18, 3), record.durationMs ?? null)
    .input("status", sql.VarChar(32), record.status ?? (record.eventKind === "span_start" ? "running" : "complete"))
    .input("json", sql.NVarChar(sql.MAX), canonicalJson(record.attributes))
    .input("error", sql.VarChar(80), typeof record.attributes.errorClass === "string" ? record.attributes.errorClass : null);
  if (record.eventKind === "span_start") {
    await request.query(`
      IF NOT EXISTS (SELECT 1 FROM telemetry.traces WHERE trace_id=@trace)
        INSERT telemetry.traces(trace_id,run_id,job_id,episode_id,trace_kind,started_at_utc,status,root_attributes_json)
        VALUES(@trace,@run,@job,@episode,'agent',@at,'running',@json);
      IF NOT EXISTS (SELECT 1 FROM telemetry.spans WHERE trace_id=@trace AND span_id=@span)
        INSERT telemetry.spans
          (trace_id,span_id,parent_span_id,span_name,span_kind,job_id,episode_id,
           started_at_utc,start_monotonic_ms,status,attributes_json,error_class)
        VALUES(@trace,@span,@parent,@name,'internal',@job,@episode,@at,@monotonic,'running',@json,@error);
    `);
  } else {
    await request.query(`
      UPDATE telemetry.spans
      SET finished_at_utc=@at,duration_ms=@duration,status=@status,
          attributes_json=@json,error_class=COALESCE(@error,error_class)
      WHERE trace_id=@trace AND span_id=@span;
      IF @parent IS NULL
        UPDATE telemetry.traces
        SET finished_at_utc=@at,duration_ms=@duration,status=@status
        WHERE trace_id=@trace;
    `);
  }
}

function semanticStream(name: string): string {
  if (/state|queue|lease|transition/.test(name)) return "state_transition";
  if (/model|inference|gateway|request|response/.test(name)) return "model";
  if (/tool|retrieval/.test(name)) return "tool";
  if (/valid|contract|parse|repair/.test(name)) return "validation";
  if (/decision|policy/.test(name)) return "decision";
  if (/metric|gpu|sql|host|vllm|sampler/.test(name)) return "system";
  if (/error|fail|timeout|oom/.test(name)) return "error";
  return "run";
}
