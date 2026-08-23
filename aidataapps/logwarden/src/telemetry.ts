import { randomUUID } from "node:crypto";
import { access, appendFile, mkdir, readdir, readFile } from "node:fs/promises";
import { dirname } from "node:path";
import { performance } from "node:perf_hooks";
import { canonicalJson, hashJson } from "./hash.js";

export interface TelemetryContext {
  runId: string;
  jobId?: number;
  episodeId?: string;
  attemptId?: number;
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
}

export interface TelemetryRecord extends TelemetryContext {
  schemaVersion: 1;
  processEpochId: string;
  sequence: number;
  eventId: string;
  eventKind: "point" | "span_start" | "span_end" | "metric" | "raw_snapshot";
  name: string;
  atUtc: string;
  monotonicMs: number;
  durationMs?: number;
  status?: string;
  attributes: Record<string, unknown>;
  previousRecordSha256: string | null;
  payloadSha256: string;
}

export interface StartedSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  startedMonotonicMs: number;
  context: TelemetryContext;
}

export interface ValidatedTelemetryLine {
  record: TelemetryRecord;
  byteOffsetAfter: number;
}

export async function readAndValidateTelemetryJournal(path: string, runId: string): Promise<ValidatedTelemetryLine[]> {
  const source = await readFile(path, "utf8");
  const output: ValidatedTelemetryLine[] = [];
  let previousRecordSha256: string | null = null;
  let byteOffset = 0;
  for (const [index, lineWithNewline] of source.match(/.*(?:\n|$)/g)?.filter((line) => line.length > 0).entries() ?? []) {
    const line = lineWithNewline.endsWith("\n") ? lineWithNewline.slice(0, -1) : lineWithNewline;
    byteOffset += Buffer.byteLength(lineWithNewline);
    if (line.trim().length === 0) continue;
    let record: TelemetryRecord;
    try {
      record = JSON.parse(line) as TelemetryRecord;
    } catch (error) {
      throw new Error(`Telemetry journal has invalid JSON at sequence ${index + 1}: ${path}`, { cause: error });
    }
    const { payloadSha256, ...payload } = record;
    const expectedSequence = output.length + 1;
    if (
      record.runId !== runId || record.sequence !== expectedSequence ||
      record.previousRecordSha256 !== previousRecordSha256 || payloadSha256 !== hashJson(payload)
    ) throw new Error(`Telemetry journal integrity check failed at sequence ${expectedSequence}: ${path}`);
    previousRecordSha256 = payloadSha256;
    output.push({ record, byteOffsetAfter: byteOffset });
  }
  return output;
}

export class FileTelemetryJournal {
  private sequence = 0;
  private previousRecordSha256: string | null = null;
  private writes: Promise<void> = Promise.resolve();
  private constructor(
    readonly path: string,
    readonly runId: string,
    readonly processEpochId: string,
  ) {}

  static async open(path: string, runId: string, processEpochId = randomUUID()): Promise<FileTelemetryJournal> {
    await mkdir(dirname(path), { recursive: true });
    const journal = new FileTelemetryJournal(path, runId, processEpochId);
    try {
      const lines = await readAndValidateTelemetryJournal(path, runId);
      journal.sequence = lines.length;
      journal.previousRecordSha256 = lines.at(-1)?.record.payloadSha256 ?? null;
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code !== "ENOENT") throw error;
    }
    return journal;
  }

  async record(
    eventKind: TelemetryRecord["eventKind"],
    name: string,
    context: Omit<TelemetryContext, "runId"> = {},
    attributes: Record<string, unknown> = {},
    fields: Pick<TelemetryRecord, "durationMs" | "status"> = {},
  ): Promise<TelemetryRecord> {
    const sequence = ++this.sequence;
    const base = {
      schemaVersion: 1 as const,
      processEpochId: this.processEpochId,
      sequence,
      eventId: randomUUID(),
      eventKind,
      name,
      atUtc: new Date().toISOString(),
      monotonicMs: Number(performance.now().toFixed(3)),
      runId: this.runId,
      ...defined(context),
      ...defined(fields),
      attributes,
      previousRecordSha256: this.previousRecordSha256,
    };
    const record: TelemetryRecord = { ...base, payloadSha256: hashJson(base) };
    this.previousRecordSha256 = record.payloadSha256;
    const line = `${canonicalJson(record)}\n`;
    this.writes = this.writes.then(async () => appendFile(this.path, line, { encoding: "utf8", flag: "a" }));
    await this.writes;
    return record;
  }

  async startSpan(
    name: string,
    context: Omit<TelemetryContext, "runId" | "traceId" | "spanId"> & { traceId?: string; parentSpanId?: string } = {},
    attributes: Record<string, unknown> = {},
  ): Promise<StartedSpan> {
    const traceId = context.traceId ?? randomUUID().replaceAll("-", "");
    const spanId = randomUUID().replaceAll("-", "").slice(0, 16);
    const startedMonotonicMs = performance.now();
    const cleanContext = defined({ ...context, traceId, spanId });
    await this.record("span_start", name, cleanContext, attributes);
    return {
      traceId,
      spanId,
      ...(context.parentSpanId === undefined ? {} : { parentSpanId: context.parentSpanId }),
      name,
      startedMonotonicMs,
      context: { runId: this.runId, ...cleanContext },
    };
  }

  async endSpan(span: StartedSpan, status: string, attributes: Record<string, unknown> = {}): Promise<TelemetryRecord> {
    return this.record(
      "span_end",
      span.name,
      {
        ...defined({
          jobId: span.context.jobId,
          episodeId: span.context.episodeId,
          attemptId: span.context.attemptId,
          parentSpanId: span.parentSpanId,
        }),
        traceId: span.traceId,
        spanId: span.spanId,
      },
      attributes,
      { durationMs: Number((performance.now() - span.startedMonotonicMs).toFixed(3)), status },
    );
  }

  async flush(): Promise<void> {
    await this.writes;
  }
}

export async function createComponentTelemetryJournal(
  runDirectory: string,
  runId: string,
  component: string,
  processEpochId = randomUUID(),
): Promise<{ journal: FileTelemetryJournal; path: string; processEpochId: string }> {
  const safeComponent = component.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  if (safeComponent.length === 0) throw new Error("Telemetry component must contain an alphanumeric character");
  const path = `${runDirectory}/telemetry/journals/${safeComponent}-${processEpochId}.jsonl`;
  return { journal: await FileTelemetryJournal.open(path, runId, processEpochId), path, processEpochId };
}

export async function discoverTelemetryJournalPaths(runDirectory: string): Promise<string[]> {
  const paths: string[] = [];
  const legacy = `${runDirectory}/telemetry/journal.jsonl`;
  if (await access(legacy).then(() => true).catch(() => false)) paths.push(legacy);
  const componentDirectory = `${runDirectory}/telemetry/journals`;
  const entries = await readdir(componentDirectory, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (entry.isFile() && entry.name.endsWith(".jsonl")) paths.push(`${componentDirectory}/${entry.name}`);
  }
  return paths.sort();
}

function defined<T extends object>(value: T): { [K in keyof T]?: Exclude<T[K], undefined> } {
  return Object.fromEntries(Object.entries(value).filter(([, child]) => child !== undefined)) as {
    [K in keyof T]?: Exclude<T[K], undefined>;
  };
}
