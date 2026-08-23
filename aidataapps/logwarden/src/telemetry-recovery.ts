import type { TelemetryRecord } from "./telemetry.js";

export function openTelemetrySpans(records: readonly TelemetryRecord[]): TelemetryRecord[] {
  const starts = new Map<string, TelemetryRecord>();
  const ended = new Set<string>();
  for (const record of records) {
    if (record.traceId === undefined || record.spanId === undefined) continue;
    const key = spanKey(record.traceId, record.spanId);
    if (record.eventKind === "span_start") starts.set(key, record);
    if (record.eventKind === "span_end") ended.add(key);
  }
  const open = [...starts].filter(([key]) => !ended.has(key)).map(([, record]) => record);
  return open.sort((left, right) => spanDepth(right, starts) - spanDepth(left, starts)
    || right.sequence - left.sequence);
}

function spanDepth(record: TelemetryRecord, starts: ReadonlyMap<string, TelemetryRecord>): number {
  let depth = 0;
  let current = record;
  const visited = new Set<string>();
  while (current.traceId !== undefined && current.parentSpanId !== undefined) {
    const key = spanKey(current.traceId, current.parentSpanId);
    if (visited.has(key)) throw new Error(`Telemetry span parent cycle: ${key}`);
    visited.add(key);
    const parent = starts.get(key);
    if (parent === undefined) break;
    depth += 1;
    current = parent;
  }
  return depth;
}

function spanKey(traceId: string, spanId: string): string {
  return `${traceId}:${spanId}`;
}
