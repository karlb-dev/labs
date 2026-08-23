import { openTelemetrySpans } from "../src/telemetry-recovery.js";
import type { TelemetryRecord } from "../src/telemetry.js";

describe("interrupted telemetry recovery", () => {
  it("orders open spans child-first while ignoring closed spans", () => {
    const records = [
      record(1, "span_start", "root", "r"),
      record(2, "span_start", "closed", "c", "r"),
      record(3, "span_end", "closed", "c", "r"),
      record(4, "span_start", "child", "a", "r"),
      record(5, "span_start", "grandchild", "b", "a"),
    ];
    expect(openTelemetrySpans(records).map((value) => value.spanId)).toEqual(["b", "a", "r"]);
  });
});

function record(
  sequence: number,
  eventKind: "span_start" | "span_end",
  name: string,
  spanId: string,
  parentSpanId?: string,
): TelemetryRecord {
  return {
    schemaVersion: 1,
    processEpochId: "00000000-0000-0000-0000-000000000001",
    sequence,
    eventId: `00000000-0000-0000-0000-${String(sequence).padStart(12, "0")}`,
    eventKind,
    name,
    atUtc: "2026-08-23T00:00:00.000Z",
    monotonicMs: sequence,
    runId: "run",
    traceId: "trace",
    spanId,
    ...(parentSpanId === undefined ? {} : { parentSpanId }),
    ...(eventKind === "span_end" ? { status: "ok" } : {}),
    attributes: {},
    previousRecordSha256: null,
    payloadSha256: String(sequence).padStart(64, "0"),
  };
}
