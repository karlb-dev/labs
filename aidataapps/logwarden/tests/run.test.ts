import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FileTelemetryJournal } from "../src/telemetry.js";

describe("append-only telemetry journal", () => {
  it("continues sequence numbers after reopening", async () => {
    const directory = await mkdtemp(join(tmpdir(), "logwarden-telemetry-"));
    const path = join(directory, "telemetry.jsonl");
    const first = await FileTelemetryJournal.open(path, "run-1");
    const span = await first.startSpan("model.request", { episodeId: "episode-1" });
    await first.endSpan(span, "ok", { outputTokens: 12 });
    await first.flush();
    const second = await FileTelemetryJournal.open(path, "run-1");
    const record = await second.record("metric", "queue.depth", {}, { value: 3 });
    expect(record.sequence).toBe(3);
    const lines = (await readFile(path, "utf8")).trim().split("\n").map((line) => JSON.parse(line) as {
      sequence: number;
      previousRecordSha256: string | null;
      payloadSha256: string;
    });
    expect(lines.map((line) => line.sequence)).toEqual([1, 2, 3]);
    expect(lines.every((line) => /^[0-9a-f]{64}$/.test(line.payloadSha256))).toBe(true);
    expect(lines[0]?.previousRecordSha256).toBeNull();
    expect(lines[1]?.previousRecordSha256).toBe(lines[0]?.payloadSha256);
  });

  it("rejects a modified journal instead of silently resuming", async () => {
    const directory = await mkdtemp(join(tmpdir(), "logwarden-telemetry-tamper-"));
    const path = join(directory, "telemetry.jsonl");
    const journal = await FileTelemetryJournal.open(path, "run-2");
    await journal.record("point", "worker.started", {}, { worker: "one" });
    const content = await readFile(path, "utf8");
    await writeFile(path, content.replace('"worker":"one"', '"worker":"two"'));
    await expect(FileTelemetryJournal.open(path, "run-2")).rejects.toThrow("integrity check failed");
  });

  it("keeps parent span correlation on the closing event", async () => {
    const directory = await mkdtemp(join(tmpdir(), "logwarden-telemetry-parent-"));
    const path = join(directory, "telemetry.jsonl");
    const journal = await FileTelemetryJournal.open(path, "run-3");
    const span = await journal.startSpan("tool.execute", { traceId: "trace-1", parentSpanId: "parent-1" });
    const end = await journal.endSpan(span, "ok");
    expect(end.parentSpanId).toBe("parent-1");
  });
});
