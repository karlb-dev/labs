import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import { relative, resolve } from "node:path";
import { hashJson, sha256 } from "../src/hash.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { FileTelemetryJournal, readAndValidateTelemetryJournal } from "../src/telemetry.js";
import { openTelemetrySpans } from "../src/telemetry-recovery.js";

const runDirectory = resolve(resolveRunDirectory());
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const source = argument("--journal");
const reason = argument("--reason");
if (source === undefined || reason === undefined || reason.trim().length === 0) {
  throw new Error("Usage: telemetry:recover -- --journal <path> --reason <text>");
}
const journalPath = resolve(source);
const journalRelativePath = relative(runDirectory, journalPath);
if (journalRelativePath.startsWith("..") || journalRelativePath.startsWith("/")) {
  throw new Error(`Telemetry journal escapes the active run: ${journalPath}`);
}

const before = await readAndValidateTelemetryJournal(journalPath, run.runId);
const open = openTelemetrySpans(before.map((line) => line.record));
if (open.length === 0) throw new Error(`Telemetry journal has no interrupted open spans: ${journalPath}`);
const recoveryProcessEpochId = randomUUID();
const recoveredAtUtc = new Date().toISOString();
const journal = await FileTelemetryJournal.open(journalPath, run.runId, recoveryProcessEpochId);
const root = open.at(-1)!;
await journal.record("point", "state.interrupted_journal_recovery", {
  traceId: root.traceId!,
  spanId: root.spanId!,
}, {
  journalRelativePath,
  reason,
  recoveryProcessEpochId,
  recoveredAtUtc,
  openSpanCount: open.length,
});
for (const span of open) {
  const durationMs = Math.max(0, new Date(recoveredAtUtc).getTime() - new Date(span.atUtc).getTime());
  await journal.record("span_end", span.name, {
    ...(span.jobId === undefined ? {} : { jobId: span.jobId }),
    ...(span.episodeId === undefined ? {} : { episodeId: span.episodeId }),
    ...(span.attemptId === undefined ? {} : { attemptId: span.attemptId }),
    traceId: span.traceId!,
    spanId: span.spanId!,
    ...(span.parentSpanId === undefined ? {} : { parentSpanId: span.parentSpanId }),
  }, {
    interruptionReason: reason,
    recoveredAtUtc,
    recoveryProcessEpochId,
    originalProcessEpochId: span.processEpochId,
  }, { durationMs, status: "interrupted" });
}
await journal.flush();

const after = await readAndValidateTelemetryJournal(journalPath, run.runId);
const remaining = openTelemetrySpans(after.map((line) => line.record));
if (remaining.length !== 0) throw new Error(`Telemetry recovery left ${remaining.length} open spans`);
const body = {
  schemaVersion: 1,
  runId: run.runId,
  journalRelativePath,
  journalPathSha256: sha256(journalPath),
  reason,
  recoveredAtUtc,
  recoveryProcessEpochId,
  recordsBefore: before.length,
  recordsAfter: after.length,
  recoveredSpans: open.map((span) => ({
    traceId: span.traceId,
    spanId: span.spanId,
    name: span.name,
    episodeId: span.episodeId ?? null,
    originalProcessEpochId: span.processEpochId,
  })),
  terminalRecordSha256: after.at(-1)!.record.payloadSha256,
  disposition: "RECOVERED_INTERRUPTION",
};
const receipt = { ...body, receiptSha256: hashJson(body) };
const receiptPath = `${runDirectory}/telemetry/recovery-${sha256(journalRelativePath).slice(0, 16)}.json`;
await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
console.log(JSON.stringify({ ...receipt, receiptPath }, null, 2));

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
