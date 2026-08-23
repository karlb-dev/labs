export interface ScheduleCheckpoint {
  plannedOffsetMs: number;
  startedAtUtc: Date;
}

/**
 * Reconstruct the immutable schedule clock from the earliest durable execution.
 * A resumed injector therefore waits only until the original planned offsets,
 * rather than starting the full schedule clock again.
 */
export function scheduleOriginEpochMs(checkpoint: ScheduleCheckpoint | undefined, nowEpochMs: number): number {
  assertFiniteEpoch(nowEpochMs, "current time");
  if (checkpoint === undefined) return nowEpochMs;
  if (!Number.isSafeInteger(checkpoint.plannedOffsetMs) || checkpoint.plannedOffsetMs < 0)
    throw new Error("Schedule checkpoint offset must be a non-negative safe integer");
  const startedAtEpochMs = checkpoint.startedAtUtc.getTime();
  assertFiniteEpoch(startedAtEpochMs, "schedule checkpoint start time");
  return startedAtEpochMs - checkpoint.plannedOffsetMs;
}

export function scheduleDelayMs(originEpochMs: number, plannedOffsetMs: number, nowEpochMs: number): number {
  assertFiniteEpoch(originEpochMs, "schedule origin");
  assertFiniteEpoch(nowEpochMs, "current time");
  if (!Number.isSafeInteger(plannedOffsetMs) || plannedOffsetMs < 0)
    throw new Error("Planned offset must be a non-negative safe integer");
  return Math.max(0, originEpochMs + plannedOffsetMs - nowEpochMs);
}

function assertFiniteEpoch(value: number, label: string): void {
  if (!Number.isFinite(value)) throw new Error(`Invalid ${label}`);
}
