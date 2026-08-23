import { scheduleDelayMs, scheduleOriginEpochMs } from "../src/schedule-timing.js";

describe("resumable schedule timing", () => {
  it("starts a new schedule at the current clock", () => {
    expect(scheduleOriginEpochMs(undefined, 10_000)).toBe(10_000);
    expect(scheduleDelayMs(10_000, 2_500, 10_100)).toBe(2_400);
  });

  it("reconstructs the original clock from the earliest durable execution", () => {
    const origin = scheduleOriginEpochMs({
      plannedOffsetMs: 2_500,
      startedAtUtc: new Date(12_500),
    }, 50_000);
    expect(origin).toBe(10_000);
    expect(scheduleDelayMs(origin, 60_000, 50_000)).toBe(20_000);
  });

  it("does not wait again for offsets already elapsed during downtime", () => {
    const origin = scheduleOriginEpochMs({
      plannedOffsetMs: 0,
      startedAtUtc: new Date(10_000),
    }, 90_000);
    expect(scheduleDelayMs(origin, 60_000, 90_000)).toBe(0);
  });

  it("rejects invalid persisted offsets", () => {
    expect(() => scheduleOriginEpochMs({
      plannedOffsetMs: -1,
      startedAtUtc: new Date(10_000),
    }, 20_000)).toThrow(/non-negative/);
  });
});
