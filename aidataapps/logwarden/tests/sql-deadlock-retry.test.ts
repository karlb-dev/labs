import { isSqlDeadlock, retrySqlDeadlock, sqlDeadlockRetryDelayMs } from "../src/sql-deadlock-retry.js";

describe("SQL deadlock retries", () => {
  it("recognizes direct, nested, and message-only SQL Server 1205 errors", () => {
    expect(isSqlDeadlock({ number: 1205 })).toBe(true);
    expect(isSqlDeadlock({ originalError: { info: { number: 1205 } } })).toBe(true);
    expect(isSqlDeadlock({ message: "Transaction was deadlocked on lock resources" })).toBe(true);
    expect(isSqlDeadlock({ number: 2601, message: "duplicate key" })).toBe(false);
  });

  it("retries only the rolled-back operation with deterministic bounded waits", async () => {
    let attempts = 0;
    const waits: number[] = [];
    const observations: number[] = [];
    const result = await retrySqlDeadlock(async () => {
      attempts += 1;
      if (attempts < 3) throw { number: 1205, message: "deadlock victim" };
      return "claimed";
    }, {
      operation: "queue_claim",
      workerIndex: 4,
      wait: async (milliseconds) => { waits.push(milliseconds); },
      onRetry: (observation) => { observations.push(observation.retryOrdinal); },
    });
    expect(result).toBe("claimed");
    expect(attempts).toBe(3);
    expect(observations).toEqual([0, 1]);
    expect(waits).toEqual([sqlDeadlockRetryDelayMs(0, 4), sqlDeadlockRetryDelayMs(1, 4)]);
  });

  it("does not retry non-deadlock failures", async () => {
    let attempts = 0;
    await expect(retrySqlDeadlock(async () => {
      attempts += 1;
      throw { number: 2627, message: "unique constraint" };
    }, { operation: "queue_claim", workerIndex: 0, wait: async () => undefined }))
      .rejects.toMatchObject({ number: 2627 });
    expect(attempts).toBe(1);
  });

  it("honors the retry ceiling", async () => {
    let attempts = 0;
    await expect(retrySqlDeadlock(async () => {
      attempts += 1;
      throw { number: 1205 };
    }, { operation: "availability_probe", workerIndex: 0, maxRetries: 1, wait: async () => undefined }))
      .rejects.toMatchObject({ number: 1205 });
    expect(attempts).toBe(2);
  });
});
