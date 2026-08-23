import { claimRetryDelayMs, claimWithAvailabilityRetry } from "../src/queue-claim.js";

describe("availability-probed queue claims", () => {
  it("returns an immediate claim without probing or waiting", async () => {
    let probes = 0;
    const result = await claimWithAvailabilityRetry({
      workerIndex: 0,
      claim: async () => ({ id: 7 }),
      countClaimableSelected: async () => { probes += 1; return 1; },
      wait: async () => undefined,
    });
    expect(result).toEqual({
      work: { id: 7 }, emptyClaimCount: 0, retryCount: 0,
      availabilityCheckCount: 0, retryWaitMs: 0,
    });
    expect(probes).toBe(0);
  });

  it("retries transient empty READPAST results while selected work remains", async () => {
    const claims = [undefined, undefined, { id: 9 }];
    const waits: number[] = [];
    const observations: number[] = [];
    const result = await claimWithAvailabilityRetry({
      workerIndex: 3,
      claim: async () => claims.shift(),
      countClaimableSelected: async () => 12,
      wait: async (milliseconds) => { waits.push(milliseconds); },
      onRetry: (observation) => { observations.push(observation.claimableSelectedCount); },
    });
    expect(result.work).toEqual({ id: 9 });
    expect(result.emptyClaimCount).toBe(2);
    expect(result.retryCount).toBe(2);
    expect(result.availabilityCheckCount).toBe(2);
    expect(result.retryWaitMs).toBe(waits.reduce((total, value) => total + value, 0));
    expect(observations).toEqual([12, 12]);
  });

  it("stops immediately when the selected queue is actually drained", async () => {
    const result = await claimWithAvailabilityRetry({
      workerIndex: 5,
      claim: async () => undefined,
      countClaimableSelected: async () => 0,
      wait: async () => { throw new Error("terminal empty claim must not wait"); },
    });
    expect(result).toEqual({
      work: undefined, emptyClaimCount: 1, retryCount: 0,
      availabilityCheckCount: 1, retryWaitMs: 0,
    });
  });

  it("fails loudly instead of abandoning persistently claimable work", async () => {
    await expect(claimWithAvailabilityRetry({
      workerIndex: 1,
      claim: async () => undefined,
      countClaimableSelected: async () => 4,
      maxRetries: 1,
      wait: async () => undefined,
    })).rejects.toThrow("remained transiently empty after 1 retries");
  });

  it("uses a deterministic bounded worker stagger", () => {
    expect(claimRetryDelayMs(0, 3)).toBe(claimRetryDelayMs(0, 3));
    expect(claimRetryDelayMs(12, 3)).toBeLessThanOrEqual(216);
    expect(claimRetryDelayMs(0, 3)).not.toBe(claimRetryDelayMs(0, 4));
  });
});
