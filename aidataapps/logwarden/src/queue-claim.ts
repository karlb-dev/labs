export const defaultClaimRetryPolicy = {
  maxRetries: 200,
  baseDelayMs: 20,
  maxDelayMs: 200,
  staggerWindowMs: 17,
} as const;

export interface ClaimRetryObservation {
  retryOrdinal: number;
  emptyClaimCount: number;
  claimableSelectedCount: number;
  delayMs: number;
}

export interface ClaimWithAvailabilityOptions<T> {
  workerIndex: number;
  claim: () => Promise<T | undefined>;
  countClaimableSelected: () => Promise<number>;
  maxRetries?: number;
  wait?: (milliseconds: number) => Promise<void>;
  onRetry?: (observation: ClaimRetryObservation) => Promise<void> | void;
}

export interface ClaimWithAvailabilityResult<T> {
  work: T | undefined;
  emptyClaimCount: number;
  retryCount: number;
  availabilityCheckCount: number;
  retryWaitMs: number;
}

/**
 * READPAST may transiently return no row when concurrent claim transactions
 * hold page locks. Probe the selected queue through RCSI before treating an
 * empty claim as terminal, and fail loudly instead of silently abandoning
 * known-claimable work if the transient condition does not clear.
 */
export async function claimWithAvailabilityRetry<T>(
  options: ClaimWithAvailabilityOptions<T>,
): Promise<ClaimWithAvailabilityResult<T>> {
  const maxRetries = options.maxRetries ?? defaultClaimRetryPolicy.maxRetries;
  if (!Number.isSafeInteger(options.workerIndex) || options.workerIndex < 0) {
    throw new Error(`Invalid queue worker index: ${options.workerIndex}`);
  }
  if (!Number.isSafeInteger(maxRetries) || maxRetries < 0) {
    throw new Error(`Invalid queue claim retry limit: ${maxRetries}`);
  }
  const wait = options.wait ?? delay;
  let emptyClaimCount = 0;
  let retryCount = 0;
  let availabilityCheckCount = 0;
  let retryWaitMs = 0;

  while (true) {
    const work = await options.claim();
    if (work !== undefined) {
      return { work, emptyClaimCount, retryCount, availabilityCheckCount, retryWaitMs };
    }
    emptyClaimCount += 1;
    const claimableSelectedCount = await options.countClaimableSelected();
    availabilityCheckCount += 1;
    if (!Number.isSafeInteger(claimableSelectedCount) || claimableSelectedCount < 0) {
      throw new Error(`Invalid selected claimable-work count: ${claimableSelectedCount}`);
    }
    if (claimableSelectedCount === 0) {
      return { work: undefined, emptyClaimCount, retryCount, availabilityCheckCount, retryWaitMs };
    }
    if (retryCount >= maxRetries) {
      throw new Error(
        `Queue claim remained transiently empty after ${retryCount} retries while ${claimableSelectedCount} selected item(s) were claimable`,
      );
    }
    const delayMs = claimRetryDelayMs(retryCount, options.workerIndex);
    const observation: ClaimRetryObservation = {
      retryOrdinal: retryCount,
      emptyClaimCount,
      claimableSelectedCount,
      delayMs,
    };
    await options.onRetry?.(observation);
    await wait(delayMs);
    retryWaitMs += delayMs;
    retryCount += 1;
  }
}

export function claimRetryDelayMs(retryOrdinal: number, workerIndex: number): number {
  if (!Number.isSafeInteger(retryOrdinal) || retryOrdinal < 0) throw new Error(`Invalid claim retry ordinal: ${retryOrdinal}`);
  if (!Number.isSafeInteger(workerIndex) || workerIndex < 0) throw new Error(`Invalid queue worker index: ${workerIndex}`);
  const exponential = defaultClaimRetryPolicy.baseDelayMs * 2 ** Math.min(retryOrdinal, 4);
  const bounded = Math.min(defaultClaimRetryPolicy.maxDelayMs, exponential);
  const stagger = (workerIndex * 7 + retryOrdinal * 11) % defaultClaimRetryPolicy.staggerWindowMs;
  return bounded + stagger;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
