export const defaultSqlDeadlockRetryPolicy = {
  maxRetries: 8,
  baseDelayMs: 25,
  maxDelayMs: 500,
  staggerWindowMs: 23,
} as const;

export interface SqlDeadlockRetryObservation {
  operation: string;
  retryOrdinal: number;
  delayMs: number;
  errorNumber: 1205;
}

export interface SqlDeadlockRetryOptions {
  operation: string;
  workerIndex: number;
  maxRetries?: number;
  wait?: (milliseconds: number) => Promise<void>;
  onRetry?: (observation: SqlDeadlockRetryObservation) => Promise<void> | void;
}

/**
 * SQL Server fully rolls back a deadlock-victim transaction before returning
 * error 1205, so retry is safe at transaction-bounded queue operations. This
 * helper must not wrap model inference or any non-idempotent external action.
 */
export async function retrySqlDeadlock<T>(
  operation: () => Promise<T>,
  options: SqlDeadlockRetryOptions,
): Promise<T> {
  const maxRetries = options.maxRetries ?? defaultSqlDeadlockRetryPolicy.maxRetries;
  if (!Number.isSafeInteger(options.workerIndex) || options.workerIndex < 0) {
    throw new Error(`Invalid SQL retry worker index: ${options.workerIndex}`);
  }
  if (!Number.isSafeInteger(maxRetries) || maxRetries < 0) {
    throw new Error(`Invalid SQL deadlock retry limit: ${maxRetries}`);
  }
  const wait = options.wait ?? delay;
  let retryOrdinal = 0;
  while (true) {
    try {
      return await operation();
    } catch (error) {
      if (!isSqlDeadlock(error) || retryOrdinal >= maxRetries) throw error;
      const delayMs = sqlDeadlockRetryDelayMs(retryOrdinal, options.workerIndex);
      await options.onRetry?.({ operation: options.operation, retryOrdinal, delayMs, errorNumber: 1205 });
      await wait(delayMs);
      retryOrdinal += 1;
    }
  }
}

export function isSqlDeadlock(error: unknown): boolean {
  if (error === null || typeof error !== "object") return false;
  const candidate = error as {
    number?: unknown;
    message?: unknown;
    originalError?: { info?: { number?: unknown } };
  };
  return candidate.number === 1205 || candidate.originalError?.info?.number === 1205
    || (typeof candidate.message === "string" && /deadlocked on lock resources|deadlock victim/i.test(candidate.message));
}

export function sqlDeadlockRetryDelayMs(retryOrdinal: number, workerIndex: number): number {
  if (!Number.isSafeInteger(retryOrdinal) || retryOrdinal < 0) throw new Error(`Invalid SQL retry ordinal: ${retryOrdinal}`);
  if (!Number.isSafeInteger(workerIndex) || workerIndex < 0) throw new Error(`Invalid SQL retry worker index: ${workerIndex}`);
  const exponential = defaultSqlDeadlockRetryPolicy.baseDelayMs * 2 ** Math.min(retryOrdinal, 5);
  const bounded = Math.min(defaultSqlDeadlockRetryPolicy.maxDelayMs, exponential);
  const stagger = (workerIndex * 11 + retryOrdinal * 7) % defaultSqlDeadlockRetryPolicy.staggerWindowMs;
  return bounded + stagger;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
