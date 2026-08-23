import { performance } from "node:perf_hooks";
import sql from "mssql";
import type { AppConfig } from "./config.js";
import { captureContextSnapshot, type CapturedSnapshot, type SnapshotPlanItem } from "./context-snapshots.js";
import { sha256 } from "./hash.js";
import { connect, sqlIdentifier } from "./repository.js";

export interface SafeSqlError {
  number?: number;
  message: string;
}

export interface InjectorRequest {
  config: AppConfig;
  control: sql.ConnectionPool;
  runId: string;
  episodeId: string;
  injector: string;
  correlationToken: string;
  parameters: Record<string, unknown>;
  snapshotPlan: SnapshotPlanItem[];
  maxRuntimeSeconds: number;
}

export interface InjectorResult {
  sessionIds: number[];
  expectedSignalCount: number;
  observedSignalCount: number;
  observedErrors: SafeSqlError[];
  snapshots: CapturedSnapshot[];
  cleanupVerified: boolean;
  disposableDatabase?: {
    id: string;
    name: string;
    creationToken: string;
  };
  unexpectedError?: SafeSqlError;
}

type MutableResult = InjectorResult;

export async function executeScenarioInjector(input: InjectorRequest): Promise<InjectorResult> {
  assertToken(input.correlationToken);
  switch (input.injector) {
    case "deadlock":
      return runDeadlock(input);
    case "blocking":
      return runBlocking(input);
    case "transaction_log_full":
      return runTransactionLogFull(input);
    case "bad_login":
      return runBadLogin(input);
    default:
      return runSingleSession(input);
  }
}

export function expectedErrorNumbers(name: string): number[] {
  const mapping: Record<string, number[]> = {
    conversion_error: [245],
    missing_object: [208],
    duplicate_key: [2627],
    truncation_error: [2628, 8152],
    controlled_signal: [50000],
    ambiguous_signal: [50000],
    bad_login: [18456],
    backup_failure: [3201, 3013],
    divide_by_zero: [8134],
    deadlock: [1205],
    transaction_log_full: [9002],
    blocking: [],
    query_pressure: [],
    benign_noise: [],
  };
  return mapping[name] ?? [];
}

export function isExpectedSignal(name: string, error: SafeSqlError): boolean {
  return expectedErrorNumbers(name).includes(error.number ?? -1) ||
    (name === "bad_login" && /login failed/i.test(error.message));
}

export function safeSqlError(error: unknown): SafeSqlError {
  const candidate = error as { number?: number; message?: string };
  return {
    ...(candidate.number === undefined ? {} : { number: candidate.number }),
    message: candidate.message ?? String(error),
  };
}

async function runSingleSession(input: InjectorRequest): Promise<InjectorResult> {
  const result = emptyResult(input);
  let pool: sql.ConnectionPool | undefined;
  try {
    pool = await episodePool(input, "main");
    result.sessionIds.push(await sessionId(pool));
    for (let index = 0; index < result.expectedSignalCount; index += 1) {
      try {
        await executeSingleSignal(input, pool, index);
        if (expectedErrorNumbers(input.injector).length === 0) result.observedSignalCount += 1;
      } catch (error) {
        const observed = safeSqlError(error);
        result.observedErrors.push(observed);
        if (isExpectedSignal(input.injector, observed)) result.observedSignalCount += 1;
        else {
          result.unexpectedError = observed;
          break;
        }
      }
    }
    requireSignals(input, result);
    if (result.unexpectedError === undefined) result.snapshots = await capturePlans(input);
  } catch (error) {
    result.unexpectedError ??= safeSqlError(error);
  } finally {
    if (pool !== undefined) await pool.close().catch(() => undefined);
  }
  return result;
}

async function executeSingleSignal(input: InjectorRequest, pool: sql.ConnectionPool, index: number): Promise<void> {
  const marker = `LW_EVT_${input.correlationToken}_${index + 1}`;
  const variant = boundedVariant(input.parameters.messageVariant);
  switch (input.injector) {
    case "conversion_error":
      await pool.request().input("value", sql.NVarChar(200), `not_an_int_${marker}_${variant}`)
        .query(`/* ${marker}:convert */ SELECT CONVERT(int,@value);`);
      return;
    case "missing_object":
      await pool.request().query(`/* ${marker}:missing */ SELECT TOP (1) * FROM lab.LW_Missing_${input.correlationToken.slice(0, 16)}_${index + 1};`);
      return;
    case "duplicate_key":
      await pool.request().input("marker", sql.VarChar(40), marker.slice(0, 40))
        .query(`/* ${marker}:duplicate */ INSERT lab.accounts(account_id,opaque_code,amount) VALUES(1,@marker,1.00);`);
      return;
    case "truncation_error":
      await pool.request().query(`/* ${marker}:truncate */ UPDATE lab.accounts SET short_value=REPLICATE('x',${80 + variant}) WHERE account_id=1;`);
      return;
    case "controlled_signal": {
      const messages = ["integrity verification warning", "consistency check requires review", "storage integrity signal", "page verification alert", "integrity evidence requires escalation"];
      await pool.request()
        .input("message", sql.NVarChar(1000), `${marker}: ${messages[variant - 1]}`)
        .input("severity", sql.Int, 16)
        .input("state", sql.Int, 40 + variant)
        .execute("lab.usp_raise_controlled_signal");
      return;
    }
    case "ambiguous_signal": {
      const messages = ["activity may require review", "evidence is incomplete", "signals disagree", "classification is uncertain", "unsupported signature observed"];
      await pool.request()
        .input("message", sql.NVarChar(1000), `${marker}: ${messages[variant - 1]}`)
        .input("severity", sql.Int, 11)
        .input("state", sql.Int, 50 + variant)
        .execute("lab.usp_raise_controlled_signal");
      return;
    }
    case "backup_failure":
      await executeFailedBackup(input, pool, marker, index);
      return;
    case "query_pressure": {
      const bound = 800 + variant * 40;
      await pool.request().query(`/* ${marker}:pressure */ SELECT SUM(CONVERT(bigint,a.value)*CONVERT(bigint,b.value)) AS bounded_work FROM GENERATE_SERIES(1,${bound}) AS a CROSS JOIN GENERATE_SERIES(1,${bound}) AS b OPTION(MAXDOP 1);`);
      return;
    }
    case "benign_noise":
      await pool.request().input("value", sql.NVarChar(400), `${input.episodeId}:${marker}:${variant}`)
        .query(`/* ${marker}:benign */ INSERT lab.noise_rows(opaque_value) VALUES(@value); SELECT COUNT_BIG(*) AS row_count FROM lab.noise_rows;`);
      return;
    case "divide_by_zero":
      await pool.request().query(`SET ARITHABORT ON; /* ${marker}:divide */ SELECT 1/0 AS impossible_value;`);
      return;
    default:
      throw new Error(`Unsupported single-session injector: ${input.injector}`);
  }
}

async function executeFailedBackup(
  input: InjectorRequest,
  pool: sql.ConnectionPool,
  marker: string,
  index: number,
): Promise<void> {
  const destination = `/root/logwarden-forbidden/${marker}_${index + 1}.bak`;
  const started = new Date();
  let succeeded = false;
  let observed: SafeSqlError | undefined;
  try {
    await pool.request().input("destination", sql.NVarChar(500), destination)
      .query(`/* ${marker}:backup */ BACKUP DATABASE [LogWardenWorkload] TO DISK=@destination WITH COPY_ONLY, CHECKSUM;`);
    succeeded = true;
  } catch (error) {
    observed = safeSqlError(error);
    throw error;
  } finally {
    await input.control.request()
      .input("episode", sql.VarChar(120), input.episodeId)
      .input("database", sql.NVarChar(128), input.config.databases.workloadName)
      .input("destination_hash", sql.Char(64), sha256(destination))
      .input("started", sql.DateTime2(7), started)
      .input("finished", sql.DateTime2(7), new Date())
      .input("succeeded", sql.Bit, succeeded)
      .input("error", sql.Int, observed?.number ?? null)
      .input("detail", sql.NVarChar(sql.MAX), JSON.stringify({ schemaVersion: 1, attemptOrdinal: index + 1, expectedFailure: true }))
      .query(`
        INSERT workload.backup_attempts
          (episode_id,database_name,destination_hash,started_at_utc,finished_at_utc,
           succeeded,error_number,detail_json)
        VALUES(@episode,@database,@destination_hash,@started,@finished,@succeeded,@error,@detail);
      `);
  }
}

async function runBadLogin(input: InjectorRequest): Promise<InjectorResult> {
  const result = emptyResult(input);
  for (let index = 0; index < result.expectedSignalCount; index += 1) {
    const invalid = new sql.ConnectionPool({
      ...input.config.databases.lab,
      user: `lw_missing_${input.correlationToken.slice(0, 12)}_${index + 1}`,
      password: `Invalid!${input.correlationToken}${index + 1}`,
      database: input.config.databases.controlName,
      connectionTimeout: Math.min(5_000, input.maxRuntimeSeconds * 1_000),
      options: {
        ...input.config.databases.lab.options,
        appName: episodeAppName(input, `auth-${index + 1}`),
      },
    });
    try {
      await invalid.connect();
      result.unexpectedError = { message: "Invalid login unexpectedly connected" };
    } catch (error) {
      const observed = safeSqlError(error);
      result.observedErrors.push(observed);
      if (isExpectedSignal(input.injector, observed)) result.observedSignalCount += 1;
      else result.unexpectedError = observed;
    } finally {
      await invalid.close().catch(() => undefined);
    }
    if (result.unexpectedError !== undefined) break;
  }
  requireSignals(input, result);
  try {
    if (result.unexpectedError === undefined) result.snapshots = await capturePlans(input);
  } catch (error) {
    result.unexpectedError = safeSqlError(error);
  }
  return result;
}

async function runDeadlock(input: InjectorRequest): Promise<InjectorResult> {
  const result = emptyResult(input);
  for (let index = 0; index < result.expectedSignalCount && result.unexpectedError === undefined; index += 1) {
    let left: sql.ConnectionPool | undefined;
    let right: sql.ConnectionPool | undefined;
    let leftTransaction: sql.Transaction | undefined;
    let rightTransaction: sql.Transaction | undefined;
    try {
      [left, right] = await Promise.all([
        episodePool(input, `dead-${index + 1}-a`),
        episodePool(input, `dead-${index + 1}-b`),
      ]);
      result.sessionIds.push(await sessionId(left), await sessionId(right));
      leftTransaction = new sql.Transaction(left);
      rightTransaction = new sql.Transaction(right);
      await Promise.all([
        leftTransaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE),
        rightTransaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE),
      ]);
      const marker = `LW_EVT_${input.correlationToken}_${index + 1}`;
      await new sql.Request(leftTransaction).query(`SET DEADLOCK_PRIORITY NORMAL; /* ${marker}:dead:a1 */ UPDATE lab.lock_a SET value=value+1 WHERE id=1;`);
      await new sql.Request(rightTransaction).query(`SET DEADLOCK_PRIORITY LOW; /* ${marker}:dead:b1 */ UPDATE lab.lock_b SET value=value+1 WHERE id=1;`);
      const terminal = await Promise.allSettled([
        new sql.Request(leftTransaction).query(`/* ${marker}:dead:a2 */ UPDATE lab.lock_b SET value=value+1 WHERE id=1;`),
        new sql.Request(rightTransaction).query(`/* ${marker}:dead:b2 */ UPDATE lab.lock_a SET value=value+1 WHERE id=1;`),
      ]);
      for (const settled of terminal) {
        if (settled.status !== "rejected") continue;
        const observed = safeSqlError(settled.reason);
        result.observedErrors.push(observed);
        if (isExpectedSignal("deadlock", observed)) result.observedSignalCount += 1;
        else result.unexpectedError ??= observed;
      }
      if (!terminal.some((settled) => settled.status === "rejected"))
        result.unexpectedError = { message: "Deadlock cycle completed without selecting a victim" };
    } catch (error) {
      const observed = safeSqlError(error);
      result.observedErrors.push(observed);
      if (isExpectedSignal("deadlock", observed)) result.observedSignalCount += 1;
      else result.unexpectedError ??= observed;
    } finally {
      await Promise.allSettled([
        leftTransaction?.rollback() ?? Promise.resolve(),
        rightTransaction?.rollback() ?? Promise.resolve(),
      ]);
      await Promise.allSettled([
        left?.close() ?? Promise.resolve(),
        right?.close() ?? Promise.resolve(),
      ]);
    }
  }
  requireSignals(input, result);
  try {
    // The victim error returns before the asynchronous event-file target is
    // guaranteed to expose the graph. Wait beyond the governed dispatch
    // latency so the replay snapshot describes this episode, not its predecessor.
    if (result.unexpectedError === undefined) {
      await new Promise((resolve) => setTimeout(resolve, 2_500));
      result.snapshots = await capturePlans(input);
    }
  } catch (error) {
    result.unexpectedError = safeSqlError(error);
  }
  return result;
}

async function runBlocking(input: InjectorRequest): Promise<InjectorResult> {
  const result = emptyResult(input);
  let blocker: sql.ConnectionPool | undefined;
  let blockerTransaction: sql.Transaction | undefined;
  const waiters: sql.ConnectionPool[] = [];
  const waiterRequests: Array<Promise<unknown>> = [];
  const started = performance.now();
  try {
    blocker = await episodePool(input, "block-hold");
    blockerTransaction = new sql.Transaction(blocker);
    await blockerTransaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
    result.sessionIds.push(await transactionSessionId(blockerTransaction));
    const marker = `LW_EVT_${input.correlationToken}`;
    await new sql.Request(blockerTransaction)
      .query(`/* ${marker}:block:hold */ UPDATE lab.lock_a SET value=value+1 WHERE id=2;`);
    for (let index = 0; index < result.expectedSignalCount; index += 1) {
      const waiter = await episodePool(input, `block-w${index + 1}`);
      waiters.push(waiter);
      result.sessionIds.push(await sessionId(waiter));
      waiterRequests.push(waiter.request().query(
        `SET LOCK_TIMEOUT 18000; /* ${marker}:block:wait:${index + 1} */ UPDATE lab.lock_a SET value=value+1 WHERE id=2;`,
      ));
    }
    result.snapshots = await capturePlans(input, started, true);
    await waitUntil(started, 12_500);
    await blockerTransaction.rollback();
    blockerTransaction = undefined;
    const settled = await Promise.allSettled(waiterRequests);
    for (const waiter of settled) {
      if (waiter.status === "fulfilled") result.observedSignalCount += 1;
      else result.unexpectedError ??= safeSqlError(waiter.reason);
    }
  } catch (error) {
    result.unexpectedError ??= safeSqlError(error);
  } finally {
    if (blockerTransaction !== undefined) await blockerTransaction.rollback().catch(() => undefined);
    await Promise.allSettled(waiters.map((pool) => pool.close()));
    if (blocker !== undefined) await blocker.close().catch(() => undefined);
  }
  requireSignals(input, result);
  return result;
}

async function runTransactionLogFull(input: InjectorRequest): Promise<InjectorResult> {
  const result = emptyResult(input);
  const databaseName = `LW_${input.correlationToken.slice(0, 24)}`;
  const identifier = sqlIdentifier(databaseName);
  const creationToken = formatUuid(input.correlationToken);
  const inserted = await input.control.request()
    .input("run", sql.VarChar(120), input.runId)
    .input("episode", sql.VarChar(120), input.episodeId)
    .input("database", sql.NVarChar(128), databaseName)
    .input("token", sql.UniqueIdentifier, creationToken)
    .query<{ disposable_database_id: string }>(`
      INSERT workload.disposable_databases
        (run_id,episode_id,database_name,creation_token,cleanup_status)
      OUTPUT INSERTED.disposable_database_id
      VALUES(@run,@episode,@database,@token,'creating');
    `);
  const disposableId = inserted.recordset[0]!.disposable_database_id;
  result.disposableDatabase = { id: disposableId, name: databaseName, creationToken };
  let master: sql.ConnectionPool | undefined;
  let holder: sql.ConnectionPool | undefined;
  let filler: sql.ConnectionPool | undefined;
  let holderTransaction: sql.Transaction | undefined;
  let databaseCreated = false;
  try {
    master = await connect(episodeConnection(input, "log-ctl"), "master", input.maxRuntimeSeconds * 1_000);
    result.sessionIds.push(await sessionId(master));
    const fileStem = `/var/opt/mssql/data/${databaseName}`;
    await master.request().query(`
      CREATE DATABASE ${identifier}
      ON PRIMARY
      (NAME=N'${databaseName}_data',FILENAME=N'${fileStem}.mdf',SIZE=8MB,MAXSIZE=64MB,FILEGROWTH=8MB)
      LOG ON
      (NAME=N'${databaseName}_log',FILENAME=N'${fileStem}.ldf',SIZE=8MB,MAXSIZE=8MB,FILEGROWTH=1MB);
    `);
    databaseCreated = true;
    await input.control.request().input("id", sql.BigInt, disposableId)
      .query("UPDATE workload.disposable_databases SET cleanup_status='active' WHERE disposable_database_id=@id;");
    [holder, filler] = await Promise.all([
      connect(episodeConnection(input, "log-hold"), databaseName, input.maxRuntimeSeconds * 1_000),
      connect(episodeConnection(input, "log-fill"), databaseName, input.maxRuntimeSeconds * 1_000),
    ]);
    result.sessionIds.push(await sessionId(holder), await sessionId(filler));
    await filler.request().query("CREATE TABLE dbo.lw_fill(id bigint IDENTITY PRIMARY KEY,payload char(8000) NOT NULL); CHECKPOINT;");
    holderTransaction = new sql.Transaction(holder);
    await holderTransaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
    const marker = `LW_EVT_${input.correlationToken}`;
    await new sql.Request(holderTransaction)
      .query(`/* ${marker}:log:hold */ INSERT dbo.lw_fill(payload) VALUES(REPLICATE('h',8000));`);
    for (let attempt = 0; attempt < 100 && result.observedSignalCount < result.expectedSignalCount; attempt += 1) {
      try {
        await filler.request().query(`/* ${marker}:log:fill:${attempt + 1} */ INSERT dbo.lw_fill(payload) SELECT REPLICATE('x',8000) FROM GENERATE_SERIES(1,64);`);
      } catch (error) {
        const observed = safeSqlError(error);
        result.observedErrors.push(observed);
        if (isExpectedSignal("transaction_log_full", observed)) result.observedSignalCount += 1;
        else {
          result.unexpectedError = observed;
          break;
        }
      }
    }
    requireSignals(input, result);
    if (result.unexpectedError === undefined)
      result.snapshots = await capturePlans(input, undefined, false, databaseName);
  } catch (error) {
    result.unexpectedError ??= safeSqlError(error);
  } finally {
    if (holderTransaction !== undefined) await holderTransaction.rollback().catch(() => undefined);
    await Promise.allSettled([
      holder?.close() ?? Promise.resolve(),
      filler?.close() ?? Promise.resolve(),
    ]);
    let dropped = !databaseCreated;
    if (master !== undefined) {
      try {
        await master.request().query(`IF DB_ID(N'${databaseName}') IS NOT NULL DROP DATABASE ${identifier};`);
        const check = await master.request().query<{ database_id: number | null }>(`SELECT DB_ID(N'${databaseName}') AS database_id;`);
        dropped = check.recordset[0]?.database_id == null;
      } catch (error) {
        result.unexpectedError ??= safeSqlError(error);
      }
      await master.close().catch(() => undefined);
    }
    result.cleanupVerified = dropped;
    await input.control.request()
      .input("id", sql.BigInt, disposableId)
      .input("dropped", sql.DateTime2(7), dropped ? new Date() : null)
      .input("status", sql.VarChar(24), dropped ? "dropped" : "cleanup_failed")
      .query(`
        UPDATE workload.disposable_databases
        SET dropped_at_utc=@dropped,cleanup_status=@status
        WHERE disposable_database_id=@id;
      `);
  }
  return result;
}

async function capturePlans(
  input: InjectorRequest,
  startedMonotonic?: number,
  honorOffsets = false,
  disposableDatabaseName?: string,
): Promise<CapturedSnapshot[]> {
  if (input.snapshotPlan.length === 0) return [];
  const agent = await connect(input.config.databases.agent, input.config.databases.controlName, 30_000);
  const captured: CapturedSnapshot[] = [];
  try {
    for (const plan of [...input.snapshotPlan].sort((left, right) => left.atMsAfterStart - right.atMsAfterStart)) {
      if (honorOffsets && startedMonotonic !== undefined) await waitUntil(startedMonotonic, plan.atMsAfterStart);
      captured.push(await captureContextSnapshot({
        plan,
        episodeId: input.episodeId,
        ...(disposableDatabaseName === undefined ? {} : { disposableDatabaseName }),
        agent,
        control: input.control,
      }));
    }
  } finally {
    await agent.close();
  }
  return captured;
}

async function episodePool(input: InjectorRequest, suffix: string): Promise<sql.ConnectionPool> {
  return connect(
    episodeConnection(input, suffix),
    input.config.databases.workloadName,
    input.maxRuntimeSeconds * 1_000,
  );
}

function episodeConnection(input: InjectorRequest, suffix: string): AppConfig["databases"]["lab"] {
  return {
    ...input.config.databases.lab,
    options: {
      ...input.config.databases.lab.options,
      appName: episodeAppName(input, suffix),
    },
  };
}

function episodeAppName(input: InjectorRequest, suffix: string): string {
  return `LogWarden-Inject-${input.correlationToken.slice(0, 12)}-${suffix}`.slice(0, 128);
}

async function sessionId(pool: sql.ConnectionPool): Promise<number> {
  const result = await pool.request().query<{ spid: number }>("SELECT @@SPID AS spid;");
  return result.recordset[0]!.spid;
}

async function transactionSessionId(transaction: sql.Transaction): Promise<number> {
  const result = await new sql.Request(transaction).query<{ spid: number }>("SELECT @@SPID AS spid;");
  return result.recordset[0]!.spid;
}

function emptyResult(input: InjectorRequest): MutableResult {
  return {
    sessionIds: [],
    expectedSignalCount: signalCount(input.parameters),
    observedSignalCount: 0,
    observedErrors: [],
    snapshots: [],
    cleanupVerified: input.injector !== "transaction_log_full",
  };
}

function signalCount(parameters: Record<string, unknown>): number {
  const value = parameters.signalCount ?? 1;
  if (!Number.isInteger(value) || Number(value) < 1 || Number(value) > 3)
    throw new Error(`signalCount must be an integer from 1 through 3; got ${String(value)}`);
  return Number(value);
}

function boundedVariant(value: unknown): number {
  const parsed = Number(value ?? 1);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 5) return 1;
  return parsed;
}

function requireSignals(input: InjectorRequest, result: MutableResult): void {
  if (result.unexpectedError === undefined && result.observedSignalCount < result.expectedSignalCount) {
    result.unexpectedError = {
      message: `${input.injector} produced ${result.observedSignalCount}/${result.expectedSignalCount} expected driver signals`,
    };
  }
}

function formatUuid(compact: string): string {
  return `${compact.slice(0, 8)}-${compact.slice(8, 12)}-${compact.slice(12, 16)}-${compact.slice(16, 20)}-${compact.slice(20, 32)}`;
}

function assertToken(token: string): void {
  if (!/^[0-9a-f]{32}$/.test(token)) throw new Error("Correlation token must be 32 lowercase hexadecimal characters");
}

async function waitUntil(startedMonotonic: number, offsetMs: number): Promise<void> {
  const remaining = offsetMs - (performance.now() - startedMonotonic);
  if (remaining > 0) await new Promise((resolve) => setTimeout(resolve, remaining));
}
