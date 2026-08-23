import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { readErrorlogFilesFromContainer } from "../src/errorlog-files.js";
import { parseErrorlogFiles, type ParsedErrorlogRecord, type ParsedErrorlogScan } from "../src/errorlog.js";
import { hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

const mode = argument("--mode") ?? "rotation";
if (mode !== "rotation" && mode !== "restart") throw new Error("--mode must be rotation or restart");
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const config = loadConfig();
const container = process.env.LOGWARDEN_SQL_CONTAINER ?? "aidataapps-logwarden-sqlserver-1";
const markerPrefix = mode === "rotation" ? "LW_ERRORLOG_ROTATION_GATE" : "LW_ERRORLOG_RESTART_GATE";
const preMarker = `${markerPrefix}_PRE_${randomUUID().replaceAll("-", "")}`;
const postMarker = `${markerPrefix}_POST_${randomUUID().replaceAll("-", "")}`;

await writeMarker(preMarker);
await delay(250);
const before = await scan();
const preBefore = marker(before, preMarker);
const beforeIngest = await persist(before, `${mode}-before`);

let recoveryEvidence: Record<string, unknown>;
if (mode === "rotation") {
  await withAdmin((pool) => pool.request().execute("sys.sp_cycle_errorlog").then(() => undefined));
  recoveryEvidence = { operation: "sys.sp_cycle_errorlog", completed: true };
} else {
  recoveryEvidence = await restartSqlContainer();
}

await writeMarker(postMarker);
await delay(500);
const after = await scan();
const preAfter = marker(after, preMarker);
const postAfter = marker(after, postMarker);
if (preBefore.sourcePositionKey !== preAfter.sourcePositionKey) throw new Error("Pre-recovery ERRORLOG source key changed after file rename");
if (preBefore.sourceFileName === preAfter.sourceFileName) throw new Error("ERRORLOG recovery did not move the pre-marker to a renamed generation");
const afterIngest = await persist(after, `${mode}-after`);
const replay = await persist(after, `${mode}-replay`);
if (replay.rows_inserted !== 0 || replay.rows_duplicate !== replay.rows_read) {
  throw new Error(`ERRORLOG immediate replay was not fully idempotent: ${JSON.stringify(replay)}`);
}
const databaseEvidence = await verifyDatabase(preBefore, postAfter);
const receiptBody = {
  schemaVersion: 1,
  runId: run.runId,
  mode,
  parserVersion: "errorlog-file-v1",
  preMarker,
  postMarker,
  preMarkerKey: preBefore.sourcePositionKey,
  postMarkerKey: postAfter.sourcePositionKey,
  preMarkerFileBefore: preBefore.sourceFileName,
  preMarkerFileAfter: preAfter.sourceFileName,
  postMarkerFile: postAfter.sourceFileName,
  generationCountBefore: before.cursor.generations.length,
  generationCountAfter: after.cursor.generations.length,
  recoveryEvidence,
  ingestion: { before: beforeIngest, after: afterIngest, immediateReplay: replay },
  databaseEvidence,
  disposition: "PASS",
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/ingestion/errorlog-${mode}-gate.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({ runId: run.runId, mode, preMarkerKey: preBefore.sourcePositionKey, postMarkerKey: postAfter.sourcePositionKey, immediateReplay: replay, disposition: "PASS", receiptSha256: receipt.receiptSha256 }, null, 2));

async function scan(): Promise<ParsedErrorlogScan> {
  return parseErrorlogFiles(await readErrorlogFilesFromContainer(container));
}

function marker(scanResult: ParsedErrorlogScan, value: string): ParsedErrorlogRecord {
  const matches = scanResult.records.filter((record) => record.rawText.includes(value));
  if (matches.length !== 1) throw new Error(`Expected exactly one ERRORLOG marker ${value}; observed ${matches.length}`);
  return matches[0]!;
}

async function persist(scanResult: ParsedErrorlogScan, worker: string): Promise<{ ingestion_batch_id: string; rows_read: number; rows_inserted: number; rows_duplicate: number }> {
  const pool = await connect(config.databases.lab, config.databases.controlName);
  try {
    const result = await pool.request()
      .input("worker_id", sql.VarChar(120), `errorlog-gate-${worker}`)
      .input("records_json", sql.NVarChar(sql.MAX), JSON.stringify(scanResult.records))
      .input("cursor_json", sql.NVarChar(sql.MAX), JSON.stringify(scanResult.cursor))
      .input("log_generation", sql.Int, scanResult.cursor.generations.length)
      .execute<{ ingestion_batch_id: string; rows_read: number; rows_inserted: number; rows_duplicate: number }>("ingest.usp_ingest_errorlog_scan");
    return result.recordset[0]!;
  } finally {
    await pool.close();
  }
}

async function verifyDatabase(pre: ParsedErrorlogRecord, post: ParsedErrorlogRecord): Promise<Record<string, unknown>> {
  const pool = await connect(config.databases.lab, config.databases.controlName);
  try {
    const result = await pool.request()
      .input("pre", sql.Char(64), pre.sourcePositionKey)
      .input("post", sql.Char(64), post.sourcePositionKey)
      .query<{ raw_rows: number; canonical_rows: number; unique_keys: number }>(`
        SELECT
          (SELECT COUNT(*) FROM ingest.raw_events WHERE source_position_key IN (@pre,@post)) AS raw_rows,
          (SELECT COUNT(*) FROM ingest.canonical_events AS c INNER JOIN ingest.raw_events AS r ON r.raw_event_id=c.raw_event_id WHERE r.source_position_key IN (@pre,@post)) AS canonical_rows,
          (SELECT COUNT(DISTINCT source_position_key) FROM ingest.raw_events WHERE source_position_key IN (@pre,@post)) AS unique_keys;
      `);
    const row = result.recordset[0]!;
    if (Number(row.raw_rows) !== 2 || Number(row.canonical_rows) !== 2 || Number(row.unique_keys) !== 2) {
      throw new Error(`ERRORLOG database evidence mismatch: ${JSON.stringify(row)}`);
    }
    return row;
  } finally {
    await pool.close();
  }
}

async function writeMarker(value: string): Promise<void> {
  await withAdmin(async (pool) => {
    await pool.request()
      .input("message", sql.NVarChar(1800), value)
      .query("DECLARE @value nvarchar(1800)=@message; RAISERROR(@value,10,1) WITH LOG;");
  });
}

async function withAdmin<T>(operation: (pool: sql.ConnectionPool) => Promise<T>): Promise<T> {
  const pool = await connect(config.databases.admin, "master", 30_000);
  try {
    return await operation(pool);
  } finally {
    await pool.close();
  }
}

function docker(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile("docker", args, { encoding: "utf8", maxBuffer: 4 * 1024 * 1024, timeout: 10_000 }, (error, stdout, stderr) => {
      if (error === null) resolve(stdout);
      else reject(new Error(`docker ${args[0]} failed: ${stderr.trim() || error.message}`));
    });
  });
}

async function restartSqlContainer(): Promise<Record<string, unknown>> {
  const beforeState = await containerState();
  if (!beforeState.Running || !Number.isSafeInteger(beforeState.Pid) || beforeState.Pid <= 1) {
    throw new Error(`SQL container is not running with a valid init PID: ${JSON.stringify(beforeState)}`);
  }
  const beforeSqlStartUtc = await sqlStartTime();
  const processRows = await processTable();
  const candidates = processRows.filter((row) => row.ppid === beforeState.Pid && row.command === "/opt/mssql/bin/sqlservr");
  if (candidates.length !== 1) {
    throw new Error(`Expected one sqlservr child of inspected container PID ${beforeState.Pid}; observed ${JSON.stringify(candidates)}`);
  }
  const sqlParentPid = candidates[0]!.pid;
  const signaledAtUtc = new Date().toISOString();
  process.kill(sqlParentPid, "SIGTERM");
  let dockerStartIssued = false;
  let afterState: ContainerState | null = null;
  let afterSqlStartUtc: string | null = null;
  await waitFor(async () => {
    try {
      afterState = await containerState();
      if (!afterState.Running) {
        if (!dockerStartIssued) {
          await docker(["start", container]);
          dockerStartIssued = true;
        }
        return false;
      }
      if (afterState.Pid === beforeState.Pid) return false;
      try {
        afterSqlStartUtc = await sqlStartTime();
        return afterSqlStartUtc !== beforeSqlStartUtc && await labDatabasesReady();
      } catch {
        return false;
      }
    } catch {
      return false;
    }
  }, 90_000, "SQL container did not restart with a new PID/start time after targeted SIGTERM");
  return {
    operation: "SIGTERM exact sqlservr child of inspected host-PID container; unless-stopped recovery",
    containerId: beforeState.Id,
    beforeContainerPid: beforeState.Pid,
    signaledSqlParentPid: sqlParentPid,
    beforeSqlStartUtc,
    signaledAtUtc,
    dockerStartIssued,
    afterContainerPid: afterState!.Pid,
    afterSqlStartUtc,
    readyAtUtc: new Date().toISOString(),
  };
}

interface ContainerState { Id: string; Running: boolean; Pid: number }

async function containerState(): Promise<ContainerState> {
  const output = await docker(["inspect", "--format", "{{json .Id}} {{json .State.Running}} {{json .State.Pid}}", container]);
  const match = /^("[0-9a-f]+")\s+(true|false)\s+(\d+)\s*$/.exec(output);
  if (match === null) throw new Error(`Unexpected docker inspect state: ${output.trim()}`);
  return { Id: JSON.parse(match[1]!) as string, Running: match[2] === "true", Pid: Number(match[3]) };
}

async function sqlStartTime(): Promise<string> {
  return withAdmin(async (pool) => {
    const result = await pool.request().query<{ sqlserver_start_time: Date }>("SELECT sqlserver_start_time FROM sys.dm_os_sys_info;");
    return result.recordset[0]!.sqlserver_start_time.toISOString();
  });
}

async function labDatabasesReady(): Promise<boolean> {
  const adminReady = await withAdmin(async (pool) => {
    const result = await pool.request().query<{ online_count: number }>(`
      SELECT COUNT(*) AS online_count
      FROM sys.databases
      WHERE name IN ('LogWardenControl','LogWardenWorkload')
        AND state_desc='ONLINE' AND user_access_desc='MULTI_USER';
    `);
    return Number(result.recordset[0]?.online_count) === 2;
  });
  if (!adminReady) return false;
  const pool = await connect(config.databases.lab, config.databases.controlName, 10_000);
  try {
    await pool.request().query("SELECT 1 AS ready;");
    return true;
  } finally {
    await pool.close();
  }
}

function processTable(): Promise<Array<{ pid: number; ppid: number; command: string }>> {
  return new Promise((resolve, reject) => {
    execFile("ps", ["-eo", "pid=,ppid=,args="], { encoding: "utf8", maxBuffer: 8 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error !== null) {
        reject(new Error(`ps failed: ${stderr.trim() || error.message}`));
        return;
      }
      resolve(stdout.split("\n").flatMap((line) => {
        const match = /^\s*(\d+)\s+(\d+)\s+(.*?)\s*$/.exec(line);
        return match === null ? [] : [{ pid: Number(match[1]), ppid: Number(match[2]), command: match[3]! }];
      }));
    });
  });
}

async function waitFor(predicate: () => Promise<boolean>, timeoutMs: number, message: string): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate()) return;
    await delay(1_000);
  }
  throw new Error(message);
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
