import sql from "mssql";
import { performance } from "node:perf_hooks";
import { canonicalJson, hashJson } from "./hash.js";
import { canonicalToolArguments, type ToolName } from "./tools.js";

export interface SnapshotPlanItem {
  tool: ToolName;
  atMsAfterStart: number;
  arguments: Record<string, unknown>;
}

export interface CapturedSnapshot {
  tool: ToolName;
  arguments: Record<string, unknown>;
  argumentsSha256: string;
  resultSha256: string;
  rowCount: number;
  latencyMs: number;
  capturedAtUtc: string;
}

interface CaptureInput {
  plan: SnapshotPlanItem;
  episodeId: string;
  disposableDatabaseName?: string;
  agent: sql.ConnectionPool;
  control: sql.ConnectionPool;
}

export function materializeSnapshotArguments(
  tool: ToolName,
  input: Record<string, unknown>,
  disposableDatabaseName?: string,
): Record<string, unknown> {
  const materialized = Object.fromEntries(Object.entries(input).map(([key, value]) => [
    key,
    value === "TARGET_DATABASE"
      ? requireDisposableDatabase(disposableDatabaseName)
      : value,
  ]));
  return canonicalToolArguments(tool, materialized);
}

export async function captureContextSnapshot(input: CaptureInput): Promise<CapturedSnapshot> {
  const args = materializeSnapshotArguments(input.plan.tool, input.plan.arguments, input.disposableDatabaseName);
  const argsSha256 = hashJson(args);
  const requestedAtUtc = new Date();
  const started = performance.now();
  const result = await invokeSnapshotTool(input.agent, input.plan.tool, args);
  const latencyMs = performance.now() - started;
  const capturedAtUtc = new Date();
  const rows = JSON.parse(JSON.stringify(result.recordset)) as unknown[];
  const body = { schemaVersion: 1, status: "ok", tool: input.plan.tool, rows };
  const resultJson = canonicalJson(body);
  const resultSha256 = hashJson(body);
  await input.control.request()
    .input("episode", sql.VarChar(120), input.episodeId)
    .input("kind", sql.VarChar(80), input.plan.tool.replace(/^get_/, ""))
    .input("tool", sql.VarChar(80), input.plan.tool)
    .input("args_hash", sql.Char(64), argsSha256)
    .input("args", sql.NVarChar(sql.MAX), canonicalJson(args))
    .input("requested", sql.DateTime2(7), requestedAtUtc)
    .input("captured", sql.DateTime2(7), capturedAtUtc)
    .input("result", sql.NVarChar(sql.MAX), resultJson)
    .input("result_hash", sql.Char(64), resultSha256)
    .input("rows", sql.Int, rows.length)
    .input("latency", sql.Decimal(18, 3), latencyMs)
    .query(`
      IF EXISTS
      (
        SELECT 1 FROM ingest.context_snapshots
        WHERE episode_id=@episode AND tool_id=@tool AND canonical_args_sha256=@args_hash
      ) THROW 51800, 'Context snapshot identity already exists', 1;
      INSERT ingest.context_snapshots
        (episode_id,snapshot_kind,tool_id,canonical_args_sha256,canonical_args_json,
         procedure_version,requested_at_utc,captured_at_utc,result_json,result_sha256,
         row_count,latency_ms,status)
      VALUES(@episode,@kind,@tool,@args_hash,@args,'scenario-snapshot-v1',@requested,
        @captured,@result,@result_hash,@rows,@latency,'complete');
    `);
  return {
    tool: input.plan.tool,
    arguments: args,
    argumentsSha256: argsSha256,
    resultSha256,
    rowCount: rows.length,
    latencyMs: Number(latencyMs.toFixed(3)),
    capturedAtUtc: capturedAtUtc.toISOString(),
  };
}

async function invokeSnapshotTool(
  pool: sql.ConnectionPool,
  name: ToolName,
  args: Record<string, unknown>,
): Promise<{ recordset: unknown[] }> {
  switch (name) {
    case "get_blocking_snapshot":
      return pool.request()
        .input("database_name", sql.NVarChar(128), String(args.databaseName))
        .input("max_rows", sql.Int, Number(args.maxRows))
        .execute("agent.usp_tool_get_blocking_snapshot");
    case "get_log_space":
      return pool.request()
        .input("database_name", sql.NVarChar(128), String(args.databaseName))
        .execute("agent.usp_tool_get_log_space");
    case "get_active_transactions":
      return pool.request()
        .input("database_name", sql.NVarChar(128), String(args.databaseName))
        .input("max_rows", sql.Int, Number(args.maxRows))
        .execute("agent.usp_tool_get_active_transactions");
    case "get_backup_history":
      return pool.request()
        .input("database_name", sql.NVarChar(128), String(args.databaseName))
        .input("max_rows", sql.Int, Number(args.maxRows))
        .execute("agent.usp_tool_get_backup_history");
    case "get_deadlock_graph":
      return pool.request()
        .input("max_rows", sql.Int, Number(args.maxRows))
        .execute("agent.usp_tool_get_deadlock_graph");
    case "get_recent_incident_counts":
    case "runbook_search":
      throw new Error(`${name} is computed during replay and must not be frozen as a context snapshot`);
  }
}

function requireDisposableDatabase(value: string | undefined): string {
  if (value === undefined) throw new Error("Snapshot plan requires a disposable database that was not created");
  return value;
}
