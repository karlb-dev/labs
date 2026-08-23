import { randomUUID } from "node:crypto";
import { appendFile, readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import type { SnapshotPlanItem } from "../src/context-snapshots.js";
import { canonicalJson, hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { scheduleDelayMs, scheduleOriginEpochMs } from "../src/schedule-timing.js";
import {
  executeScenarioInjector,
  isExpectedSignal,
  safeSqlError,
  type InjectorResult,
  type SafeSqlError,
} from "../src/scenario-injectors.js";

interface RunManifest { runId: string }
interface ScenarioConfig {
  snapshotPlan?: SnapshotPlanItem[];
  maxRuntimeSeconds?: number;
}
interface ScheduleRow {
  schedule_item_id: string;
  schedule_id: string;
  job_key: string;
  scenario_variant_id: string;
  ordinal: number;
  planned_offset_ms: string;
  scenario_id: string;
  injector_procedure: string;
  parameter_json: string;
  config_json: string;
  max_runtime_seconds: number;
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const scheduleName = argument("--schedule") ?? "smoke-v1";
const noWait = process.argv.includes("--no-wait");
const maxItems = Number(argument("--max-items") ?? "2147483647");
if (!Number.isInteger(maxItems) || maxItems < 1) throw new Error("--max-items must be a positive integer");
const control = await connect(config.databases.lab, config.databases.controlName);
const marker = await connect({
  ...config.databases.lab,
  options: { ...config.databases.lab.options, appName: "LogWarden-Marker" },
}, config.databases.workloadName);

const schedule = await control.request()
  .input("name", sql.VarChar(80), scheduleName)
  .query<ScheduleRow>(`
    SELECT item.schedule_item_id,item.schedule_id,item.job_key,item.scenario_variant_id,
           item.ordinal,item.planned_offset_ms,scenario.scenario_id,
           scenario.injector_procedure,variant.parameter_json,scenario.config_json,
           scenario.max_runtime_seconds
    FROM workload.schedule_items AS item
    INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=item.schedule_id
    INNER JOIN workload.scenario_variants AS variant ON variant.scenario_variant_id=item.scenario_variant_id
    INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
    WHERE schedule.schedule_name=@name
    ORDER BY item.ordinal;
  `);
if (schedule.recordset.length === 0) throw new Error(`Schedule not found or empty: ${scheduleName}`);

const priorExecution = await control.request()
  .input("run", sql.VarChar(120), run.runId)
  .input("schedule", sql.BigInt, schedule.recordset[0]!.schedule_id)
  .query<{ started_at_utc: Date; planned_offset_ms: string }>(`
    SELECT TOP (1) execution.started_at_utc,item.planned_offset_ms
    FROM workload.injection_executions AS execution
    INNER JOIN workload.schedule_items AS item ON item.schedule_item_id=execution.schedule_item_id
    WHERE execution.run_id=@run AND item.schedule_id=@schedule
    ORDER BY item.ordinal;
  `);
const checkpoint = priorExecution.recordset[0];
const scheduleOriginMs = scheduleOriginEpochMs(checkpoint === undefined ? undefined : {
  plannedOffsetMs: Number(checkpoint.planned_offset_ms),
  startedAtUtc: checkpoint.started_at_utc,
}, Date.now());
const resumed = checkpoint !== undefined;

const summary: Array<Record<string, unknown>> = [];
try {
  for (const item of schedule.recordset.slice(0, maxItems)) {
    const injector = item.injector_procedure.replace(/^driver:/, "");
    const existing = await control.request()
      .input("run", sql.VarChar(120), run.runId)
      .input("item", sql.BigInt, item.schedule_item_id)
      .query<{ injection_execution_id: string; verified: boolean; return_code: number | null; error_detail: string | null }>(`
        SELECT injection_execution_id,verified,return_code,error_detail
        FROM workload.injection_executions
        WHERE run_id=@run AND schedule_item_id=@item;
      `);
    if (existing.recordset.length > 0) {
      const prior = existing.recordset[0]!;
      const priorError = parsePriorError(prior.error_detail);
      if (prior.return_code !== 0 && priorError !== undefined && isExpectedSignal(injector, priorError)) {
        await control.request()
          .input("id", sql.BigInt, prior.injection_execution_id)
          .query("UPDATE workload.injection_executions SET return_code=0,error_detail=NULL WHERE injection_execution_id=@id;");
        summary.push({ scheduleItemId: item.schedule_item_id, status: "driver_signal_reclassified", verified: prior.verified });
        continue;
      }
      summary.push({ scheduleItemId: item.schedule_item_id, status: "already_injected", verified: prior.verified });
      continue;
    }
    if (!noWait) {
      const remaining = scheduleDelayMs(scheduleOriginMs, Number(item.planned_offset_ms), Date.now());
      if (remaining > 0) await delay(remaining);
    }
    const episodeId = `lw-${scheduleName}-${String(item.ordinal).padStart(4, "0")}-${item.job_key.slice(0, 8)}`;
    const correlationToken = randomUUID().replaceAll("-", "");
    const parameters = parseRecord(item.parameter_json, "variant parameters");
    const scenarioConfig = JSON.parse(item.config_json) as ScenarioConfig;
    const snapshotPlan = scenarioConfig.snapshotPlan ?? [];
    const startUtc = new Date();
    await writeMarker(marker, episodeId, injector, correlationToken, "start", {
      phase: "start",
      runId: run.runId,
      scenarioVariantId: item.scenario_variant_id,
    });

    let execution: InjectorResult;
    try {
      execution = await executeScenarioInjector({
        config,
        control,
        runId: run.runId,
        episodeId,
        injector,
        correlationToken,
        parameters,
        snapshotPlan,
        maxRuntimeSeconds: item.max_runtime_seconds,
      });
    } catch (error) {
      execution = {
        sessionIds: [],
        expectedSignalCount: Number(parameters.signalCount ?? 1),
        observedSignalCount: 0,
        observedErrors: [],
        snapshots: [],
        cleanupVerified: false,
        unexpectedError: safeSqlError(error),
      };
    }

    let resetVerified = false;
    try {
      await marker.request().execute("lab.usp_reset_seed");
      resetVerified = true;
    } catch (error) {
      execution.unexpectedError ??= safeSqlError(error);
    }
    const cleanupVerified = execution.cleanupVerified && resetVerified;
    if (!cleanupVerified && execution.unexpectedError === undefined)
      execution.unexpectedError = { message: "Injector cleanup or seed-reset postcondition failed" };
    const outcome = execution.unexpectedError === undefined
      ? execution.observedErrors.length > 0 ? "expected_signal" : "success"
      : "unexpected_failure";
    const finishUtc = new Date();
    await writeMarker(marker, episodeId, injector, correlationToken, "end", {
      phase: "end",
      outcome,
      cleanupVerified,
      expectedSignalCount: execution.expectedSignalCount,
      observedSignalCount: execution.observedSignalCount,
      snapshotCount: execution.snapshots.length,
    });

    const sessionIds = [...new Set(execution.sessionIds)].sort((left, right) => left - right);
    const requestManifest = {
      schemaVersion: 2,
      episodeId,
      correlationToken,
      injector,
      scenarioId: item.scenario_id,
      scenarioVariantId: item.scenario_variant_id,
      parameters,
      parametersSha256: hashJson(parameters),
      sessionIds,
      expectedSignalCount: execution.expectedSignalCount,
      observedSignalCount: execution.observedSignalCount,
      observedErrors: execution.observedErrors,
      snapshots: execution.snapshots,
      ...(execution.disposableDatabase === undefined ? {} : { disposableDatabase: execution.disposableDatabase }),
    };
    const inserted = await control.request()
      .input("run", sql.VarChar(120), run.runId)
      .input("item", sql.BigInt, item.schedule_item_id)
      .input("episode", sql.VarChar(120), episodeId)
      .input("disposable", sql.BigInt, execution.disposableDatabase?.id ?? null)
      .input("request", sql.NVarChar(sql.MAX), canonicalJson(requestManifest))
      .input("started", sql.DateTime2(7), startUtc)
      .input("finished", sql.DateTime2(7), finishUtc)
      .input("sessions", sql.NVarChar(sql.MAX), canonicalJson(sessionIds))
      .input("return_code", sql.Int, outcome === "unexpected_failure" ? 1 : 0)
      .input("cleanup", sql.Bit, cleanupVerified)
      .input("error", sql.NVarChar(sql.MAX), execution.unexpectedError === undefined ? null : canonicalJson(execution.unexpectedError))
      .query<{ injection_execution_id: string }>(`
        INSERT workload.injection_executions
          (run_id,schedule_item_id,episode_id,disposable_database_id,injector_request_json,
           started_at_utc,finished_at_utc,sql_session_ids_json,return_code,verified,
           cleanup_verified,error_detail)
        OUTPUT INSERTED.injection_execution_id
        VALUES(@run,@item,@episode,@disposable,@request,@started,@finished,@sessions,
          @return_code,0,@cleanup,@error);
      `);
    const terminal = {
      scheduleItemId: item.schedule_item_id,
      injectionExecutionId: inserted.recordset[0]!.injection_execution_id,
      episodeId,
      injector,
      outcome,
      cleanupVerified,
      sessionCount: sessionIds.length,
      expectedSignalCount: execution.expectedSignalCount,
      observedSignalCount: execution.observedSignalCount,
      snapshotCount: execution.snapshots.length,
      startedAtUtc: startUtc.toISOString(),
      finishedAtUtc: finishUtc.toISOString(),
    };
    await appendFile(`${runDirectory}/capture/injection-journal.jsonl`, `${canonicalJson({
      schemaVersion: 2,
      eventId: randomUUID(),
      eventType: "injection_terminal",
      atUtc: new Date().toISOString(),
      payload: terminal,
      payloadSha256: hashJson(terminal),
    })}\n`, { encoding: "utf8", mode: 0o600 });
    summary.push(terminal);
    if (execution.unexpectedError !== undefined)
      throw new Error(`Injector ${injector} failed for ${episodeId}: ${execution.unexpectedError.message}`);
  }
} finally {
  await Promise.allSettled([control.close(), marker.close()]);
}

const receiptBody = {
  schemaVersion: 2,
  runId: run.runId,
  scheduleName,
  noWait,
  resumed,
  scheduleOriginUtc: new Date(scheduleOriginMs).toISOString(),
  episodes: summary,
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
const receiptName = scheduleName === "smoke-v1" ? "injection-summary.json" : `injection-summary-${safeName(scheduleName)}.json`;
await atomicWrite(`${runDirectory}/capture/${receiptName}`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({
  runId: run.runId,
  scheduleName,
  count: summary.length,
  receiptPath: `capture/${receiptName}`,
  receiptSha256: receipt.receiptSha256,
}, null, 2));

async function writeMarker(
  pool: sql.ConnectionPool,
  episodeId: string,
  injector: string,
  token: string,
  phase: "start" | "end",
  detail: Record<string, unknown>,
): Promise<void> {
  await pool.request()
    .input("episode", sql.VarChar(120), episodeId)
    .input("injector", sql.VarChar(100), injector)
    .input("token", sql.UniqueIdentifier, formatUuid(token))
    .input("phase", sql.VarChar(32), phase)
    .input("detail", sql.NVarChar(sql.MAX), canonicalJson(detail))
    .query(`
      INSERT lab.execution_markers(episode_id,injector_id,opaque_token,phase,detail_json)
      VALUES(@episode,@injector,@token,@phase,@detail);
    `);
}

function parseRecord(value: string, label: string): Record<string, unknown> {
  const parsed = JSON.parse(value) as unknown;
  if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object")
    throw new Error(`${label} must be a JSON object`);
  return parsed as Record<string, unknown>;
}

function parsePriorError(value: string | null): SafeSqlError | undefined {
  if (value === null) return undefined;
  try {
    const parsed = JSON.parse(value) as SafeSqlError;
    return typeof parsed.message === "string" ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function formatUuid(compact: string): string {
  return `${compact.slice(0, 8)}-${compact.slice(8, 12)}-${compact.slice(12, 16)}-${compact.slice(16, 20)}-${compact.slice(20, 32)}`;
}

function safeName(value: string): string {
  if (!/^[a-z0-9-]{1,80}$/i.test(value)) throw new Error(`Unsafe schedule name: ${value}`);
  return value.toLowerCase();
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
