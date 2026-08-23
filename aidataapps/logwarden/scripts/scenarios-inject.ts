import { randomUUID } from "node:crypto";
import { appendFile, readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

interface RunManifest { runId: string }
interface ScheduleRow {
  schedule_item_id: string;
  schedule_id: string;
  job_key: string;
  scenario_variant_id: string;
  ordinal: number;
  planned_offset_ms: string;
  scenario_id: string;
  injector_procedure: string;
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const scheduleName = argument("--schedule") ?? "smoke-v1";
const noWait = process.argv.includes("--no-wait");
const maxItems = Number(argument("--max-items") ?? "2147483647");
const control = await connect(config.databases.lab, config.databases.controlName);
const marker = await connect({
  ...config.databases.lab,
  options: { ...config.databases.lab.options, appName: "LogWarden-Marker" },
}, config.databases.workloadName);
const inject = await connect({
  ...config.databases.lab,
  options: { ...config.databases.lab.options, appName: "LogWarden-Inject" },
}, config.databases.workloadName);

const schedule = await control.request()
  .input("name", sql.VarChar(80), scheduleName)
  .query<ScheduleRow>(`
    SELECT i.schedule_item_id, i.schedule_id, i.job_key, i.scenario_variant_id,
           i.ordinal, i.planned_offset_ms, s.scenario_id, s.injector_procedure
    FROM workload.schedule_items AS i
    INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=i.schedule_id
    INNER JOIN workload.scenario_variants AS v ON v.scenario_variant_id=i.scenario_variant_id
    INNER JOIN workload.scenario_definitions AS s ON s.scenario_id=v.scenario_id
    WHERE schedule.schedule_name=@name
    ORDER BY i.ordinal;
  `);
if (schedule.recordset.length === 0) throw new Error(`Schedule not found or empty: ${scheduleName}`);

const summary: Array<Record<string, unknown>> = [];
const started = performance.now();
try {
  for (const item of schedule.recordset.slice(0, maxItems)) {
    const injector = item.injector_procedure.replace(/^driver:/, "");
    const existing = await control.request()
      .input("run", sql.VarChar(120), run.runId)
      .input("item", sql.BigInt, item.schedule_item_id)
      .query<{ injection_execution_id: string; verified: boolean; return_code: number | null; error_detail: string | null }>(`
        SELECT injection_execution_id, verified, return_code, error_detail FROM workload.injection_executions
        WHERE run_id=@run AND schedule_item_id=@item;
      `);
    if (existing.recordset.length > 0) {
      const prior = existing.recordset[0]!;
      const priorError = prior.error_detail === null ? undefined : JSON.parse(prior.error_detail) as { number?: number; message: string };
      if (prior.return_code !== 0 && priorError !== undefined && isExpectedSignal(injector, priorError)) {
        await control.request()
          .input("id", sql.BigInt, prior.injection_execution_id)
          .query("UPDATE workload.injection_executions SET return_code=0,error_detail=NULL WHERE injection_execution_id=@id;");
        summary.push({ scheduleItemId: item.schedule_item_id, status: "driver_signal_reclassified", verified: prior.verified });
        continue;
      }
      summary.push({ scheduleItemId: item.schedule_item_id, status: "already_injected", verified: existing.recordset[0]!.verified });
      continue;
    }
    if (!noWait) {
      const remaining = Number(item.planned_offset_ms) - (performance.now() - started);
      if (remaining > 0) await delay(remaining);
    }
    const episodeId = `lw-${scheduleName}-${String(item.ordinal).padStart(4, "0")}-${item.job_key.slice(0, 8)}`;
    const correlationToken = randomUUID().replaceAll("-", "");
    const startUtc = new Date();
    const spid = await inject.request().query<{ spid: number }>("SELECT @@SPID AS spid;");
    const sessionId = spid.recordset[0]!.spid;
    const requestManifest = { schemaVersion: 1, episodeId, correlationToken, injector, sessionIds: [sessionId] };
    await marker.request()
      .input("episode", sql.VarChar(120), episodeId)
      .input("injector", sql.VarChar(100), injector)
      .input("token", sql.UniqueIdentifier, formatUuid(correlationToken))
      .input("detail", sql.NVarChar(sql.MAX), canonicalJson({ phase: "start", runId: run.runId }))
      .query(`
        INSERT lab.execution_markers(episode_id,injector_id,opaque_token,phase,detail_json)
        VALUES(@episode,@injector,@token,'start',@detail);
      `);

    let outcome: "expected_signal" | "success" | "unexpected_failure" = "success";
    let observedError: { number?: number; message: string } | undefined;
    try {
      await executeInjector(injector, inject, episodeId, correlationToken);
    } catch (error) {
      observedError = safeError(error);
      outcome = isExpectedSignal(injector, observedError) ? "expected_signal" : "unexpected_failure";
    }
    if (expectedErrorNumbers(injector).length > 0 && observedError === undefined) outcome = "unexpected_failure";

    let cleanupVerified = false;
    try {
      await marker.request().execute("lab.usp_reset_seed");
      cleanupVerified = true;
    } catch (error) {
      observedError ??= safeError(error);
      outcome = "unexpected_failure";
    }
    const finishUtc = new Date();
    await marker.request()
      .input("episode", sql.VarChar(120), episodeId)
      .input("injector", sql.VarChar(100), injector)
      .input("token", sql.UniqueIdentifier, formatUuid(correlationToken))
      .input("detail", sql.NVarChar(sql.MAX), canonicalJson({ phase: "end", outcome, cleanupVerified }))
      .query(`
        INSERT lab.execution_markers(episode_id,injector_id,opaque_token,phase,detail_json)
        VALUES(@episode,@injector,@token,'end',@detail);
      `);

    const inserted = await control.request()
      .input("run", sql.VarChar(120), run.runId)
      .input("item", sql.BigInt, item.schedule_item_id)
      .input("episode", sql.VarChar(120), episodeId)
      .input("request", sql.NVarChar(sql.MAX), canonicalJson(requestManifest))
      .input("started", sql.DateTime2(7), startUtc)
      .input("finished", sql.DateTime2(7), finishUtc)
      .input("sessions", sql.NVarChar(sql.MAX), canonicalJson([sessionId]))
      .input("return_code", sql.Int, outcome === "unexpected_failure" ? 1 : 0)
      .input("cleanup", sql.Bit, cleanupVerified)
      .input("error", sql.NVarChar(sql.MAX), outcome === "unexpected_failure" ? canonicalJson(observedError ?? { message: "expected signal was not observed by driver" }) : null)
      .query<{ injection_execution_id: string }>(`
        INSERT workload.injection_executions
          (run_id,schedule_item_id,episode_id,injector_request_json,started_at_utc,
           finished_at_utc,sql_session_ids_json,return_code,verified,cleanup_verified,error_detail)
        OUTPUT INSERTED.injection_execution_id
        VALUES(@run,@item,@episode,@request,@started,@finished,@sessions,@return_code,0,@cleanup,@error);
      `);
    const result = {
      scheduleItemId: item.schedule_item_id,
      injectionExecutionId: inserted.recordset[0]!.injection_execution_id,
      episodeId,
      injector,
      outcome,
      cleanupVerified,
      startedAtUtc: startUtc.toISOString(),
      finishedAtUtc: finishUtc.toISOString(),
    };
    await appendFile(`${runDirectory}/capture/injection-journal.jsonl`, `${canonicalJson({
      schemaVersion: 1, eventId: randomUUID(), eventType: "injection_terminal",
      atUtc: new Date().toISOString(), payload: result, payloadSha256: hashJson(result),
    })}\n`, { encoding: "utf8", mode: 0o600 });
    summary.push(result);
    if (outcome === "unexpected_failure") throw new Error(`Injector ${injector} failed for ${episodeId}`);
  }
} finally {
  await Promise.allSettled([control.close(), marker.close(), inject.close()]);
}

const receiptBody = { schemaVersion: 1, runId: run.runId, scheduleName, noWait, episodes: summary };
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/capture/injection-summary.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({ runId: run.runId, scheduleName, count: summary.length, receiptSha256: receipt.receiptSha256 }, null, 2));

async function executeInjector(name: string, pool: sql.ConnectionPool, episodeId: string, token: string): Promise<void> {
  const marker = `LW_EVT_${token}`;
  switch (name) {
    case "conversion_error":
      await pool.request().input("value", sql.NVarChar(200), `not_an_int_${marker}`).query("/* LW:inject.conversion */ SELECT CONVERT(int,@value);");
      return;
    case "missing_object":
      await pool.request().query(`/* LW:inject.missing_object */ SELECT TOP (1) * FROM lab.LW_Missing_${token.slice(0, 16)};`);
      return;
    case "duplicate_key":
      await pool.request().input("marker", sql.VarChar(40), marker.slice(0, 40)).query("/* LW:inject.duplicate */ INSERT lab.accounts(account_id,opaque_code,amount) VALUES(1,@marker,1.00);");
      return;
    case "truncation_error":
      await pool.request().query("/* LW:inject.truncation */ UPDATE lab.accounts SET short_value=REPLICATE('x',100) WHERE account_id=1;");
      return;
    case "controlled_signal":
      await pool.request().input("message", sql.NVarChar(1000), `${marker}: integrity verification warning`).input("severity", sql.Int, 16).input("state", sql.Int, 41).execute("lab.usp_raise_controlled_signal");
      return;
    case "ambiguous_signal":
      await pool.request().input("message", sql.NVarChar(1000), `${marker}: activity may require review; evidence is incomplete`).input("severity", sql.Int, 11).input("state", sql.Int, 42).execute("lab.usp_raise_controlled_signal");
      return;
    case "bad_login": {
      const invalid = new sql.ConnectionPool({
        ...config.databases.lab,
        user: `lw_missing_${token.slice(0, 12)}`,
        password: randomUUID(),
        database: config.databases.controlName,
        connectionTimeout: 5_000,
        options: { ...config.databases.lab.options, appName: `LogWarden-Inject-Auth-${token.slice(0, 8)}` },
      });
      try { await invalid.connect(); }
      finally { await invalid.close().catch(() => undefined); }
      return;
    }
    case "backup_failure":
      await pool.request().input("destination", sql.NVarChar(500), `/root/logwarden-forbidden/${marker}.bak`).query("/* LW:inject.backup_failure */ BACKUP DATABASE [LogWardenWorkload] TO DISK=@destination WITH COPY_ONLY;");
      return;
    case "query_pressure":
      await pool.request().query("/* LW:inject.query_pressure */ SELECT SUM(CONVERT(bigint,a.value)*CONVERT(bigint,b.value)) AS bounded_work FROM GENERATE_SERIES(1,1000) AS a CROSS JOIN GENERATE_SERIES(1,1000) AS b OPTION(MAXDOP 1);");
      return;
    case "benign_noise":
      await pool.request().input("value", sql.NVarChar(400), `${episodeId}:${marker}`).query("/* LW:inject.benign */ INSERT lab.noise_rows(opaque_value) VALUES(@value); SELECT COUNT(*) AS row_count FROM lab.noise_rows;");
      return;
    default:
      throw new Error(`Unsupported injector: ${name}`);
  }
}

function expectedErrorNumbers(name: string): number[] {
  const mapping: Record<string, number[]> = {
    conversion_error: [245], missing_object: [208], duplicate_key: [2627], truncation_error: [2628, 8152],
    controlled_signal: [50000], ambiguous_signal: [50000], bad_login: [18456], backup_failure: [3201, 3013],
    query_pressure: [], benign_noise: [],
  };
  return mapping[name] ?? [];
}

function isExpectedSignal(name: string, error: { number?: number; message: string }): boolean {
  return expectedErrorNumbers(name).includes(error.number ?? -1) ||
    (name === "bad_login" && /login failed/i.test(error.message));
}

function safeError(error: unknown): { number?: number; message: string } {
  const candidate = error as { number?: number; message?: string };
  return { ...(candidate.number === undefined ? {} : { number: candidate.number }), message: candidate.message ?? String(error) };
}

function formatUuid(compact: string): string {
  return `${compact.slice(0, 8)}-${compact.slice(8, 12)}-${compact.slice(12, 16)}-${compact.slice(16, 20)}-${compact.slice(20, 32)}`;
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
