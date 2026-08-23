import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { stat, statfs, writeFile } from "node:fs/promises";
import os from "node:os";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { parsePrometheusExposition, sumPrometheusMetric, type PrometheusSample } from "../src/prometheus.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { createComponentTelemetryJournal } from "../src/telemetry.js";

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await import("node:fs/promises").then(({ readFile }) => readFile(`${runDirectory}/run.json`, "utf8"))) as { runId: string };
const phase = argument("--phase") ?? "development";
const watch = process.argv.includes("--watch");
const epochId = randomUUID();
const { journal, path: journalPath } = await createComponentTelemetryJournal(runDirectory, run.runId, "systems-sampler", epochId);
const stopController = new AbortController();
let stopSignal: NodeJS.Signals | null = null;
const requestStop = (signal: NodeJS.Signals): void => {
  stopSignal ??= signal;
  stopController.abort();
};
const onSigint = (): void => requestStop("SIGINT");
const onSigterm = (): void => requestStop("SIGTERM");
process.on("SIGINT", onSigint);
process.on("SIGTERM", onSigterm);

try {
  do {
    // A stop request lets an in-flight sample finish so its durable span always
    // receives a matching end record. The abort signal only wakes the idle delay.
    await sampleOnce();
    if (!watch || stopSignal !== null) break;
    await delay(config.telemetrySampleSeconds * 1000, stopController.signal);
  } while (stopSignal === null);
  if (stopSignal !== null) {
    await journal.record("point", "sampler.shutdown", {}, { epochId, phase, signal: stopSignal, disposition: "GRACEFUL" });
    await journal.flush();
  }
} finally {
  process.off("SIGINT", onSigint);
  process.off("SIGTERM", onSigterm);
}

async function sampleOnce(): Promise<void> {
  const sampledAt = new Date();
  const stamp = sampledAt.toISOString().replaceAll(/[-:.]/g, "");
  const journalSpan = await journal.startSpan("sampler.sample", {}, { epochId, phase });
  const rawDirectory = `${runDirectory}/telemetry/raw/${stamp}`;
  const pool = await connect({
    ...config.databases.lab,
    options: { ...config.databases.lab.options, appName: "LogWarden-Telemetry" },
  }, config.databases.controlName);
  const receipt: Record<string, unknown> = { schemaVersion: 1, runId: run.runId, epochId, phase, journalPath, sampledAtUtc: sampledAt.toISOString() };
  try {
    const [gpu, sqlSample, queue, xe, host] = await Promise.all([
      collectGpu(), collectSql(pool), collectQueue(pool), collectXe(pool), collectHost(),
    ]);
    receipt.gpu = gpu;
    receipt.sql = sqlSample;
    receipt.queue = queue;
    receipt.xe = xe;
    receipt.host = host;
    await persistGpu(pool, sampledAt, gpu);
    await persistSql(pool, sampledAt, sqlSample);
    await persistQueue(pool, sampledAt, queue);
    await persistXe(pool, sampledAt, xe);
    await persistHost(pool, sampledAt, host);

    const services = [
      { kind: "chat", instance: "chat-primary", baseUrl: config.inference.chatBaseUrl },
      { kind: "embedding", instance: "embedding-qwen", baseUrl: config.inference.qwenEmbeddingBaseUrl },
      { kind: "embedding", instance: "embedding-bge", baseUrl: config.inference.bgeEmbeddingBaseUrl },
    ];
    const serviceResults = [];
    for (const service of services) serviceResults.push(await samplePrometheus(pool, sampledAt, rawDirectory, service));
    receipt.modelServices = serviceResults;
    await atomicWrite(`${rawDirectory}/system.json`, `${JSON.stringify({ gpu, sql: sqlSample, queue, xe, host }, null, 2)}\n`);
    await journal.record("metric", "sampler.sample.persisted", {}, {
      epochId, phase, sampleKey: hashJson({ runId: run.runId, epochId, sampledAtUtc: sampledAt.toISOString() }),
      services: serviceResults.map((result) => ({ instance: result.instance, status: result.status, metricCount: result.metricCount })),
    });
    await journal.endSpan(journalSpan, "ok", { sourceCount: 8 });
  } catch (error) {
    await journal.endSpan(journalSpan, "error", { error: safeError(error) });
    throw error;
  } finally {
    await pool.close();
    await journal.flush();
  }
  const receiptWithHash = { ...receipt, receiptSha256: hashJson(receipt) };
  await atomicWrite(`${rawDirectory}/receipt.json`, `${JSON.stringify(receiptWithHash, null, 2)}\n`);
  console.log(JSON.stringify({ sampledAtUtc: sampledAt.toISOString(), epochId, phase, receiptSha256: receiptWithHash.receiptSha256 }, null, 2));
}

async function samplePrometheus(
  pool: sql.ConnectionPool,
  sampledAt: Date,
  rawDirectory: string,
  service: { kind: string; instance: string; baseUrl: string },
): Promise<{ instance: string; status: string; metricCount: number; discoveredMetricNames: string[]; rawSnapshotId: string }> {
  const endpoint = new URL(service.baseUrl);
  endpoint.pathname = "/metrics";
  endpoint.search = "";
  const started = performance.now();
  let payload: string | null = null;
  let httpStatus: number | null = null;
  let contentType: string | null = null;
  let errorDetail: string | null = null;
  try {
    const response = await fetch(endpoint, { signal: AbortSignal.timeout(5_000) });
    httpStatus = response.status;
    contentType = response.headers.get("content-type");
    payload = await response.text();
    if (!response.ok) errorDetail = `HTTP ${response.status}`;
  } catch (error) {
    errorDetail = safeError(error).message;
  }
  const latencyMs = round(performance.now() - started);
  const parsed = payload === null ? { samples: [] as PrometheusSample[], metricNames: [] as string[] } : parsePrometheusExposition(payload);
  const snapshotKey = hashJson({ runId: run.runId, instance: service.instance, sampledAtUtc: sampledAt.toISOString() });
  if (payload !== null) await atomicWrite(`${rawDirectory}/${service.instance}.prom`, payload);
  const inserted = await pool.request()
    .input("key", sql.Char(64), snapshotKey)
    .input("run", sql.VarChar(120), run.runId)
    .input("kind", sql.VarChar(40), `vllm_${service.kind}`)
    .input("instance", sql.VarChar(160), service.instance)
    .input("endpoint", sql.NVarChar(1000), endpoint.toString())
    .input("phase", sql.VarChar(40), phase)
    .input("status", sql.Int, httpStatus)
    .input("content_type", sql.VarChar(160), contentType)
    .input("payload", sql.VarBinary(sql.MAX), payload === null ? null : Buffer.from(payload))
    .input("hash", sql.Char(64), payload === null ? null : sha256(payload))
    .input("bytes", sql.BigInt, payload === null ? null : Buffer.byteLength(payload))
    .input("names", sql.NVarChar(sql.MAX), canonicalJson(parsed.metricNames))
    .input("at", sql.DateTime2(7), sampledAt)
    .input("latency", sql.Decimal(18, 3), latencyMs)
    .input("snapshot_status", sql.VarChar(24), errorDetail === null && httpStatus !== null && httpStatus >= 200 && httpStatus < 300 ? "available" : "unavailable")
    .input("error", sql.NVarChar(sql.MAX), errorDetail)
    .query<{ raw_metric_snapshot_id: string }>(`
      INSERT telemetry.raw_metric_snapshots
        (snapshot_key,run_id,source_kind,source_instance,endpoint_uri,phase,http_status,
         content_type,payload,payload_sha256,payload_bytes,discovered_metric_names_json,
         observed_at_utc,latency_ms,status,error_detail)
      OUTPUT INSERTED.raw_metric_snapshot_id
      VALUES(@key,@run,@kind,@instance,@endpoint,@phase,@status,@content_type,@payload,
        @hash,@bytes,@names,@at,@latency,@snapshot_status,@error);
    `);
  const rawSnapshotId = inserted.recordset[0]!.raw_metric_snapshot_id;
  for (const sample of parsed.samples) {
    const finite = Number.isFinite(sample.value);
    const sampleKey = hashJson({ snapshotKey, name: sample.name, labels: sample.labels });
    await pool.request()
      .input("key", sql.Char(64), sampleKey)
      .input("run", sql.VarChar(120), run.runId)
      .input("kind", sql.VarChar(40), `vllm_${service.kind}`)
      .input("instance", sql.VarChar(160), service.instance)
      .input("name", sql.NVarChar(300), sample.name)
      .input("type", sql.VarChar(24), sample.type)
      .input("value", sql.Float, finite ? sample.value : null)
      .input("labels", sql.NVarChar(sql.MAX), canonicalJson(sample.labels))
      .input("at", sql.DateTime2(7), sampledAt)
      .input("unavailable", sql.NVarChar(500), finite ? null : "non-finite Prometheus value")
      .query(`
        IF NOT EXISTS (SELECT 1 FROM telemetry.metric_samples WHERE sample_key=@key)
          INSERT telemetry.metric_samples
            (sample_key,run_id,source_kind,source_instance,metric_name,metric_type,
             metric_value,provenance,labels_json,observed_at_utc,unavailable_reason)
          VALUES(@key,@run,@kind,@instance,@name,@type,@value,'raw_service',@labels,@at,@unavailable);
      `);
  }
  const modelAvailability = modelFields(parsed.samples);
  const modelProfileId = service.kind === "chat" && errorDetail === null
    ? process.env.MODEL_PROFILE ?? null
    : null;
  await pool.request()
    .input("key", sql.Char(64), snapshotKey)
    .input("run", sql.VarChar(120), run.runId)
    .input("at", sql.DateTime2(7), sampledAt)
    .input("raw", sql.BigInt, rawSnapshotId)
    .input("instance", sql.VarChar(160), service.instance)
    .input("phase", sql.VarChar(40), phase)
    .input("profile", sql.VarChar(80), modelProfileId)
    .input("running", sql.Float, modelAvailability.runningRequests)
    .input("waiting", sql.Float, modelAvailability.waitingRequests)
    .input("swapped", sql.Float, modelAvailability.swappedRequests)
    .input("prompt_total", sql.Float, modelAvailability.promptTokensTotal)
    .input("generation_total", sql.Float, modelAvailability.generationTokensTotal)
    .input("prompt_rate", sql.Float, modelAvailability.promptThroughput)
    .input("generation_rate", sql.Float, modelAvailability.generationThroughput)
    .input("kv", sql.Float, modelAvailability.kvCacheUsageRatio)
    .input("prefix_hits", sql.Float, modelAvailability.prefixCacheHitsTotal)
    .input("prefix_queries", sql.Float, modelAvailability.prefixCacheQueriesTotal)
    .input("preemptions", sql.Float, modelAvailability.preemptionsTotal)
    .input("errors", sql.Float, modelAvailability.requestErrorsTotal)
    .input("cancellations", sql.Float, modelAvailability.cancellationsTotal)
    .input("json", sql.NVarChar(sql.MAX), canonicalJson({ discoveredMetricNames: parsed.metricNames, mappedFields: modelAvailability }))
    .query(`
      INSERT telemetry.model_service_samples
        (sample_key,run_id,source_instance,phase,model_profile_id,sampled_at_utc,
         running_requests,waiting_requests,swapped_requests,
         prompt_tokens_total,generation_tokens_total,prompt_throughput,generation_throughput,
         kv_cache_usage_ratio,prefix_cache_hits_total,prefix_cache_queries_total,
         preemptions_total,request_errors_total,cancellations_total,raw_metric_snapshot_id,availability_json)
      VALUES(@key,@run,@instance,@phase,@profile,@at,@running,@waiting,@swapped,@prompt_total,@generation_total,
        @prompt_rate,@generation_rate,@kv,@prefix_hits,@prefix_queries,@preemptions,
        @errors,@cancellations,@raw,@json);
    `);
  return { instance: service.instance, status: errorDetail === null && httpStatus !== null && httpStatus >= 200 && httpStatus < 300 ? "available" : "unavailable", metricCount: parsed.samples.length, discoveredMetricNames: parsed.metricNames, rawSnapshotId };
}

async function collectGpu(): Promise<Record<string, unknown>> {
  try {
    const output = await commandOutput("nvidia-smi", ["--query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,pstate", "--format=csv,noheader,nounits"]);
    const values = output.trim().split(",").map((value) => value.trim());
    const processes = await commandOutput("nvidia-smi", ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]).catch(() => "");
    return { status: "available", uuid: values[0], utilizationGpuPct: numberOrNull(values[1]), utilizationMemoryPct: numberOrNull(values[2]), memoryUsedMiB: numberOrNull(values[3]), memoryTotalMiB: numberOrNull(values[4]), powerDrawW: numberOrNull(values[5]), powerLimitW: numberOrNull(values[6]), temperatureC: numberOrNull(values[7]), smClockMHz: numberOrNull(values[8]), memoryClockMHz: numberOrNull(values[9]), pstate: values[10] ?? null, processes: processes.trim().split("\n").filter(Boolean) };
  } catch (error) { return { status: "unavailable", error: safeError(error).message, processes: [] }; }
}

async function collectSql(pool: sql.ConnectionPool): Promise<Record<string, unknown>> {
  const result = await pool.request().query(`
    SELECT
      (SELECT TOP (1) physical_memory_in_use_kb FROM sys.dm_os_process_memory) AS process_memory_kb,
      (SELECT TOP (1) committed_target_kb FROM sys.dm_os_sys_info) AS target_memory_kb,
      (SELECT COUNT(*) FROM sys.dm_exec_requests WHERE session_id<>@@SPID) AS request_count,
      (SELECT COUNT(*) FROM sys.dm_exec_requests WHERE blocking_session_id<>0) AS blocked_request_count,
      (SELECT COUNT(*) FROM sys.dm_os_tasks WHERE task_state='RUNNABLE') AS runnable_task_count,
      (SELECT COUNT(*) FROM sys.dm_io_pending_io_requests) AS pending_io_count,
      (SELECT SUM(CONVERT(bigint,size))*8192 FROM sys.master_files WHERE database_id IN (DB_ID(N'LogWardenControl'),DB_ID(N'LogWardenWorkload')) AND type=0) AS data_file_bytes,
      (SELECT SUM(CONVERT(bigint,size))*8192 FROM sys.master_files WHERE database_id IN (DB_ID(N'LogWardenControl'),DB_ID(N'LogWardenWorkload')) AND type=1) AS log_file_bytes;
  `);
  return { status: "available", ...result.recordset[0], waits: [], io: [] };
}

async function collectQueue(pool: sql.ConnectionPool): Promise<Record<string, unknown>> {
  const result = await pool.request().query(`
    SELECT
      SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending_count,
      SUM(CASE WHEN status IN ('leased','packet_loaded','model_requested','tool_requested','tool_completed','decision_received','validated','persisted','proposed_action_recorded') THEN 1 ELSE 0 END) AS leased_count,
      SUM(CASE WHEN status='retryable_failure' THEN 1 ELSE 0 END) AS retryable_count,
      SUM(CASE WHEN status IN ('complete','contract_rejected','policy_rejected','model_timeout','tool_timeout','stopped') THEN 1 ELSE 0 END) AS terminal_count,
      DATEDIFF_BIG(MILLISECOND,MIN(CASE WHEN status='pending' THEN created_at_utc END),SYSUTCDATETIME()) AS oldest_pending_age_ms,
      SUM(CASE WHEN leased_until_utc<SYSUTCDATETIME() AND status NOT IN ('complete','contract_rejected','policy_rejected','model_timeout','tool_timeout','stopped') THEN 1 ELSE 0 END) AS lease_expired_count,
      COUNT(DISTINCT CASE WHEN lease_owner IS NOT NULL THEN lease_owner END) AS worker_count
    FROM ops.work_items;
  `);
  return { status: "available", ...normalizeCounts(result.recordset[0] ?? {}) };
}

async function collectXe(pool: sql.ConnectionPool): Promise<Record<string, unknown>> {
  const result = await pool.request().query(`
    SELECT c.file_name,c.file_offset,c.watermark_utc,
      (SELECT MAX(source_timestamp_utc) FROM ingest.raw_events WHERE source_id='xe-logwarden-capture') AS newest_event_at_utc,
      (SELECT TOP (1) rows_inserted FROM ingest.ingestion_batches WHERE source_id='xe-logwarden-capture' ORDER BY ingestion_batch_id DESC) AS batch_rows,
      (SELECT COUNT(*) FROM ingest.raw_events WHERE source_id='xe-logwarden-capture' AND parse_status<>'parsed') AS parse_errors_total,
      runtime.dropped_event_count AS dropped_events_total,
      runtime.dropped_buffer_count AS dropped_buffers_total,
      runtime.blocked_event_fire_time AS blocked_event_fire_time_ms,
      target.failed_buffer_count AS failed_target_buffers_total,
      target.bytes_written AS source_bytes,
      target.target_xml.value('count((/EventFileTarget/File))','int') AS rollover_files
    FROM ingest.source_cursors AS c
    OUTER APPLY
    (
      SELECT TOP (1) dropped_event_count,dropped_buffer_count,blocked_event_fire_time,address
      FROM sys.dm_xe_sessions WHERE name=N'logwarden_capture'
    ) AS runtime
    OUTER APPLY
    (
      SELECT TOP (1) failed_buffer_count,bytes_written,CONVERT(xml,target_data) AS target_xml
      FROM sys.dm_xe_session_targets WHERE event_session_address=runtime.address AND target_name=N'event_file'
    ) AS target
    WHERE c.source_id='xe-logwarden-capture';
  `);
  return { status: "available", ...(result.recordset[0] ?? {}) };
}

async function collectHost(): Promise<Record<string, unknown>> {
  const disk = await statfs(runDirectory);
  const journalBytes = await stat(journalPath).then((value) => value.size).catch(() => 0);
  return { status: "available", load1m: os.loadavg()[0], memoryUsedBytes: os.totalmem() - os.freemem(), diskFreeBytes: Number(disk.bavail) * Number(disk.bsize), journalBytes };
}

async function persistGpu(pool: sql.ConnectionPool, at: Date, value: Record<string, unknown>): Promise<void> {
  const key = hashJson({ runId: run.runId, source: "gpu", at: at.toISOString() });
  await pool.request().input("key",sql.Char(64),key).input("run",sql.VarChar(120),run.runId).input("uuid",sql.VarChar(120),value.uuid ?? "unavailable")
    .input("at",sql.DateTime2(7),at).input("gpu",sql.Float,value.utilizationGpuPct ?? null).input("mem_util",sql.Float,value.utilizationMemoryPct ?? null)
    .input("mem_used",sql.Float,value.memoryUsedMiB ?? null).input("mem_total",sql.Float,value.memoryTotalMiB ?? null).input("power",sql.Float,value.powerDrawW ?? null)
    .input("power_limit",sql.Float,value.powerLimitW ?? null).input("temp",sql.Float,value.temperatureC ?? null).input("sm",sql.Float,value.smClockMHz ?? null)
    .input("mem_clock",sql.Float,value.memoryClockMHz ?? null).input("pstate",sql.VarChar(16),value.pstate ?? null).input("processes",sql.NVarChar(sql.MAX),canonicalJson(value.processes ?? []))
    .input("error",sql.NVarChar(500),value.status === "available" ? null : value.error).query(`INSERT telemetry.gpu_samples(sample_key,run_id,gpu_uuid,sampled_at_utc,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,power_draw_w,power_limit_w,temperature_c,sm_clock_mhz,memory_clock_mhz,pstate,process_json,unavailable_reason) VALUES(@key,@run,@uuid,@at,@gpu,@mem_util,@mem_used,@mem_total,@power,@power_limit,@temp,@sm,@mem_clock,@pstate,@processes,@error);`);
}

async function persistSql(pool: sql.ConnectionPool, at: Date, value: Record<string, unknown>): Promise<void> {
  const key=hashJson({runId:run.runId,source:"sql",at:at.toISOString()});
  await pool.request().input("key",sql.Char(64),key).input("run",sql.VarChar(120),run.runId).input("at",sql.DateTime2(7),at)
    .input("memory",sql.BigInt,value.process_memory_kb ?? null).input("target",sql.BigInt,value.target_memory_kb ?? null).input("requests",sql.Int,value.request_count ?? null)
    .input("blocked",sql.Int,value.blocked_request_count ?? null).input("runnable",sql.Int,value.runnable_task_count ?? null).input("io_pending",sql.Int,value.pending_io_count ?? null)
    .input("data",sql.BigInt,value.data_file_bytes ?? null).input("log",sql.BigInt,value.log_file_bytes ?? null).input("waits",sql.NVarChar(sql.MAX),canonicalJson(value.waits ?? []))
    .input("io",sql.NVarChar(sql.MAX),canonicalJson(value.io ?? [])).query(`INSERT telemetry.sql_resource_samples(sample_key,run_id,sampled_at_utc,process_memory_kb,target_memory_kb,request_count,blocked_request_count,runnable_task_count,pending_io_count,data_file_bytes,log_file_bytes,waits_json,io_json) VALUES(@key,@run,@at,@memory,@target,@requests,@blocked,@runnable,@io_pending,@data,@log,@waits,@io);`);
}

async function persistQueue(pool: sql.ConnectionPool, at: Date, value: Record<string, unknown>): Promise<void> {
  const key=hashJson({runId:run.runId,source:"queue",at:at.toISOString()});
  await pool.request().input("key",sql.Char(64),key).input("run",sql.VarChar(120),run.runId).input("at",sql.DateTime2(7),at)
    .input("pending",sql.Int,value.pending_count).input("leased",sql.Int,value.leased_count).input("retry",sql.Int,value.retryable_count).input("terminal",sql.Int,value.terminal_count)
    .input("age",sql.BigInt,value.oldest_pending_age_ms ?? null).input("expired",sql.Int,value.lease_expired_count).input("workers",sql.Int,value.worker_count)
    .input("epoch",sql.UniqueIdentifier,epochId).query(`INSERT telemetry.queue_samples(sample_key,run_id,sampled_at_utc,pending_count,leased_count,retryable_count,terminal_count,oldest_pending_age_ms,lease_expired_count,worker_count,sampler_epoch_id) VALUES(@key,@run,@at,@pending,@leased,@retry,@terminal,@age,@expired,@workers,@epoch);`);
}

async function persistXe(pool: sql.ConnectionPool, at: Date, value: Record<string, unknown>): Promise<void> {
  const key=hashJson({runId:run.runId,source:"xe",at:at.toISOString()});
  const newest=value.newest_event_at_utc instanceof Date ? value.newest_event_at_utc : null;
  await pool.request().input("key",sql.Char(64),key).input("run",sql.VarChar(120),run.runId).input("at",sql.DateTime2(7),at).input("file",sql.NVarChar(500),value.file_name ?? null)
    .input("offset",sql.BigInt,value.file_offset ?? null).input("newest",sql.DateTime2(7),newest).input("lag",sql.BigInt,newest===null?null:at.getTime()-newest.getTime())
    .input("batch",sql.Int,value.batch_rows ?? null).input("errors",sql.BigInt,value.parse_errors_total ?? 0)
    .input("rollover",sql.Int,value.rollover_files ?? null).input("dropped",sql.BigInt,value.dropped_events_total ?? null)
    .input("dropped_buffers",sql.BigInt,value.dropped_buffers_total ?? null).input("blocked_fire",sql.BigInt,value.blocked_event_fire_time_ms ?? null)
    .input("failed_buffers",sql.BigInt,value.failed_target_buffers_total ?? null).input("bytes",sql.BigInt,value.source_bytes ?? null)
    .query(`INSERT telemetry.xe_pipeline_samples(sample_key,run_id,source_id,sampled_at_utc,cursor_file_name,cursor_file_offset,newest_event_at_utc,ingest_lag_ms,batch_rows,parse_errors_total,rollover_files,dropped_events_total,source_bytes,dropped_buffers_total,blocked_event_fire_time_ms,failed_target_buffers_total) VALUES(@key,@run,'xe-logwarden-capture',@at,@file,@offset,@newest,@lag,@batch,@errors,@rollover,@dropped,@bytes,@dropped_buffers,@blocked_fire,@failed_buffers);`);
}

async function persistHost(pool: sql.ConnectionPool, at: Date, value: Record<string, unknown>): Promise<void> {
  const key=hashJson({runId:run.runId,source:"host",at:at.toISOString()});
  await pool.request().input("key",sql.Char(64),key).input("run",sql.VarChar(120),run.runId).input("at",sql.DateTime2(7),at).input("load",sql.Float,value.load1m)
    .input("memory",sql.BigInt,value.memoryUsedBytes).input("disk",sql.BigInt,value.diskFreeBytes).input("journal",sql.BigInt,value.journalBytes).input("json",sql.NVarChar(sql.MAX),canonicalJson(value))
    .query(`INSERT telemetry.host_samples(sample_key,run_id,sampled_at_utc,load_1m,memory_used_bytes,disk_free_bytes,journal_bytes,attributes_json) VALUES(@key,@run,@at,@load,@memory,@disk,@journal,@json);`);
}

function modelFields(samples: PrometheusSample[]) {
  const metric = (...names: string[]) => sumPrometheusMetric(samples, names);
  return {
    runningRequests: metric("vllm:num_requests_running","vllm_num_requests_running"), waitingRequests: metric("vllm:num_requests_waiting","vllm_num_requests_waiting"), swappedRequests: metric("vllm:num_requests_swapped","vllm_num_requests_swapped"),
    promptTokensTotal: metric("vllm:prompt_tokens_total","vllm_prompt_tokens_total"), generationTokensTotal: metric("vllm:generation_tokens_total","vllm_generation_tokens_total"),
    promptThroughput: metric("vllm:avg_prompt_throughput_toks_per_s","vllm_avg_prompt_throughput_toks_per_s"), generationThroughput: metric("vllm:avg_generation_throughput_toks_per_s","vllm_avg_generation_throughput_toks_per_s"),
    kvCacheUsageRatio: metric("vllm:gpu_cache_usage_perc","vllm_gpu_cache_usage_perc"), prefixCacheHitsTotal: metric("vllm:prefix_cache_hits_total","vllm_prefix_cache_hits_total"),
    prefixCacheQueriesTotal: metric("vllm:prefix_cache_queries_total","vllm_prefix_cache_queries_total"), preemptionsTotal: metric("vllm:num_preemptions_total","vllm_num_preemptions_total"),
    requestErrorsTotal: metric("vllm:request_errors_total","vllm_request_errors_total"), cancellationsTotal: metric("vllm:request_cancellations_total","vllm_request_cancellations_total"),
  };
}

function normalizeCounts(value: Record<string, unknown>): Record<string, unknown> { return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, child ?? 0])); }
function numberOrNull(value: string | undefined): number | null { const parsed=Number(value); return value===undefined||!Number.isFinite(parsed)?null:parsed; }
function round(value:number):number{return Math.round(value*1000)/1000;}
function safeError(error:unknown):{message:string}{return{message:(error as {message?:string}).message??String(error)};}
function delay(milliseconds:number,signal?:AbortSignal):Promise<void>{return new Promise((resolve)=>{let timer:NodeJS.Timeout;const done=():void=>{clearTimeout(timer);signal?.removeEventListener("abort",done);resolve();};timer=setTimeout(done,milliseconds);if(signal?.aborted)done();else signal?.addEventListener("abort",done,{once:true});});}
function argument(name:string):string|undefined{const index=process.argv.indexOf(name);return index<0?undefined:process.argv[index+1];}
function commandOutput(command:string,args:string[]):Promise<string>{return new Promise((resolve,reject)=>execFile(command,args,{encoding:"utf8",maxBuffer:16*1024*1024},(error,stdout,stderr)=>error===null?resolve(stdout):reject(new Error(`${command} failed: ${stderr.trim()||error.message}`))));}
