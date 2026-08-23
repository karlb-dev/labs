import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { parseErrorlogFiles } from "../src/errorlog.js";
import { readErrorlogFilesFromContainer } from "../src/errorlog-files.js";
import { hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

interface RunManifest { runId: string }

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const requestedSource = argument("--source") ?? "all";
if (!['all', 'xe', 'errorlog'].includes(requestedSource))
  throw new Error("--source must be all, xe, or errorlog");

const ingestConnection = {
  ...config.databases.lab,
  options: { ...config.databases.lab.options, appName: "LogWarden-Ingest" },
};
const pool = await connect(ingestConnection, config.databases.controlName);
const receiptBody: Record<string, unknown> = {
  schemaVersion: 1,
  runId: run.runId,
  source: requestedSource,
  startedAtUtc: new Date().toISOString(),
};
try {
  if (requestedSource === "all" || requestedSource === "xe") {
    const result = await pool.request()
      .input("run_id", sql.VarChar(120), run.runId)
      .input("worker_id", sql.VarChar(120), `ingest-${process.pid}`)
      .execute("ingest.usp_ingest_xe");
    receiptBody.xe = result.recordset[0] ?? {};
  }
  if (requestedSource === "all" || requestedSource === "errorlog") {
    const files = await readErrorlogFilesFromContainer(process.env.LOGWARDEN_SQL_CONTAINER ?? "aidataapps-logwarden-sqlserver-1");
    const scan = parseErrorlogFiles(files);
    const result = await pool.request()
      .input("worker_id", sql.VarChar(120), `ingest-${process.pid}`)
      .input("records_json", sql.NVarChar(sql.MAX), JSON.stringify(scan.records))
      .input("cursor_json", sql.NVarChar(sql.MAX), JSON.stringify(scan.cursor))
      .input("log_generation", sql.Int, scan.cursor.generations.length)
      .execute("ingest.usp_ingest_errorlog_scan");
    receiptBody.errorlog = {
      ...(result.recordset[0] ?? {}),
      scannedFiles: files.map((file) => file.path),
      parsedRecords: scan.records.length,
      cursorSha256: hashJson(scan.cursor),
    };
  }
} finally {
  await pool.close();
}
receiptBody.finishedAtUtc = new Date().toISOString();
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
const stamp = new Date().toISOString().replaceAll(/[-:.]/g, "");
await atomicWrite(`${runDirectory}/ingestion/${stamp}-${requestedSource}.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
