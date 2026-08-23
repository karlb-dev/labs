import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

const cases = [
  ["deadlock victim cycle retry", "TSG-DLK-"],
  ["head blocker open transaction waiters", "TSG-BLK-"],
  ["transaction log full 9002 active transaction", "TSG-LOG-"],
  ["login failed 18456 default database", "TSG-AUT-"],
  ["integrity checksum allocation consistency", "TSG-INT-"],
  ["backup destination failure restore checksum", "TSG-BAK-"],
  ["query high CPU memory grant timeout", "TSG-QRY-"],
  ["conversion constraint missing object truncation", "TSG-DAT-"],
  ["ambiguous conflicting truncated unsupported", "TSG-UNK-"],
  ["successful duplicate subthreshold no action", "TSG-NOI-"],
  ["timeout (error 9002) 'active transaction'", "TSG-LOG-"],
] as const;

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const agent = await connect(config.databases.agent, config.databases.controlName, 30_000);
const results: Array<Record<string, unknown>> = [];
try {
  for (const [query, expectedPrefix] of cases) {
    const started = performance.now();
    const response = await agent.request()
      .input("query", sql.NVarChar(1000), query)
      .input("top_k", sql.Int, 5)
      .input("corpus_id", sql.VarChar(80), "primary-v1")
      .execute("kb.usp_search_runbooks");
    const rows = response.recordset as Array<{ chunk_id: string; runbook_id: string; lexical_score: number; rank_ordinal: string | number }>;
    if (rows.length === 0 || !rows.some((row) => row.runbook_id.startsWith(expectedPrefix)))
      throw new Error(`Full-text case did not retrieve ${expectedPrefix}: ${query}`);
    if (rows.some((row) => !Number.isFinite(Number(row.lexical_score)) || Number(row.rank_ordinal) < 1))
      throw new Error(`Full-text case returned invalid score/rank: ${query}`);
    results.push({
      query,
      querySha256: hashJson(query),
      expectedPrefix,
      returnedRunbooks: rows.map((row) => row.runbook_id),
      topChunkId: rows[0]!.chunk_id,
      latencyMs: Number((performance.now() - started).toFixed(3)),
      disposition: "PASS",
    });
  }
} finally {
  await agent.close();
}

const admin = await connect(config.databases.admin, config.databases.controlName, 30_000);
let administrative: Record<string, unknown>;
try {
  const evidence = await admin.request().query<{ indexed_column_count: number; item_count: number; populate_status: number; primary_runbooks: number; primary_chunks: number }>(`
    SELECT
      (SELECT COUNT(*) FROM sys.fulltext_index_columns WHERE object_id=OBJECT_ID('kb.runbook_chunks')
       AND column_id IN (COLUMNPROPERTY(OBJECT_ID('kb.runbook_chunks'),'heading_path','ColumnId'),COLUMNPROPERTY(OBJECT_ID('kb.runbook_chunks'),'content','ColumnId'))) AS indexed_column_count,
      CONVERT(int,FULLTEXTCATALOGPROPERTY('logwarden_runbooks_fts','ItemCount')) AS item_count,
      CONVERT(int,FULLTEXTCATALOGPROPERTY('logwarden_runbooks_fts','PopulateStatus')) AS populate_status,
      (SELECT COUNT(*) FROM kb.runbooks WHERE corpus_id='primary-v1') AS primary_runbooks,
      (SELECT COUNT(*) FROM kb.runbook_chunks WHERE corpus_id='primary-v1') AS primary_chunks;
  `);
  const row = evidence.recordset[0]!;
  if (Number(row.indexed_column_count) !== 2 || Number(row.populate_status) !== 0 ||
      Number(row.primary_runbooks) !== 60 || Number(row.primary_chunks) !== 480 || Number(row.item_count) < 480)
    throw new Error(`Full-text administrative evidence failed: ${JSON.stringify(row)}`);
  administrative = row;
} finally {
  await admin.close();
}

const receiptBody = {
  schemaVersion: 1,
  runId: run.runId,
  corpusId: "primary-v1",
  queryCases: results,
  administrative,
  disposition: "PASS",
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/knowledge/lexical-gate.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({ runId: run.runId, cases: results.length, disposition: "PASS", receiptSha256: receipt.receiptSha256 }, null, 2));
