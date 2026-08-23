import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { claimWithAvailabilityRetry } from "../src/queue-claim.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

interface RunManifest {
  runId: string;
}

interface CaseResult {
  name: string;
  status: "pass" | "fail";
  elapsedMs: number;
  detail?: unknown;
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const suffix = randomUUID().replaceAll("-", "").slice(0, 16);
const prefix = `sqlit-${suffix}`;
const queueClaimWorkers = 16;
const cases: CaseResult[] = [];
const cleanupWorkItemIds: number[] = [];
const cleanupJobIds: number[] = [];
let queueConcurrencyEvidence: Record<string, unknown> | null = null;

const admin = await connect(config.databases.admin, config.databases.controlName);
const agentA = await connect(config.databases.agent, config.databases.controlName, 600_000, queueClaimWorkers + 2);
const agentB = await connect(config.databases.agent, config.databases.controlName);

try {
  await test("migration_count_and_hash_shape", async () => {
    const result = await admin.request().query<{ migration_count: number; malformed_hashes: number }>(`
      SELECT COUNT(*) AS migration_count,
             SUM(CASE WHEN migration_sha256 LIKE REPLICATE('[0-9a-f]', 64) THEN 0 ELSE 1 END) AS malformed_hashes
      FROM control.schema_migrations;
    `);
    assert.ok(result.recordset[0]!.migration_count >= 12);
    assert.equal(result.recordset[0]!.malformed_hashes, 0);
  });

  await test("invalid_json_rejected", async () => {
    await expectSqlError(admin, 547, () => admin.request()
      .input("key", sql.VarChar(120), `${prefix}-invalid-json`)
      .query(`
        INSERT control.evidence_events(event_key, stage, scientific_tier, disposition, detail_json, detail_sha256)
        VALUES(@key, 'integration', 'foundation', 'expected_failure', N'{not-json', REPLICATE('0',64));
      `));
  });

  await test("invalid_enum_rejected", async () => {
    await expectSqlError(admin, 547, () => admin.request()
      .input("id", sql.VarChar(100), `${prefix}-scenario`)
      .query(`
        INSERT workload.scenario_definitions
        (scenario_id, scenario_group_id, family, regime, scenario_version, description,
         max_runtime_seconds, safety_class, cleanup_procedure, expected_class,
         expected_severity, should_abstain, is_multi_event, is_context_dependent,
         config_json, ground_truth_json, scenario_sha256)
        VALUES(@id, @id, 'integration', 'X', 1, N'constraint probe', 5,
          'benign', N'lab.usp_reset', 'probe', 'low', 0, 0, 0, N'{}', N'{}', REPLICATE('1',64));
      `));
  });

  await test("invalid_foreign_key_rejected", async () => {
    await expectSqlError(admin, 547, () => admin.request()
      .input("key", sql.Char(64), sha256(`${prefix}-fk`))
      .query(`
        INSERT control.jobs(job_key, campaign_id, run_kind, sample_index, status)
        VALUES(@key, 9223372036854775807, 'integration', 0, 'pending');
      `));
  });

  await test("agent_permission_matrix", async () => {
    const allowed = await agentA.request().query(`
      SELECT TOP (0) work_item_id FROM ops.work_items;
      SELECT CONVERT(int, HAS_PERMS_BY_NAME(N'ops.usp_claim_work_item', N'OBJECT', N'EXECUTE')) AS can_claim;
    `);
    const allowedSets = allowed.recordsets as unknown as Array<Array<{ can_claim?: number }>>;
    assert.equal(allowedSets[1]?.[0]?.can_claim, 1);
    await expectSqlError(agentA, 229, () => agentA.request().query("SELECT TOP (1) * FROM eval.ground_truth_episodes;"));
    await expectSqlError(agentA, 229, () => agentA.request().query("SELECT TOP (1) ground_truth_json FROM workload.scenario_definitions;"));
    await expectSqlError(agentA, 229, () => agentA.request().query("UPDATE kb.runbook_chunks SET content = content WHERE 1 = 0;"));
  });

  const campaign = await admin.request().query<{ campaign_id: number }>("SELECT TOP (1) campaign_id FROM control.campaigns ORDER BY campaign_id;");
  const campaignId = campaign.recordset[0]?.campaign_id;
  assert.ok(campaignId !== undefined, "foundation campaign is missing");

  const queueRows = await seedQueueItems(queueClaimWorkers + 1, campaignId);
  cleanupJobIds.push(...queueRows.map((row) => row.job_id));
  cleanupWorkItemIds.push(...queueRows.map((row) => row.work_item_id));

  await test("queue_sixteen_worker_claims_are_distinct", async () => {
    const selectedWorkItemIds = queueRows.map((row) => row.work_item_id);
    const claims = await Promise.all(Array.from({ length: queueClaimWorkers }, async (_, workerIndex) => {
      const token = randomUUID();
      const result = await claimWithAvailabilityRetry({
        workerIndex,
        claim: async () => {
          const value = await claim(agentA, `${prefix}-worker-${String(workerIndex).padStart(2, "0")}`, token);
          return value.work_item_id === undefined ? undefined : value;
        },
        countClaimableSelected: () => countClaimableSelected(agentA, selectedWorkItemIds),
      });
      assert.ok(result.work !== undefined, `worker ${workerIndex} did not receive a fixture claim`);
      return { token, work: result.work, retry: result };
    }));
    const claimedIds = claims.map((entry) => Number(entry.work.work_item_id));
    assert.equal(new Set(claimedIds).size, queueClaimWorkers);
    assert.deepEqual(new Set(claimedIds), new Set(queueRows.slice(0, queueClaimWorkers).map((row) => row.work_item_id)));
    queueConcurrencyEvidence = {
      configuredWorkers: queueClaimWorkers,
      distinctClaims: new Set(claimedIds).size,
      workersWithTransientEmptyClaims: claims.filter((entry) => entry.retry.emptyClaimCount > 0).length,
      totalEmptyClaims: claims.reduce((total, entry) => total + entry.retry.emptyClaimCount, 0),
      totalRetries: claims.reduce((total, entry) => total + entry.retry.retryCount, 0),
      totalRetryWaitMs: claims.reduce((total, entry) => total + entry.retry.retryWaitMs, 0),
    };

    const claimA = claims[0]!;
    const claimB = claims[1]!;

    await transition(agentA, Number(claimA.work.work_item_id), claimA.token, "packet_loaded");
    await heartbeat(agentA, Number(claimA.work.work_item_id), claimA.token);
    await expectSqlError(agentA, 51201, () => transition(agentA, Number(claimA.work.work_item_id), claimA.token, "validated"));
    for (const state of ["model_requested", "decision_received", "validated", "persisted", "complete"])
      await transition(agentA, Number(claimA.work.work_item_id), claimA.token, state);

    await agentB.request()
      .input("work_item_id", sql.BigInt, claimB.work.work_item_id)
      .input("lease_token", sql.UniqueIdentifier, claimB.token)
      .execute("ops.usp_complete_work_item");
    await expectSqlError(agentB, 51001, () => heartbeat(agentB, Number(claimB.work.work_item_id), claimB.token));
    await Promise.all(claims.slice(2).map((entry) => agentA.request()
      .input("work_item_id", sql.BigInt, entry.work.work_item_id)
      .input("lease_token", sql.UniqueIdentifier, entry.token)
      .execute("ops.usp_complete_work_item")));
  });

  await test("expired_active_lease_is_recoverable", async () => {
    const firstToken = randomUUID();
    const secondToken = randomUUID();
    const firstClaim = await claim(agentA, `${prefix}-crash-worker`, firstToken);
    const recoveryRow = queueRows[queueClaimWorkers]!;
    assert.equal(Number(firstClaim.work_item_id), recoveryRow.work_item_id);
    await transition(agentA, Number(firstClaim.work_item_id), firstToken, "packet_loaded");
    await admin.request()
      .input("id", sql.BigInt, firstClaim.work_item_id)
      .query("UPDATE ops.work_items SET leased_until_utc = DATEADD(SECOND,-1,SYSUTCDATETIME()) WHERE work_item_id = @id;");
    const recovered = await claim(agentB, `${prefix}-recovery-worker`, secondToken);
    assert.equal(Number(recovered.work_item_id), recoveryRow.work_item_id);
    assert.equal(Number(recovered.attempt_count), 2);
    await agentB.request()
      .input("work_item_id", sql.BigInt, recovered.work_item_id)
      .input("lease_token", sql.UniqueIdentifier, secondToken)
      .execute("ops.usp_complete_work_item");
  });

  await test("fulltext_and_exact_vector_retrieval", async () => {
    const corpusId = `${prefix}-corpus`;
    const runbookId = `${prefix}-runbook`;
    const chunkId = `${prefix}-chunk`;
    const embedding = JSON.stringify([1, ...Array.from({ length: 1023 }, () => 0)]);
    try {
      await admin.request()
        .input("corpus", sql.VarChar(80), corpusId)
        .input("runbook", sql.VarChar(100), runbookId)
        .input("chunk", sql.VarChar(120), chunkId)
        .input("embedding", sql.NVarChar(sql.MAX), embedding)
        .query(`
          INSERT kb.search_corpora(corpus_id, corpus_version, corpus_kind, source_manifest_json, source_manifest_sha256)
          VALUES(@corpus, 'integration-v1', 'primary', N'{}', CONVERT(char(64), HASHBYTES('SHA2_256', @corpus), 2));
          INSERT kb.runbooks(runbook_id, corpus_id, title, incident_class, severity_floor, source_kind,
            source_sha256, body_markdown, body_sha256, metadata_json)
          VALUES(@runbook, @corpus, N'Deadlock integration runbook', 'deadlock', 'high', 'generated',
            CONVERT(char(64), HASHBYTES('SHA2_256', @runbook), 2), N'Deadlock victim retry guidance',
            CONVERT(char(64), HASHBYTES('SHA2_256', N'Deadlock victim retry guidance'), 2), N'{}');
          INSERT kb.runbook_chunks(chunk_id, runbook_id, corpus_id, ordinal, heading_path, content,
            token_count, chunker_version, content_sha256, metadata_json)
          VALUES(@chunk, @runbook, @corpus, 0, N'Root', N'Deadlock victim retry guidance', 4, 'integration-v1',
            CONVERT(char(64), HASHBYTES('SHA2_256', @chunk), 2), N'{}');
          INSERT kb.chunk_embeddings(chunk_id, embedding_profile_id, embedding, embedding_sha256, normalized)
          VALUES(@chunk, 'qwen3-embedding-0.6b', CAST(@embedding AS vector(1024)),
            CONVERT(char(64), HASHBYTES('SHA2_256', @embedding), 2), 1);
        `);

      let lexicalHit = false;
      for (let attempt = 0; attempt < 40 && !lexicalHit; attempt += 1) {
        const lexical = await admin.request()
          .input("chunk", sql.VarChar(120), chunkId)
          .query(`
            SELECT CONVERT(int, COUNT(*)) AS hit_count
            FROM CONTAINSTABLE(kb.runbook_chunks, content, N'"deadlock"') AS ft
            WHERE ft.[KEY] = @chunk;
          `);
        lexicalHit = lexical.recordset[0]?.hit_count === 1;
        if (!lexicalHit) await delay(250);
      }
      assert.equal(lexicalHit, true, "full-text change tracking did not expose the test chunk");

      const semantic = await agentA.request()
        .input("corpus", sql.VarChar(80), corpusId)
        .input("embedding", sql.NVarChar(sql.MAX), embedding)
        .batch(`
          DECLARE @query vector(1024) = CAST(@embedding AS vector(1024));
          EXEC kb.search_runbook_exact @corpus_id=@corpus,
            @embedding_profile_id='qwen3-embedding-0.6b', @query_embedding=@query, @top_k=1;
        `);
      assert.equal(semantic.recordset[0]?.chunk_id, chunkId);
      assert.equal(Number(semantic.recordset[0]?.vector_distance), 0);
    } finally {
      await admin.request()
        .input("corpus", sql.VarChar(80), corpusId)
        .query(`
          DELETE e FROM kb.chunk_embeddings AS e INNER JOIN kb.runbook_chunks AS c ON c.chunk_id=e.chunk_id WHERE c.corpus_id=@corpus;
          DELETE FROM kb.runbook_chunks WHERE corpus_id=@corpus;
          DELETE FROM kb.runbooks WHERE corpus_id=@corpus;
          DELETE FROM kb.search_corpora WHERE corpus_id=@corpus;
        `);
    }
  });
} finally {
  if (cleanupWorkItemIds.length > 0) {
    const ids = cleanupWorkItemIds.join(",");
    await admin.request().batch(`DELETE FROM ops.transitions WHERE entity_kind='work_item' AND entity_id IN (${ids}); DELETE FROM ops.work_items WHERE work_item_id IN (${ids});`).catch(() => undefined);
  }
  if (cleanupJobIds.length > 0) {
    const ids = cleanupJobIds.join(",");
    await admin.request().batch(`DELETE FROM control.job_attempts WHERE job_id IN (${ids}); DELETE FROM control.jobs WHERE job_id IN (${ids});`).catch(() => undefined);
  }
  await Promise.allSettled([admin.close(), agentA.close(), agentB.close()]);
}

const receiptBody = {
  schemaVersion: 1,
  runId: run.runId,
  cases,
  queueConcurrencyEvidence,
  passed: cases.filter((entry) => entry.status === "pass").length,
  failed: cases.filter((entry) => entry.status === "fail").length,
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/validation/sql-integration.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));
if (receipt.failed > 0) throw new Error(`${receipt.failed} SQL integration case(s) failed`);

async function test(name: string, operation: () => Promise<void>): Promise<void> {
  const started = performance.now();
  try {
    await operation();
    cases.push({ name, status: "pass", elapsedMs: roundMs(performance.now() - started) });
  } catch (error) {
    const detail = sqlError(error);
    cases.push({ name, status: "fail", elapsedMs: roundMs(performance.now() - started), detail });
  }
}

async function expectSqlError(pool: sql.ConnectionPool, number: number, operation: () => Promise<unknown>): Promise<void> {
  void pool;
  try {
    await operation();
  } catch (error) {
    assert.equal((error as { number?: number }).number, number);
    return;
  }
  assert.fail(`Expected SQL error ${number}`);
}

async function seedQueueItems(count: number, campaignId: number): Promise<Array<{ job_id: number; work_item_id: number }>> {
  const rows: Array<{ job_id: number; work_item_id: number }> = [];
  for (let ordinal = 0; ordinal < count; ordinal += 1) {
    const job = await admin.request()
      .input("key", sql.Char(64), sha256(`${prefix}-job-${ordinal}`))
      .input("campaign", sql.BigInt, campaignId)
      .input("episode", sql.VarChar(120), `${prefix}-episode-${ordinal}`)
      .query<{ job_id: number }>(`
        INSERT control.jobs(job_key, campaign_id, run_kind, episode_id, sample_index, priority, status)
        OUTPUT INSERTED.job_id
        VALUES(@key, @campaign, 'integration', @episode, 0, 2147483000, 'pending');
      `);
    const jobId = Number(job.recordset[0]!.job_id);
    const item = await admin.request()
      .input("run", sql.VarChar(120), run.runId)
      .input("job", sql.BigInt, jobId)
      .input("episode", sql.VarChar(120), `${prefix}-episode-${ordinal}`)
      .query<{ work_item_id: number }>(`
        INSERT ops.work_items(run_id, job_id, episode_id, status, priority)
        OUTPUT INSERTED.work_item_id
        VALUES(@run, @job, @episode, 'pending', 2147483000);
      `);
    rows.push({ job_id: jobId, work_item_id: Number(item.recordset[0]!.work_item_id) });
  }
  return rows;
}

async function claim(pool: sql.ConnectionPool, worker: string, token: string): Promise<Record<string, unknown>> {
  const result = await pool.request()
    .input("worker_id", sql.VarChar(120), worker)
    .input("lease_token", sql.UniqueIdentifier, token)
    .input("lease_seconds", sql.Int, 120)
    .execute("ops.usp_claim_work_item");
  return (result.recordset[0] ?? {}) as Record<string, unknown>;
}

async function countClaimableSelected(pool: sql.ConnectionPool, workItemIds: number[]): Promise<number> {
  const result = await pool.request().input("ids", sql.NVarChar(sql.MAX), JSON.stringify(workItemIds))
    .query<{ claimable_count: number }>(`
      WITH selected AS (SELECT CONVERT(bigint,value) work_item_id FROM OPENJSON(@ids))
      SELECT COUNT(*) claimable_count
      FROM ops.work_items item INNER JOIN selected ON selected.work_item_id=item.work_item_id
      WHERE item.next_attempt_at_utc<=SYSUTCDATETIME()
        AND (item.status IN ('pending','retryable_failure','lease_expired') OR
          (item.status IN ('leased','packet_loaded','model_requested','tool_requested','tool_completed',
            'decision_received','validated','persisted','proposed_action_recorded')
           AND item.leased_until_utc<SYSUTCDATETIME()));
    `);
  return Number(result.recordset[0]?.claimable_count ?? 0);
}

async function heartbeat(pool: sql.ConnectionPool, workItemId: number, token: string): Promise<void> {
  await pool.request()
    .input("work_item_id", sql.BigInt, workItemId)
    .input("lease_token", sql.UniqueIdentifier, token)
    .input("lease_seconds", sql.Int, 120)
    .execute("ops.usp_heartbeat_work_item");
}

async function transition(pool: sql.ConnectionPool, workItemId: number, token: string, state: string): Promise<void> {
  await pool.request()
    .input("work_item_id", sql.BigInt, workItemId)
    .input("lease_token", sql.UniqueIdentifier, token)
    .input("to_state", sql.VarChar(40), state)
    .input("actor", sql.VarChar(120), prefix)
    .input("reason", sql.NVarChar(1000), "SQL integration state-machine probe")
    .execute("ops.usp_transition_work_item");
}

function sqlError(error: unknown): Record<string, unknown> {
  const candidate = error as { number?: number; message?: string };
  return {
    ...(candidate.number === undefined ? {} : { number: candidate.number }),
    message: candidate.message ?? String(error),
  };
}

function roundMs(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
