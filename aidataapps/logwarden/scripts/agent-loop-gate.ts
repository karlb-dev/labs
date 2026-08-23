import { randomUUID } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import sql from "mssql";
import {
  DEFAULT_AGENT_LOOP_BUDGET,
  buildInitialAgentMessages,
  runAgentLoop,
  type AgentLoopBudget,
  type AgentLoopResult,
  type PrimaryAgentArm,
} from "../src/agent-loop.js";
import { loadConfig } from "../src/config.js";
import { OPERATING_CONTRACT_VERSION } from "../src/contracts.js";
import { summarizeEmbeddingMetrics, validateEmbeddingMetricDelta } from "../src/embedding-metrics.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { resolveEmbeddingProfile, resolveModelProfile } from "../src/models.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal } from "../src/telemetry.js";
import { loadToolRegistry, toolNames, toolRegistrySha256 } from "../src/tools.js";

interface RunManifest { runId: string }
interface Fixture {
  jobId: number;
  jobAttemptId: number;
  workItemId: number;
  leaseToken: string;
  episodeId: string;
  workerId: string;
}
interface GateCase {
  name: string;
  arm: PrimaryAgentArm;
  episodeId?: string;
  expectedStatus: AgentLoopResult["status"];
  expectedTurns: number;
  expectedTools: number;
  expectedCacheHits: number;
  expectedSnapshotMisses: number;
  expectedDecisions: number;
  budget?: AgentLoopBudget;
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const gateId = randomUUID();
const registry = loadToolRegistry();
const modelProfile = resolveModelProfile("qwen-smoke");
const embeddingProfile = resolveEmbeddingProfile("qwen3-embedding-0.6b");
const lab = await connect(config.databases.lab, config.databases.controlName, 180_000);
const agent = await connect(config.databases.agent, config.databases.controlName, 180_000);
const snapshotFixture = await findSnapshotFixture(lab);
const fake = createServer((request, response) => routeFakeModel(request, response, snapshotFixture));
await new Promise<void>((resolve) => fake.listen(0, "127.0.0.1", resolve));
const address = fake.address();
if (address === null || typeof address === "string") throw new Error("Agent-loop fake gateway failed to bind TCP");
const fakeBase = `http://127.0.0.1:${address.port}`;
const rawRoot = `${runDirectory}/raw/agent-loop-gate/${gateId}`;
const receiptPath = `${runDirectory}/agent/agent-loop-gate.json`;
const cases: GateCase[] = [
  { name: "hybrid-cache-success", arm: "A-tools", expectedStatus: "complete", expectedTurns: 3, expectedTools: 2, expectedCacheHits: 1, expectedSnapshotMisses: 0, expectedDecisions: 1 },
  { name: "snapshot-hit-success", arm: "A-tools", episodeId: snapshotFixture.episodeId, expectedStatus: "complete", expectedTurns: 2, expectedTools: 1, expectedCacheHits: 0, expectedSnapshotMisses: 0, expectedDecisions: 1 },
  { name: "snapshot-miss-success", arm: "A-tools", expectedStatus: "complete", expectedTurns: 2, expectedTools: 1, expectedCacheHits: 0, expectedSnapshotMisses: 1, expectedDecisions: 1 },
  { name: "unknown-tool-policy", arm: "A-tools", expectedStatus: "policy_rejected", expectedTurns: 1, expectedTools: 1, expectedCacheHits: 0, expectedSnapshotMisses: 0, expectedDecisions: 0 },
  { name: "invalid-arguments-policy", arm: "A-tools", expectedStatus: "policy_rejected", expectedTurns: 1, expectedTools: 1, expectedCacheHits: 0, expectedSnapshotMisses: 0, expectedDecisions: 0 },
  { name: "contract-rejected", arm: "A-direct", expectedStatus: "contract_rejected", expectedTurns: 1, expectedTools: 0, expectedCacheHits: 0, expectedSnapshotMisses: 0, expectedDecisions: 0 },
  { name: "same-tool-budget", arm: "A-tools", expectedStatus: "policy_rejected", expectedTurns: 3, expectedTools: 3, expectedCacheHits: 1, expectedSnapshotMisses: 0, expectedDecisions: 0 },
  {
    name: "model-turn-budget",
    arm: "A-tools",
    expectedStatus: "policy_rejected",
    expectedTurns: 4,
    expectedTools: 4,
    expectedCacheHits: 0,
    expectedSnapshotMisses: 0,
    expectedDecisions: 0,
    budget: { ...DEFAULT_AGENT_LOOP_BUDGET, maxSameToolCalls: 4 },
  },
];

try {
  const campaignId = await ensureGateArms(lab);
  const recoveredFixtures = await recoverIncompleteAgentGateFixtures(lab);
  const metricsBefore = await retainEmbeddingMetrics("before");
  const caseResults: Array<{ gateCase: GateCase; fixture: Fixture; result: AgentLoopResult; journalPath: string; telemetry: unknown; database: unknown }> = [];
  for (const gateCase of cases) {
    const fixture = await createFixture(lab, campaignId, gateCase);
    const created = await createComponentTelemetryJournal(runDirectory, run.runId, `agent-loop-gate-${gateCase.name}`);
    const packet = syntheticPacket(fixture.episodeId);
    const result = await runAgentLoop({
      identity: {
        runId: run.runId,
        campaignId,
        jobId: fixture.jobId,
        jobAttemptId: fixture.jobAttemptId,
        workItemId: fixture.workItemId,
        leaseToken: fixture.leaseToken,
        episodeId: fixture.episodeId,
        modelProfileId: modelProfile.key,
        agentArmId: gateArmId(gateCase.arm),
        decodeConfigId: "primary-json-v2",
        workerId: fixture.workerId,
      },
      controlPool: lab,
      toolPool: agent,
      packet,
      arm: gateCase.arm,
      endpoint: `${fakeBase}/${gateCase.name}`,
      modelProfile,
      decode: { temperature: 0, top_p: 1, seed: 0, max_tokens: 900 },
      embeddingEndpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`,
      embeddingProfile,
      retrievalMode: "hybrid_rrf",
      runDirectory,
      journal: created.journal,
      toolRegistry: registry,
      budget: gateCase.budget ?? DEFAULT_AGENT_LOOP_BUDGET,
      timeoutMs: 10_000,
      maxRetries: 0,
    });
    assertResult(gateCase, result);
    const telemetry = await ingestTelemetryJournal(lab, run.runId, created.path, { batchSize: 100 });
    await linkTelemetry(lab, result);
    const database = await verifyCase(lab, gateCase, fixture, result);
    caseResults.push({ gateCase, fixture, result, journalPath: created.path, telemetry, database });
    console.log(`${gateCase.name}: ${result.status}; turns=${result.modelTurnCount}; tools=${result.toolCallCount}`);
  }
  const metricsAfter = await retainEmbeddingMetrics("after");
  const metricDelta = validateEmbeddingMetricDelta(metricsBefore.summary, metricsAfter.summary, 2, 2);
  if (metricDelta.embeddingHttpRequests !== 2 || metricDelta.successfulRequests !== 2 || metricDelta.latencyObservations !== 2) {
    throw new Error(`Agent-loop gate embedding metrics include unrelated traffic: ${JSON.stringify(metricDelta)}`);
  }
  const global = await verifyGlobal(lab, caseResults.map((entry) => entry.fixture.jobId));
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    gateId,
    createdAtUtc: new Date().toISOString(),
    contractVersion: OPERATING_CONTRACT_VERSION,
    toolRegistrySha256: toolRegistrySha256(registry),
    modelProfile: { key: modelProfile.key, modelId: modelProfile.modelId, revision: modelProfile.revision, transport: "fake_openai_compatible" },
    embeddingProfile: { key: embeddingProfile.key, modelId: embeddingProfile.modelId, revision: embeddingProfile.revision, transport: "real_vllm_cuda" },
    budget: DEFAULT_AGENT_LOOP_BUDGET,
    cases: caseResults.map(({ gateCase, fixture, result, journalPath, telemetry, database }) => ({
      name: gateCase.name,
      arm: gateCase.arm,
      jobId: fixture.jobId,
      jobAttemptId: fixture.jobAttemptId,
      workItemId: fixture.workItemId,
      episodeId: fixture.episodeId,
      agentRunId: result.agentRunId,
      traceId: result.traceId,
      status: result.status,
      terminalReason: result.terminalReason,
      decisionId: result.decisionId,
      modelTurnCount: result.modelTurnCount,
      toolCallCount: result.toolCallCount,
      toolResultChars: result.toolResultChars,
      cacheHitCount: result.cacheHitCount,
      snapshotMissCount: result.snapshotMissCount,
      elapsedMs: result.elapsedMs,
      journalPath,
      telemetry,
      database,
    })),
    embeddingMetrics: { before: metricsBefore, after: metricsAfter, delta: metricDelta },
    global,
    recoveredFixtures,
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const serialized = `${JSON.stringify(receipt, null, 2)}\n`;
  await atomicWrite(`${rawRoot}/receipt.json`, serialized);
  await atomicWrite(receiptPath, serialized);
  console.log(JSON.stringify({
    runId: run.runId,
    gateId,
    cases: cases.length,
    modelTurns: caseResults.reduce((sum, entry) => sum + entry.result.modelTurnCount, 0),
    toolCalls: caseResults.reduce((sum, entry) => sum + entry.result.toolCallCount, 0),
    decisions: caseResults.filter((entry) => entry.result.decisionId !== null).length,
    embeddingMetricDelta: metricDelta,
    disposition: "PASS",
    receiptSha256: receipt.receiptSha256,
  }, null, 2));
} catch (error) {
  const errorDetail = safeError(error);
  await recoverInterruptedFixtures(lab, gateId).catch(() => undefined);
  const failureBody = { schemaVersion: 1, runId: run.runId, gateId, createdAtUtc: new Date().toISOString(), errorDetail, disposition: "FAIL" };
  const failure = { ...failureBody, receiptSha256: hashJson(failureBody) };
  await atomicWrite(`${rawRoot}/failure-receipt.json`, `${JSON.stringify(failure, null, 2)}\n`).catch(() => undefined);
  console.error(JSON.stringify(failure, null, 2));
  process.exitCode = 1;
} finally {
  await Promise.all([lab.close(), agent.close()]);
  await new Promise<void>((resolve) => fake.close(() => resolve()));
}

async function ensureGateArms(pool: sql.ConnectionPool): Promise<number> {
  const campaign = await pool.request().query<{ campaign_id: string }>("SELECT TOP (1) campaign_id FROM control.campaigns WHERE status='building' ORDER BY campaign_id;");
  const campaignId = Number(campaign.recordset[0]?.campaign_id);
  if (!Number.isSafeInteger(campaignId)) throw new Error("No building campaign exists");
  for (const arm of ["A-direct", "A-rag", "A-tools"] as const) {
    const packet = syntheticPacket("arm-identity");
    const prompt = buildInitialAgentMessages({ arm, packet, registry })[0]!.content.split("\n\nIncident packet:")[0]!;
    const identity = {
      schemaVersion: 1,
      arm,
      promptSha256: sha256(prompt),
      toolRegistrySha256: toolRegistrySha256(registry),
      policySha256: hashJson({ version: "policy-v1-building", noRemediation: true, unknownRequiresAbstention: true }),
      contractSha256: hashJson({ version: OPERATING_CONTRACT_VERSION }),
      packetVersion: "agent-loop-gate-v1",
      retrievalMode: arm === "A-direct" ? "none" : "hybrid_rrf",
      correlationMode: "provided",
      budget: DEFAULT_AGENT_LOOP_BUDGET,
    };
    await pool.request()
      .input("id", sql.VarChar(80), gateArmId(arm))
      .input("hash", sql.Char(64), hashJson(identity))
      .input("prompt", sql.Char(64), identity.promptSha256)
      .input("tools", sql.Char(64), identity.toolRegistrySha256)
      .input("policy", sql.Char(64), identity.policySha256)
      .input("contract", sql.Char(64), identity.contractSha256)
      .input("retrieval", sql.VarChar(40), identity.retrievalMode)
      .input("json", sql.NVarChar(sql.MAX), canonicalJson(identity))
      .query(`
        IF NOT EXISTS (SELECT 1 FROM control.agent_arms WHERE agent_arm_id=@id)
          INSERT control.agent_arms
            (agent_arm_id,arm_hash,prompt_sha256,tool_registry_sha256,policy_sha256,
             contract_sha256,packet_version,retrieval_mode,correlation_mode,config_json)
          VALUES(@id,@hash,@prompt,@tools,@policy,@contract,'agent-loop-gate-v1',
            @retrieval,'provided',@json);
      `);
  }
  return campaignId;
}

async function createFixture(pool: sql.ConnectionPool, campaignId: number, gateCase: GateCase): Promise<Fixture> {
  const episodeId = gateCase.episodeId ?? `agent-loop-gate-${gateId.slice(0, 20)}-${gateCase.name}`;
  const workerId = `agent-loop-gate-${gateId.slice(0, 12)}-${gateCase.name}`.slice(0, 120);
  const leaseToken = randomUUID();
  const jobKey = hashJson({ gateId, case: gateCase.name });
  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  let jobId: number;
  let workItemId: number;
  try {
    const job = await new sql.Request(transaction)
      .input("key", sql.Char(64), jobKey)
      .input("campaign", sql.BigInt, campaignId)
      .input("episode", sql.VarChar(120), episodeId)
      .input("arm", sql.VarChar(80), gateArmId(gateCase.arm))
      .query<{ job_id: string }>(`
        INSERT control.jobs
          (job_key,campaign_id,run_kind,episode_id,model_profile_id,agent_arm_id,
           decode_config_id,sample_index,priority,status)
        OUTPUT INSERTED.job_id
        VALUES(@key,@campaign,'agent_loop_gate',@episode,'qwen-smoke',@arm,
          'primary-json-v2',0,2147483647,'pending');
      `);
    jobId = Number(job.recordset[0]!.job_id);
    const work = await new sql.Request(transaction)
      .input("run", sql.VarChar(120), run.runId)
      .input("job", sql.BigInt, jobId)
      .input("episode", sql.VarChar(120), episodeId)
      .query<{ work_item_id: string }>(`
        INSERT ops.work_items(run_id,job_id,episode_id,status,priority)
        OUTPUT INSERTED.work_item_id
        VALUES(@run,@job,@episode,'pending',2147483647);
      `);
    workItemId = Number(work.recordset[0]!.work_item_id);
    await transaction.commit();
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
  const claimed = await pool.request()
    .input("worker_id", sql.VarChar(120), workerId)
    .input("lease_token", sql.UniqueIdentifier, leaseToken)
    .input("lease_seconds", sql.Int, 600)
    .execute("ops.usp_claim_work_item");
  if (Number(claimed.recordset[0]?.work_item_id) !== workItemId) throw new Error(`Queue claim selected the wrong work item for ${gateCase.name}`);
  const attempt = await pool.request()
    .input("job", sql.BigInt, jobId)
    .input("worker", sql.VarChar(120), workerId)
    .input("manifest", sql.NVarChar(sql.MAX), canonicalJson({ schemaVersion: 1, gateId, case: gateCase.name, fakeModel: true, realEmbedding: true }))
    .query<{ job_attempt_id: string }>(`
      UPDATE control.jobs
      SET status='running',started_at_utc=SYSUTCDATETIME(),attempt_count=attempt_count+1
      WHERE job_id=@job AND status='pending';
      INSERT control.job_attempts(job_id,attempt_number,worker_id,request_manifest_json,status)
      OUTPUT INSERTED.job_attempt_id
      VALUES(@job,1,@worker,@manifest,'running');
    `);
  return { jobId, jobAttemptId: Number(attempt.recordset[0]!.job_attempt_id), workItemId, leaseToken, episodeId, workerId };
}

async function linkTelemetry(pool: sql.ConnectionPool, result: AgentLoopResult): Promise<void> {
  for (const link of result.spanLinks) {
    await pool.request()
      .input("trace", sql.Char(32), link.traceId)
      .input("span", sql.Char(16), link.spanId)
      .input("turn", sql.BigInt, link.turnId ?? null)
      .input("request", sql.BigInt, link.modelRequestId ?? null)
      .input("tool", sql.BigInt, link.toolInvocationId ?? null)
      .query(`
        UPDATE telemetry.spans
        SET turn_id=COALESCE(@turn,turn_id),model_request_id=COALESCE(@request,model_request_id),
            tool_invocation_id=COALESCE(@tool,tool_invocation_id)
        WHERE trace_id=@trace AND span_id=@span;
      `);
  }
}

async function verifyCase(
  pool: sql.ConnectionPool,
  gateCase: GateCase,
  fixture: Fixture,
  result: AgentLoopResult,
): Promise<Record<string, unknown>> {
  const observed = await pool.request()
    .input("job", sql.BigInt, fixture.jobId)
    .input("work", sql.BigInt, fixture.workItemId)
    .input("run", sql.BigInt, result.agentRunId)
    .input("trace", sql.Char(32), result.traceId)
    .query<Record<string, unknown>>(`
      SELECT
        (SELECT status FROM control.jobs WHERE job_id=@job) AS job_status,
        (SELECT status FROM ops.work_items WHERE work_item_id=@work) AS work_status,
        (SELECT status FROM agent.agent_runs WHERE agent_run_id=@run) AS agent_status,
        (SELECT model_turn_count FROM agent.agent_runs WHERE agent_run_id=@run) AS recorded_model_turn_count,
        (SELECT tool_call_count FROM agent.agent_runs WHERE agent_run_id=@run) AS recorded_tool_call_count,
        (SELECT tool_result_chars FROM agent.agent_runs WHERE agent_run_id=@run) AS recorded_tool_result_chars,
        (SELECT COUNT(*) FROM agent.turns WHERE agent_run_id=@run) AS turn_count,
        (SELECT COUNT(*) FROM agent.model_requests q INNER JOIN agent.turns t ON t.turn_id=q.turn_id WHERE t.agent_run_id=@run) AS request_count,
        (SELECT COUNT(*) FROM agent.model_responses p INNER JOIN agent.model_requests q ON q.model_request_id=p.model_request_id INNER JOIN agent.turns t ON t.turn_id=q.turn_id WHERE t.agent_run_id=@run) AS response_count,
        (SELECT COUNT(*) FROM agent.tool_invocations WHERE agent_run_id=@run) AS tool_count,
        (SELECT COUNT(*) FROM agent.tool_invocations WHERE agent_run_id=@run AND cache_hit=1) AS cache_count,
        (SELECT COUNT(*) FROM agent.tool_invocations WHERE agent_run_id=@run AND snapshot_miss=1) AS snapshot_miss_count,
        (SELECT COUNT(*) FROM agent.decisions WHERE agent_run_id=@run) AS decision_count,
        (SELECT COUNT(*) FROM agent.agent_steps WHERE agent_run_id=@run) AS step_count,
        (SELECT COUNT(*) FROM agent.validation_events WHERE agent_run_id=@run) AS validation_count,
        (SELECT COUNT(*) FROM ops.action_proposals p INNER JOIN agent.decisions d ON d.decision_id=p.decision_id WHERE d.agent_run_id=@run AND p.caller_opted_in=0 AND p.execution_status='not_executed') AS safe_proposal_count,
        (SELECT COUNT(*) FROM agent.model_requests q INNER JOIN agent.turns t ON t.turn_id=q.turn_id WHERE t.agent_run_id=@run AND LOWER(CONVERT(varchar(64),HASHBYTES('SHA2_256',q.request_body),2))<>q.request_body_sha256) AS bad_request_hashes,
        (SELECT COUNT(*) FROM agent.model_responses p INNER JOIN agent.model_requests q ON q.model_request_id=p.model_request_id INNER JOIN agent.turns t ON t.turn_id=q.turn_id WHERE t.agent_run_id=@run AND LOWER(CONVERT(varchar(64),HASHBYTES('SHA2_256',p.response_body),2))<>p.response_body_sha256) AS bad_response_hashes,
        (SELECT COUNT(*) FROM agent.model_requests q INNER JOIN agent.turns t ON t.turn_id=q.turn_id WHERE t.agent_run_id=@run AND CONVERT(nvarchar(max),q.request_body) LIKE '%response_format%') AS guided_request_count,
        (SELECT COUNT(*) FROM telemetry.traces WHERE trace_id=@trace AND finished_at_utc IS NULL) AS open_trace_count,
        (SELECT COUNT(*) FROM telemetry.spans WHERE trace_id=@trace AND finished_at_utc IS NULL) AS open_span_count,
        (SELECT COUNT(*) FROM telemetry.spans WHERE trace_id=@trace AND span_name='model.request' AND model_request_id IS NULL) AS unlinked_model_spans,
        (SELECT COUNT(*) FROM telemetry.spans WHERE trace_id=@trace AND span_name='tool.operation' AND tool_invocation_id IS NULL) AS unlinked_tool_spans,
        (SELECT COUNT(*) FROM telemetry.spans WHERE trace_id=@trace AND span_name IN ('prompt.assemble','decision.validate','decision.persist') AND turn_id IS NULL) AS unlinked_phase_spans;
    `);
  const row = observed.recordset[0]!;
  const toolRows = await pool.request()
    .input("run", sql.BigInt, result.agentRunId)
    .query<{ result_json: string; result_sha256: string; raw_result_path: string; raw_result_sha256: string; raw_result_bytes: number }>(`
      SELECT result_json,result_sha256,raw_result_path,raw_result_sha256,raw_result_bytes
      FROM agent.tool_invocations WHERE agent_run_id=@run ORDER BY tool_invocation_id;
    `);
  const toolHashChecks = await Promise.all(toolRows.recordset.map(async (tool) => {
    const raw = await readFile(tool.raw_result_path);
    return sha256(tool.result_json) === tool.result_sha256
      && sha256(raw) === tool.raw_result_sha256
      && raw.length === Number(tool.raw_result_bytes);
  }));
  const expectedWork = gateCase.expectedStatus === "complete" ? "complete" : gateCase.expectedStatus;
  const checks = {
    jobTerminal: row.job_status === (gateCase.expectedStatus === "complete" ? "complete" : "failed"),
    workTerminal: row.work_status === expectedWork,
    agentTerminal: row.agent_status === gateCase.expectedStatus,
    loopCounts: Number(row.recorded_model_turn_count) === result.modelTurnCount
      && Number(row.recorded_tool_call_count) === result.toolCallCount
      && Number(row.recorded_tool_result_chars) === result.toolResultChars,
    turns: Number(row.turn_count) === gateCase.expectedTurns,
    requestsAndResponses: Number(row.request_count) === gateCase.expectedTurns && Number(row.response_count) === gateCase.expectedTurns,
    tools: Number(row.tool_count) === gateCase.expectedTools,
    cache: Number(row.cache_count) === gateCase.expectedCacheHits,
    snapshotMisses: Number(row.snapshot_miss_count) === gateCase.expectedSnapshotMisses,
    decisions: Number(row.decision_count) === gateCase.expectedDecisions,
    steps: Number(row.step_count) === gateCase.expectedTurns * 3 + gateCase.expectedTools + gateCase.expectedDecisions * 2,
    validations: Number(row.validation_count) >= gateCase.expectedTurns * 3 + gateCase.expectedTools * 2 + gateCase.expectedDecisions * 4,
    actionSafety: Number(row.safe_proposal_count) === gateCase.expectedDecisions,
    byteHashes: Number(row.bad_request_hashes) === 0 && Number(row.bad_response_hashes) === 0 && toolHashChecks.every(Boolean),
    unconstrainedPrimary: Number(row.guided_request_count) === 0,
    traceClosed: Number(row.open_trace_count) === 0 && Number(row.open_span_count) === 0,
    telemetryLinked: Number(row.unlinked_model_spans) === 0 && Number(row.unlinked_tool_spans) === 0 && Number(row.unlinked_phase_spans) === 0,
  };
  if (Object.values(checks).some((value) => !value)) throw new Error(`${gateCase.name} database evidence failed: ${JSON.stringify({ checks, row })}`);
  const transitions = await pool.request()
    .input("work", sql.BigInt, fixture.workItemId)
    .query<{ from_state: string; to_state: string }>("SELECT from_state,to_state FROM ops.transitions WHERE entity_kind='work_item' AND entity_id=@work ORDER BY transition_id;");
  return { checks, observed: row, transitions: transitions.recordset };
}

async function verifyGlobal(pool: sql.ConnectionPool, jobIds: number[]): Promise<Record<string, unknown>> {
  const result = await pool.request()
    .input("jobs", sql.NVarChar(sql.MAX), JSON.stringify(jobIds))
    .query<Record<string, unknown>>(`
      WITH selected AS (SELECT CONVERT(bigint,value) AS job_id FROM OPENJSON(@jobs))
      SELECT
        (SELECT COUNT(*) FROM selected) AS expected_jobs,
        (SELECT COUNT(*) FROM control.jobs j INNER JOIN selected s ON s.job_id=j.job_id WHERE j.status IN ('complete','failed')) AS terminal_jobs,
        (SELECT COUNT(*) FROM control.job_attempts a INNER JOIN selected s ON s.job_id=a.job_id WHERE a.status IN ('complete','failed')) AS terminal_attempts,
        (SELECT COUNT(*) FROM ops.work_items w INNER JOIN selected s ON s.job_id=w.job_id WHERE w.lease_owner IS NOT NULL OR w.lease_token IS NOT NULL OR w.leased_until_utc IS NOT NULL) AS uncleared_leases,
        (SELECT COUNT(*) FROM agent.tool_invocations i INNER JOIN agent.agent_runs r ON r.agent_run_id=i.agent_run_id INNER JOIN selected s ON s.job_id=r.job_id WHERE i.raw_result_path IS NULL OR i.raw_result_sha256 IS NULL OR i.raw_result_bytes IS NULL) AS incomplete_tool_provenance,
        (SELECT COUNT(*) FROM ops.action_proposals p INNER JOIN agent.decisions d ON d.decision_id=p.decision_id INNER JOIN agent.agent_runs r ON r.agent_run_id=d.agent_run_id INNER JOIN selected s ON s.job_id=r.job_id WHERE p.caller_opted_in=1 OR p.executed_at_utc IS NOT NULL OR p.execution_status<>'not_executed') AS unsafe_actions;
    `);
  const row = result.recordset[0]!;
  const pass = Number(row.expected_jobs) === jobIds.length
    && Number(row.terminal_jobs) === jobIds.length
    && Number(row.terminal_attempts) === jobIds.length
    && Number(row.uncleared_leases) === 0
    && Number(row.incomplete_tool_provenance) === 0
    && Number(row.unsafe_actions) === 0;
  if (!pass) throw new Error(`Global agent-loop evidence failed: ${JSON.stringify(row)}`);
  return { ...row, disposition: "PASS" };
}

async function retainEmbeddingMetrics(phase: string) {
  const endpoint = new URL(config.inference.qwenEmbeddingBaseUrl);
  endpoint.pathname = "/metrics";
  endpoint.search = "";
  const response = await fetch(endpoint, { signal: AbortSignal.timeout(10_000) });
  const body = await response.text();
  if (!response.ok) throw new Error(`Embedding metrics ${phase} failed with HTTP ${response.status}`);
  const path = `${rawRoot}/embedding-${phase}.prom`;
  await atomicWrite(path, body);
  return { phase, path, bytes: Buffer.byteLength(body), sha256: sha256(body), summary: summarizeEmbeddingMetrics(body, embeddingProfile.modelId) };
}

async function findSnapshotFixture(pool: sql.ConnectionPool): Promise<{ episodeId: string; tool: string; arguments: Record<string, unknown> }> {
  const result = await pool.request().query<{ episode_id: string; tool_id: string; canonical_args_json: string }>(`
    SELECT TOP (1) episode_id,tool_id,canonical_args_json
    FROM ingest.context_snapshots
    WHERE status='complete'
    ORDER BY context_snapshot_id;
  `);
  const row = result.recordset[0];
  if (row === undefined) throw new Error("Agent-loop gate requires at least one captured context snapshot");
  return { episodeId: row.episode_id, tool: row.tool_id, arguments: JSON.parse(row.canonical_args_json) as Record<string, unknown> };
}

async function recoverInterruptedFixtures(pool: sql.ConnectionPool, id: string): Promise<void> {
  await pool.request()
    .input("prefix", sql.NVarChar(1000), `%${id}%`)
    .query(`
      DECLARE @recovered TABLE(work_item_id bigint,from_state varchar(40));
      UPDATE item SET status='stopped',lease_owner=NULL,lease_token=NULL,leased_until_utc=NULL
      OUTPUT INSERTED.work_item_id,DELETED.status INTO @recovered
      FROM ops.work_items item
      INNER JOIN control.jobs job ON job.job_id=item.job_id
      INNER JOIN control.job_attempts attempt ON attempt.job_id=job.job_id
      WHERE job.run_kind='agent_loop_gate' AND attempt.request_manifest_json LIKE @prefix
        AND item.status NOT IN ('complete','contract_rejected','policy_rejected','model_timeout','tool_timeout','stopped');
      INSERT ops.transitions(entity_kind,entity_id,from_state,to_state,actor,reason)
      SELECT 'work_item',work_item_id,from_state,'stopped','agent-loop-gate','recovered failed gate' FROM @recovered;
      UPDATE attempt SET status='failed',finished_at_utc=SYSUTCDATETIME(),error_class='gate_interrupted',error_detail='Recovered failed gate'
      FROM control.job_attempts attempt INNER JOIN control.jobs job ON job.job_id=attempt.job_id
      WHERE job.run_kind='agent_loop_gate' AND attempt.request_manifest_json LIKE @prefix AND attempt.status='running';
      UPDATE job SET status='failed',completed_at_utc=SYSUTCDATETIME(),error_class='gate_interrupted',error_detail='Recovered failed gate'
      FROM control.jobs job INNER JOIN control.job_attempts attempt ON attempt.job_id=job.job_id
      WHERE job.run_kind='agent_loop_gate' AND attempt.request_manifest_json LIKE @prefix AND job.status='running';
    `);
}

async function recoverIncompleteAgentGateFixtures(pool: sql.ConnectionPool): Promise<number> {
  const result = await pool.request().query<{ recovered: number }>(`
    DECLARE @recovered TABLE(work_item_id bigint,job_id bigint,from_state varchar(40));
    UPDATE item SET status='stopped',lease_owner=NULL,lease_token=NULL,leased_until_utc=NULL
    OUTPUT INSERTED.work_item_id,INSERTED.job_id,DELETED.status INTO @recovered
    FROM ops.work_items item
    INNER JOIN control.jobs job ON job.job_id=item.job_id
    WHERE job.run_kind='agent_loop_gate' AND job.status IN ('pending','running')
      AND item.status NOT IN ('complete','contract_rejected','policy_rejected','model_timeout','tool_timeout','stopped');
    INSERT ops.transitions(entity_kind,entity_id,from_state,to_state,actor,reason)
    SELECT 'work_item',work_item_id,from_state,'stopped','agent-loop-gate','recovered interrupted development gate'
    FROM @recovered;
    UPDATE attempt SET status='failed',finished_at_utc=SYSUTCDATETIME(),
      error_class='interrupted_gate',error_detail='Recovered before a new agent-loop gate run'
    FROM control.job_attempts attempt
    INNER JOIN @recovered recovered ON recovered.job_id=attempt.job_id
    WHERE attempt.status='running';
    UPDATE job SET status='failed',completed_at_utc=SYSUTCDATETIME(),
      error_class='interrupted_gate',error_detail='Recovered before a new agent-loop gate run'
    FROM control.jobs job
    INNER JOIN @recovered recovered ON recovered.job_id=job.job_id
    WHERE job.status IN ('pending','running');
    SELECT COUNT(*) AS recovered FROM @recovered;
  `);
  return Number(result.recordset[0]?.recovered ?? 0);
}

function assertResult(gateCase: GateCase, result: AgentLoopResult): void {
  const observed = {
    status: result.status,
    turns: result.modelTurnCount,
    tools: result.toolCallCount,
    cache: result.cacheHitCount,
    misses: result.snapshotMissCount,
    decisions: result.decisionId === null ? 0 : 1,
  };
  const expected = {
    status: gateCase.expectedStatus,
    turns: gateCase.expectedTurns,
    tools: gateCase.expectedTools,
    cache: gateCase.expectedCacheHits,
    misses: gateCase.expectedSnapshotMisses,
    decisions: gateCase.expectedDecisions,
  };
  if (canonicalJson(observed) !== canonicalJson(expected)) throw new Error(`${gateCase.name} result mismatch: ${JSON.stringify({ observed, expected })}`);
}

function syntheticPacket(episodeId: string) {
  return {
    packetVersion: "agent-loop-gate-v1",
    episodeId,
    anchorTimeUtc: "2026-08-23T00:00:00.000Z",
    sourceEvents: [{ sourceKind: "synthetic", eventName: "gate_signal", occurredAtUtc: "2026-08-23T00:00:00.000Z", message: "Bounded development evidence only." }],
    recentHistory: { sameFingerprint5m: 1, sameClass1h: 2, openRelatedIncidents: 0 },
    availableTools: [...toolNames],
    sourceDiagnostics: { linkedEventCount: 1, sources: ["synthetic"] },
    redactions: [],
  };
}

function gateArmId(arm: PrimaryAgentArm): string {
  return `dev-agent-loop-${arm.toLowerCase()}-v1`;
}

async function routeFakeModel(request: IncomingMessage, response: ServerResponse, snapshot: { tool: string; arguments: Record<string, unknown> }): Promise<void> {
  try {
    const body = await requestBody(request);
    const parsed = JSON.parse(body) as { messages?: Array<{ role?: string; content?: string }>; response_format?: unknown };
    if (parsed.response_format !== undefined) return openAi(response, JSON.stringify({ error: "primary gate unexpectedly used guided decoding" }), 400);
    const name = (request.url ?? "/").split("?")[0]!.replace(/^\//, "");
    const messages = parsed.messages ?? [];
    if (messages.some((message) => message.role === "system")) return openAi(response, JSON.stringify({ error: "system message forbidden" }), 400);
    const turn = messages.filter((message) => message.role === "assistant").length;
    let value: unknown;
    switch (name) {
      case "hybrid-cache-success":
        value = turn < 2
          ? { kind: "tool_request", tool: "runbook_search", arguments: { query: "transaction log full 9002 active transaction", topK: 1, corpusId: "primary-v1" } }
          : decision("transaction_log_full", false, returnedChunk(messages));
        break;
      case "snapshot-hit-success":
        value = turn === 0
          ? { kind: "tool_request", tool: snapshot.tool, arguments: snapshot.arguments }
          : decision("blocking", false);
        break;
      case "snapshot-miss-success":
        value = turn === 0
          ? { kind: "tool_request", tool: "get_log_space", arguments: { databaseName: "LW_NOT_CAPTURED" } }
          : decision("unknown_ambiguous", true);
        break;
      case "unknown-tool-policy":
        value = { kind: "tool_request", tool: "execute_sql", arguments: { sql: "select 1" } };
        break;
      case "invalid-arguments-policy":
        value = { kind: "tool_request", tool: "get_log_space", arguments: { databaseName: "master" } };
        break;
      case "contract-rejected":
        value = { kind: "decision", incidentClass: "not_a_class" };
        break;
      case "same-tool-budget":
        value = { kind: "tool_request", tool: "runbook_search", arguments: { query: "deadlock victim cycle retry", topK: 1, corpusId: "primary-v1" } };
        break;
      case "model-turn-budget":
        value = { kind: "tool_request", tool: "get_recent_incident_counts", arguments: { incidentClass: null, windowMinutes: turn + 1 } };
        break;
      default:
        return openAi(response, JSON.stringify({ error: "unknown fake route" }), 404);
    }
    openAi(response, JSON.stringify(value));
  } catch (error) {
    openAi(response, JSON.stringify({ error: safeError(error) }), 500);
  }
}

function decision(incidentClass: string, abstain: boolean, chunkId?: string) {
  return {
    kind: "decision",
    incidentClass,
    severity: incidentClass === "transaction_log_full" ? "high" : "medium",
    action: abstain ? "escalate_to_human" : "open_work_item",
    actionArguments: {},
    citedChunkIds: chunkId === undefined ? [] : [chunkId],
    confidence: abstain ? 0.3 : 0.8,
    abstain,
    correlationKey: "agent-loop-gate",
    summary: abstain ? "Evidence is insufficient; escalate for review." : "Bounded evidence supports this triage.",
    rationale: abstain ? "The frozen tool lookup returned no matching snapshot, so a confident diagnosis is not justified." : "The packet and approved bounded diagnostic agree; record a non-executing work proposal for review.",
  };
}

function returnedChunk(messages: Array<{ role?: string; content?: string }>): string {
  const last = [...messages].reverse().find((message) => message.role === "user" && message.content?.includes('"kind":"tool_result"'));
  if (last?.content === undefined) throw new Error("Fake model did not receive a tool result");
  const parsed = JSON.parse(last.content) as { result?: { rows?: Array<{ chunkId?: string }> } };
  const chunk = parsed.result?.rows?.[0]?.chunkId;
  if (typeof chunk !== "string") throw new Error("Fake model did not receive a returned chunk ID");
  return chunk;
}

function requestBody(request: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    request.on("data", (chunk: Buffer) => chunks.push(chunk));
    request.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    request.on("error", reject);
  });
}

function openAi(response: ServerResponse, content: string, status = 200): void {
  response.writeHead(status, { "content-type": "application/json", "x-request-id": randomUUID() });
  response.end(JSON.stringify({
    id: randomUUID(),
    choices: [{ finish_reason: "stop", message: { role: "assistant", content } }],
    usage: { prompt_tokens: 64, completion_tokens: 32, total_tokens: 96 },
  }));
}

function safeError(error: unknown): string {
  return (error as { message?: string }).message ?? String(error);
}
