import { randomUUID } from "node:crypto";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import sql from "mssql";
import { persistGatewayEvidence } from "../src/agent-evidence.js";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson } from "../src/hash.js";
import { callOpenAiCompatibleModel, type GatewayCallResult } from "../src/model-gateway.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal } from "../src/telemetry.js";

interface FixtureIdentity {
  jobId: number;
  jobAttemptId: number;
  workItemId: number;
  leaseToken: string;
  episodeId: string;
}

const cases = [
  { route: "success", expectedStatus: "success", expectedState: "decision_received", expectedError: null, maxRetries: 0, timeoutMs: 200, maxResponseBytes: 4096 },
  { route: "invalid-json", expectedStatus: "failed", expectedState: "contract_rejected", expectedError: "invalid_json", maxRetries: 0, timeoutMs: 200, maxResponseBytes: 4096 },
  { route: "invalid-envelope", expectedStatus: "failed", expectedState: "contract_rejected", expectedError: "invalid_json", maxRetries: 0, timeoutMs: 200, maxResponseBytes: 4096 },
  { route: "empty", expectedStatus: "failed", expectedState: "contract_rejected", expectedError: "empty_output", maxRetries: 0, timeoutMs: 200, maxResponseBytes: 4096 },
  { route: "schema", expectedStatus: "failed", expectedState: "contract_rejected", expectedError: "contract_schema", maxRetries: 0, timeoutMs: 200, maxResponseBytes: 4096 },
  { route: "http-error", expectedStatus: "failed", expectedState: "retryable_failure", expectedError: "http_error", maxRetries: 0, timeoutMs: 200, maxResponseBytes: 4096 },
  { route: "timeout", expectedStatus: "failed", expectedState: "model_timeout", expectedError: "timeout", maxRetries: 0, timeoutMs: 40, maxResponseBytes: 4096 },
  { route: "truncated", expectedStatus: "failed", expectedState: "contract_rejected", expectedError: "output_limit", maxRetries: 0, timeoutMs: 200, maxResponseBytes: 512 },
  { route: "retry", expectedStatus: "success", expectedState: "decision_received", expectedError: null, maxRetries: 1, timeoutMs: 200, maxResponseBytes: 4096 },
] as const;

const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const config = loadConfig();
const gateId = randomUUID();
const armId = "dev-model-client-gate-v1";
const modelProfileId = "qwen-smoke";
const decodeConfigId = "primary-json-v1";
const messages = [{ role: "system" as const, content: "Return a LogWarden JSON decision." }, { role: "user" as const, content: "Synthetic observability fixture; no production action." }];
const decode = { temperature: 0, max_tokens: 256, seed: 0 };
const routeCounts = new Map<string, number>();
const server = await startFakeGateway(routeCounts);
const address = server.address();
if (address === null || typeof address === "string") throw new Error("Fake gateway failed to bind TCP");
const baseUrl = `http://127.0.0.1:${address.port}`;
const pool = await connect(config.databases.lab, config.databases.controlName);
const caseReceipts: Array<Record<string, unknown>> = [];
let recoveredIncompleteFixtures = 0;

try {
  const campaignId = await ensureGateArm(pool, armId);
  recoveredIncompleteFixtures = await recoverIncompleteGateFixtures(pool);
  for (const testCase of cases) {
    const fixture = await createFixture(pool, {
      runId: run.runId, campaignId, gateId, route: testCase.route,
      modelProfileId, armId, decodeConfigId,
    });
    await transition(pool, fixture, "packet_loaded", "fake-gateway packet ready");
    await transition(pool, fixture, "model_requested", "fake-gateway request started");
    const created = await createComponentTelemetryJournal(runDirectory, run.runId, `model-client-gate-${testCase.route}`);
    const result = await callOpenAiCompatibleModel({
      endpoint: `${baseUrl}/${testCase.route}`,
      model: "logwarden-fake-model",
      messages,
      decode,
      toolRegistry: [],
      responseSchema: { contract: "logwarden-json-v1" },
      journal: created.journal,
      runDirectory,
      context: { jobId: fixture.jobId, episodeId: fixture.episodeId, attemptId: 1 },
      timeoutMs: testCase.timeoutMs,
      maxResponseBytes: testCase.maxResponseBytes,
      maxRetries: testCase.maxRetries,
      retryBackoffMs: 1,
    });
    assertExpected(testCase, result);
    const persisted = await persistGatewayEvidence(pool, {
      runId: run.runId,
      jobId: fixture.jobId,
      jobAttemptId: fixture.jobAttemptId,
      workItemId: fixture.workItemId,
      episodeId: fixture.episodeId,
      modelProfileId,
      agentArmId: armId,
      decodeConfigId,
      messages,
    }, result);
    await finalizeWorkItem(pool, fixture, result);
    const ingestion = await ingestTelemetryJournal(pool, run.runId, created.path, { batchSize: 4 });
    await linkTelemetry(pool, result, persisted.turnId, persisted.modelRequests);
    const observed = await verifyCase(pool, fixture, result, ingestion.validatedRecords);
    caseReceipts.push({
      route: testCase.route,
      status: result.status,
      terminalState: result.terminalState,
      errorClass: result.errorClass,
      traceId: result.traceId,
      attemptCount: result.attempts.length,
      rawRequestSha256: result.attempts.map((attempt) => attempt.requestBodySha256),
      rawResponseSha256: result.attempts.map((attempt) => attempt.responseBodySha256),
      ingestion,
      persisted,
      observed,
      disposition: "PASS",
    });
  }
  const retiredVerifiedRetryableFixtures = await retireVerifiedGateRetries(pool);
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    gateId,
    recoveredIncompleteFixtures,
    retiredVerifiedRetryableFixtures,
    fakeGateway: { address: "127.0.0.1", routeCounts: Object.fromEntries([...routeCounts].sort()) },
    cases: caseReceipts,
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(`${runDirectory}/telemetry/model-client-gate.json`, `${JSON.stringify(receipt, null, 2)}\n`);
  console.log(JSON.stringify({ runId: run.runId, gateId, caseCount: caseReceipts.length, disposition: "PASS", receiptSha256: receipt.receiptSha256 }, null, 2));
} finally {
  await pool.close();
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

async function ensureGateArm(pool: sql.ConnectionPool, armId: string): Promise<number> {
  const prompt = hashJson({ contract: "fake-gateway-observability-v1" });
  const tools = hashJson([]);
  const policy = hashJson({ policy: "no-execution-gate-v1" });
  const contract = hashJson({ contract: "logwarden-json-v1" });
  const arm = { prompt, tools, policy, contract, packetVersion: "synthetic-gate-v1", retrievalMode: "none", correlationMode: "provided" };
  await pool.request()
    .input("id", sql.VarChar(80), armId)
    .input("hash", sql.Char(64), hashJson(arm))
    .input("prompt", sql.Char(64), prompt)
    .input("tools", sql.Char(64), tools)
    .input("policy", sql.Char(64), policy)
    .input("contract", sql.Char(64), contract)
    .input("config", sql.NVarChar(sql.MAX), canonicalJson({ schemaVersion: 1, purpose: "development observability gate", noProductionAction: true }))
    .query(`
      IF NOT EXISTS (SELECT 1 FROM control.agent_arms WHERE agent_arm_id=@id)
        INSERT control.agent_arms
          (agent_arm_id,arm_hash,prompt_sha256,tool_registry_sha256,policy_sha256,
           contract_sha256,packet_version,retrieval_mode,correlation_mode,config_json)
        VALUES(@id,@hash,@prompt,@tools,@policy,@contract,'synthetic-gate-v1','none','provided',@config);
    `);
  const campaign = await pool.request().query<{ campaign_id: string }>("SELECT TOP (1) campaign_id FROM control.campaigns ORDER BY campaign_id;");
  if (campaign.recordset[0] === undefined) throw new Error("No development campaign exists");
  return Number(campaign.recordset[0].campaign_id);
}

async function createFixture(pool: sql.ConnectionPool, input: {
  runId: string; campaignId: number; gateId: string; route: string;
  modelProfileId: string; armId: string; decodeConfigId: string;
}): Promise<FixtureIdentity> {
  const episodeId = `gate-${input.route}-${input.gateId}`;
  const jobKey = hashJson({ gateId: input.gateId, route: input.route });
  const leaseToken = randomUUID();
  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    const job = await new sql.Request(transaction)
      .input("key", sql.Char(64), jobKey)
      .input("campaign", sql.BigInt, input.campaignId)
      .input("episode", sql.VarChar(120), episodeId)
      .input("model", sql.VarChar(80), input.modelProfileId)
      .input("arm", sql.VarChar(80), input.armId)
      .input("decode", sql.VarChar(40), input.decodeConfigId)
      .query<{ job_id: string }>(`
        INSERT control.jobs
          (job_key,campaign_id,run_kind,episode_id,model_profile_id,agent_arm_id,
           decode_config_id,sample_index,priority,status,attempt_count,started_at_utc)
        OUTPUT INSERTED.job_id
        VALUES(@key,@campaign,'fake_gateway_gate',@episode,@model,@arm,@decode,0,2000000000,'running',1,SYSUTCDATETIME());
      `);
    const jobId = Number(job.recordset[0]!.job_id);
    const attempt = await new sql.Request(transaction)
      .input("job", sql.BigInt, jobId)
      .input("manifest", sql.NVarChar(sql.MAX), canonicalJson({ gateId: input.gateId, route: input.route, synthetic: true }))
      .query<{ job_attempt_id: string }>(`
        INSERT control.job_attempts(job_id,attempt_number,worker_id,request_manifest_json,status)
        OUTPUT INSERTED.job_attempt_id
        VALUES(@job,1,'fake-gateway-gate',@manifest,'running');
      `);
    const jobAttemptId = Number(attempt.recordset[0]!.job_attempt_id);
    const work = await new sql.Request(transaction)
      .input("run", sql.VarChar(120), input.runId)
      .input("job", sql.BigInt, jobId)
      .input("episode", sql.VarChar(120), episodeId)
      .input("token", sql.UniqueIdentifier, leaseToken)
      .query<{ work_item_id: string }>(`
        INSERT ops.work_items
          (run_id,job_id,episode_id,status,lease_owner,lease_token,leased_until_utc,
           attempt_count,next_attempt_at_utc,priority)
        OUTPUT INSERTED.work_item_id
        VALUES(@run,@job,@episode,'leased','fake-gateway-gate',@token,
          DATEADD(MINUTE,10,SYSUTCDATETIME()),1,SYSUTCDATETIME(),2000000000);
      `);
    const workItemId = Number(work.recordset[0]!.work_item_id);
    await new sql.Request(transaction)
      .input("work", sql.BigInt, workItemId)
      .query("INSERT ops.transitions(entity_kind,entity_id,from_state,to_state,actor,reason) VALUES('work_item',@work,'pending','leased','fake-gateway-gate','synthetic gate lease');");
    await transaction.commit();
    return { jobId, jobAttemptId, workItemId, leaseToken, episodeId };
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
}

async function transition(pool: sql.ConnectionPool, fixture: FixtureIdentity, toState: string, reason: string): Promise<void> {
  await pool.request()
    .input("work_item_id", sql.BigInt, fixture.workItemId)
    .input("lease_token", sql.UniqueIdentifier, fixture.leaseToken)
    .input("to_state", sql.VarChar(40), toState)
    .input("actor", sql.VarChar(120), "fake-gateway-gate")
    .input("reason", sql.NVarChar(1000), reason)
    .execute("ops.usp_transition_work_item");
}

async function recoverIncompleteGateFixtures(pool: sql.ConnectionPool): Promise<number> {
  const result = await pool.request().query<{ recovered: number }>(`
    DECLARE @recovered TABLE(work_item_id bigint NOT NULL,from_state varchar(40) NOT NULL);
    UPDATE item
    SET status='stopped',lease_owner=NULL,lease_token=NULL,leased_until_utc=NULL,
        next_attempt_at_utc=NULL,completed_at_utc=COALESCE(item.completed_at_utc,SYSUTCDATETIME())
    OUTPUT INSERTED.work_item_id,DELETED.status INTO @recovered(work_item_id,from_state)
    FROM ops.work_items AS item
    INNER JOIN control.jobs AS job ON job.job_id=item.job_id
    WHERE job.run_kind='fake_gateway_gate'
      AND item.status NOT IN ('complete','contract_rejected','policy_rejected','model_timeout','tool_timeout','stopped');
    INSERT ops.transitions(entity_kind,entity_id,from_state,to_state,actor,reason)
    SELECT 'work_item',work_item_id,from_state,'stopped','fake-gateway-gate','recovered interrupted development gate'
    FROM @recovered;
    UPDATE attempt
    SET status='failed',finished_at_utc=SYSUTCDATETIME(),error_class='interrupted_gate',
        error_detail='Recovered before a new fake-gateway gate run'
    FROM control.job_attempts AS attempt
    INNER JOIN control.jobs AS job ON job.job_id=attempt.job_id
    WHERE job.run_kind='fake_gateway_gate' AND job.status='running';
    UPDATE control.jobs
    SET status='failed',completed_at_utc=SYSUTCDATETIME(),error_class='interrupted_gate',
        error_detail='Recovered before a new fake-gateway gate run'
    WHERE run_kind='fake_gateway_gate' AND status='running';
    SELECT COUNT(*) AS recovered FROM @recovered;
  `);
  return Number(result.recordset[0]?.recovered ?? 0);
}

async function retireVerifiedGateRetries(pool: sql.ConnectionPool): Promise<number> {
  const result = await pool.request().query<{ retired: number }>(`
    DECLARE @retired TABLE(work_item_id bigint NOT NULL,from_state varchar(40) NOT NULL);
    UPDATE item
    SET status='stopped',lease_owner=NULL,lease_token=NULL,leased_until_utc=NULL,
        next_attempt_at_utc=NULL,completed_at_utc=COALESCE(item.completed_at_utc,SYSUTCDATETIME())
    OUTPUT INSERTED.work_item_id,DELETED.status INTO @retired(work_item_id,from_state)
    FROM ops.work_items AS item
    INNER JOIN control.jobs AS job ON job.job_id=item.job_id
    WHERE job.run_kind='fake_gateway_gate' AND item.status='retryable_failure';
    INSERT ops.transitions(entity_kind,entity_id,from_state,to_state,actor,reason)
    SELECT 'work_item',work_item_id,from_state,'stopped','fake-gateway-gate',
      'verified retry semantics retained; synthetic fixture retired from shared queue'
    FROM @retired;
    SELECT COUNT(*) AS retired FROM @retired;
  `);
  return Number(result.recordset[0]?.retired ?? 0);
}

async function finalizeWorkItem(pool: sql.ConnectionPool, fixture: FixtureIdentity, result: GatewayCallResult): Promise<void> {
  if (result.status === "success" && result.terminalState === "decision_received") {
    for (const [state, reason] of [["decision_received", "structured decision received"], ["validated", "contract and policy passed"], ["persisted", "agent evidence committed"], ["complete", "fake-gateway gate complete"]] as const) {
      await transition(pool, fixture, state, reason);
    }
    return;
  }
  await transition(pool, fixture, result.terminalState, result.errorClass ?? "terminal model disposition");
}

async function linkTelemetry(
  pool: sql.ConnectionPool,
  result: GatewayCallResult,
  turnId: number,
  requests: Array<{ modelRequestId: number; spanId: string }>,
): Promise<void> {
  await pool.request()
    .input("trace", sql.Char(32), result.traceId)
    .input("span", sql.Char(16), result.rootSpanId)
    .input("turn", sql.BigInt, turnId)
    .query("UPDATE telemetry.spans SET turn_id=@turn WHERE trace_id=@trace AND span_id=@span;");
  for (const request of requests) {
    await pool.request()
      .input("trace", sql.Char(32), result.traceId)
      .input("span", sql.Char(16), request.spanId)
      .input("turn", sql.BigInt, turnId)
      .input("request", sql.BigInt, request.modelRequestId)
      .query("UPDATE telemetry.spans SET turn_id=@turn,model_request_id=@request WHERE trace_id=@trace AND span_id=@span;");
  }
}

async function verifyCase(pool: sql.ConnectionPool, fixture: FixtureIdentity, result: GatewayCallResult, journalRecords: number): Promise<Record<string, unknown>> {
  const observed = await pool.request()
    .input("job", sql.BigInt, fixture.jobId)
    .input("work", sql.BigInt, fixture.workItemId)
    .input("trace", sql.Char(32), result.traceId)
    .query<Record<string, number | string | null>>(`
      SELECT
        (SELECT status FROM control.jobs WHERE job_id=@job) AS job_status,
        (SELECT status FROM ops.work_items WHERE work_item_id=@work) AS work_status,
        (SELECT status FROM agent.agent_runs WHERE trace_id=@trace) AS agent_status,
        (SELECT COUNT(*) FROM agent.turns AS t INNER JOIN agent.agent_runs AS r ON r.agent_run_id=t.agent_run_id WHERE r.trace_id=@trace) AS turn_count,
        (SELECT COUNT(*) FROM agent.model_requests AS q INNER JOIN agent.turns AS t ON t.turn_id=q.turn_id INNER JOIN agent.agent_runs AS r ON r.agent_run_id=t.agent_run_id WHERE r.trace_id=@trace) AS request_count,
        (SELECT COUNT(*) FROM agent.model_responses AS p INNER JOIN agent.model_requests AS q ON q.model_request_id=p.model_request_id INNER JOIN agent.turns AS t ON t.turn_id=q.turn_id INNER JOIN agent.agent_runs AS r ON r.agent_run_id=t.agent_run_id WHERE r.trace_id=@trace) AS response_count,
        (SELECT COUNT(*) FROM agent.validation_events AS v INNER JOIN agent.agent_runs AS r ON r.agent_run_id=v.agent_run_id WHERE r.trace_id=@trace) AS validation_count,
        (SELECT COUNT(*) FROM agent.decisions AS d INNER JOIN agent.agent_runs AS r ON r.agent_run_id=d.agent_run_id WHERE r.trace_id=@trace) AS decision_count,
        (SELECT COUNT(*) FROM agent.agent_steps AS s INNER JOIN agent.agent_runs AS r ON r.agent_run_id=s.agent_run_id WHERE r.trace_id=@trace) AS step_count,
        (SELECT COUNT(*) FROM telemetry.journal_events WHERE trace_id=@trace) AS journal_count,
        (SELECT COUNT(*) FROM telemetry.traces WHERE trace_id=@trace AND finished_at_utc IS NULL) AS open_trace_count,
        (SELECT COUNT(*) FROM telemetry.spans WHERE trace_id=@trace AND finished_at_utc IS NULL) AS open_span_count,
        (SELECT COUNT(*) FROM telemetry.spans WHERE trace_id=@trace AND span_name='model.request' AND model_request_id IS NULL) AS unlinked_model_spans,
        (SELECT COUNT(*) FROM agent.model_requests AS q INNER JOIN agent.turns AS t ON t.turn_id=q.turn_id INNER JOIN agent.agent_runs AS r ON r.agent_run_id=t.agent_run_id WHERE r.trace_id=@trace AND LOWER(CONVERT(varchar(64),HASHBYTES('SHA2_256',q.request_body),2))<>q.request_body_sha256) AS bad_request_hashes,
        (SELECT COUNT(*) FROM agent.model_responses AS p INNER JOIN agent.model_requests AS q ON q.model_request_id=p.model_request_id INNER JOIN agent.turns AS t ON t.turn_id=q.turn_id INNER JOIN agent.agent_runs AS r ON r.agent_run_id=t.agent_run_id WHERE r.trace_id=@trace AND LOWER(CONVERT(varchar(64),HASHBYTES('SHA2_256',p.response_body),2))<>p.response_body_sha256) AS bad_response_hashes;
    `);
  const row = observed.recordset[0]!;
  const expectedWork = result.status === "success" ? "complete" : result.terminalState;
  const expectedAgent = result.status === "success" ? "complete" : result.terminalState === "contract_rejected" ? "contract_rejected" : result.terminalState === "model_timeout" ? "model_timeout" : "failed";
  const expectedDecisions = result.value?.kind === "decision" ? 1 : 0;
  const checks = {
    jobTerminal: row.job_status === (result.status === "success" ? "complete" : "failed"),
    workTerminal: row.work_status === expectedWork,
    agentTerminal: row.agent_status === expectedAgent,
    turns: Number(row.turn_count) === 1,
    requests: Number(row.request_count) === result.attempts.length,
    responses: Number(row.response_count) === result.attempts.length,
    validations: Number(row.validation_count) === result.attempts.length * 3,
    decisions: Number(row.decision_count) === expectedDecisions,
    steps: Number(row.step_count) === result.attempts.length + 1,
    journal: Number(row.journal_count) === journalRecords,
    closed: Number(row.open_trace_count) === 0 && Number(row.open_span_count) === 0,
    linked: Number(row.unlinked_model_spans) === 0,
    rawHashes: Number(row.bad_request_hashes) === 0 && Number(row.bad_response_hashes) === 0,
  };
  if (Object.values(checks).some((passed) => !passed)) throw new Error(`Model-client SQL reconciliation failed for ${fixture.episodeId}: ${JSON.stringify({ row, checks })}`);
  return { ...row, checks };
}

function assertExpected(testCase: typeof cases[number], result: GatewayCallResult): void {
  if (result.status !== testCase.expectedStatus || result.terminalState !== testCase.expectedState || result.errorClass !== testCase.expectedError) {
    throw new Error(`Unexpected ${testCase.route} disposition: ${JSON.stringify({ status: result.status, terminalState: result.terminalState, errorClass: result.errorClass })}`);
  }
  if (testCase.route === "retry" && result.attempts.length !== 2) throw new Error("Retry route did not retain both attempts");
}

async function startFakeGateway(counts: Map<string, number>): Promise<Server> {
  const server = createServer((request, response) => route(request, response, counts));
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => { server.off("error", reject); resolve(); });
  });
  return server;
}

function route(request: IncomingMessage, response: ServerResponse, counts: Map<string, number>): void {
  const name = request.url?.slice(1) ?? "";
  counts.set(name, (counts.get(name) ?? 0) + 1);
  request.resume();
  if (name === "timeout") {
    setTimeout(() => sendOpenAi(response, decision()), 150);
    return;
  }
  if (name === "http-error" || (name === "retry" && counts.get(name) === 1)) {
    response.writeHead(503, { "content-type": "application/json" });
    response.end('{"error":"retry later"}');
    return;
  }
  if (name === "invalid-envelope") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end("not-json");
    return;
  }
  if (name === "invalid-json") return sendOpenAi(response, "{\"kind\":");
  if (name === "empty") return sendOpenAi(response, "");
  if (name === "schema") return sendOpenAi(response, { ...decision(), incidentClass: "not-a-class" });
  if (name === "truncated") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({ choices: [{ message: { content: "x".repeat(4096) } }] }));
    return;
  }
  sendOpenAi(response, decision());
}

function sendOpenAi(response: ServerResponse, value: unknown): void {
  if (response.destroyed) return;
  const content = typeof value === "string" ? value : JSON.stringify(value);
  response.writeHead(200, { "content-type": "application/json", "x-request-id": "fake-service-request" });
  response.end(JSON.stringify({
    id: "fake-completion",
    choices: [{ finish_reason: "stop", message: { role: "assistant", content } }],
    usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
  }));
}

function decision() {
  return {
    kind: "decision",
    incidentClass: "benign_noise",
    severity: "info",
    action: "no_action",
    actionArguments: {},
    citedChunkIds: [],
    confidence: 0.99,
    abstain: false,
    correlationKey: "synthetic-gate",
    summary: "Synthetic gate response.",
    rationale: "No operational action is permitted by this fixture.",
  };
}
