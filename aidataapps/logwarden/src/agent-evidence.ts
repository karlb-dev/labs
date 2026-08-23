import sql from "mssql";
import type { AgentResponse } from "./contracts.js";
import { canonicalJson, hashJson } from "./hash.js";
import type { GatewayAttemptEvidence, GatewayCallResult, GatewayMessage } from "./model-gateway.js";

export interface GatewayEvidenceIdentity {
  runId: string;
  jobId: number;
  jobAttemptId: number;
  workItemId: number;
  episodeId: string;
  modelProfileId: string;
  agentArmId: string;
  decodeConfigId: string;
  messages: GatewayMessage[];
}

export interface PersistedGatewayEvidence {
  agentRunId: number;
  turnId: number;
  decisionId: number | null;
  modelRequests: Array<{ modelRequestId: number; modelResponseId: number; clientRequestId: string; spanId: string }>;
}

export async function persistGatewayEvidence(
  pool: sql.ConnectionPool,
  identity: GatewayEvidenceIdentity,
  result: GatewayCallResult,
): Promise<PersistedGatewayEvidence> {
  if (result.attempts.length === 0) throw new Error("Cannot persist gateway evidence without an attempt");
  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    const first = result.attempts[0]!;
    const last = result.attempts.at(-1)!;
    const finishedAt = last.parseFinishedAtUtc ?? last.bodyFinishedAtUtc ?? new Date().toISOString();
    const agentStatus = agentStatusFor(result);
    const insertedRun = await new sql.Request(transaction)
      .input("run", sql.VarChar(120), identity.runId)
      .input("job", sql.BigInt, identity.jobId)
      .input("attempt", sql.BigInt, identity.jobAttemptId)
      .input("work", sql.BigInt, identity.workItemId)
      .input("episode", sql.VarChar(120), identity.episodeId)
      .input("model", sql.VarChar(80), identity.modelProfileId)
      .input("arm", sql.VarChar(80), identity.agentArmId)
      .input("decode", sql.VarChar(40), identity.decodeConfigId)
      .input("trace", sql.Char(32), result.traceId)
      .input("contract", sql.Char(64), result.identities.responseSchemaSha256)
      .input("tools", sql.Char(64), result.identities.toolRegistrySha256)
      .input("status", sql.VarChar(32), agentStatus)
      .input("reason", sql.NVarChar(1000), result.errorDetail)
      .input("started", sql.DateTime2(7), new Date(first.startedAtUtc))
      .input("finished", sql.DateTime2(7), new Date(finishedAt))
      .input("elapsed", sql.Decimal(18, 3), result.elapsedMs)
      .query<{ agent_run_id: string }>(`
        INSERT agent.agent_runs
          (run_id,job_id,job_attempt_id,work_item_id,episode_id,model_profile_id,
           agent_arm_id,decode_config_id,trace_id,prompt_contract_sha256,tool_registry_sha256,
           status,terminal_reason,started_at_utc,finished_at_utc,elapsed_ms)
        OUTPUT INSERTED.agent_run_id
        VALUES(@run,@job,@attempt,@work,@episode,@model,@arm,@decode,@trace,@contract,@tools,
          @status,@reason,@started,@finished,@elapsed);
      `);
    const agentRunId = Number(insertedRun.recordset[0]!.agent_run_id);
    const promptBytes = Buffer.byteLength(canonicalJson(identity.messages));
    const insertedTurn = await new sql.Request(transaction)
      .input("run", sql.BigInt, agentRunId)
      .input("before", sql.VarChar(40), "model_requested")
      .input("after", sql.VarChar(40), result.terminalState)
      .input("messages", sql.Char(64), result.identities.messagesSha256)
      .input("bytes", sql.Int, promptBytes)
      .input("status", sql.VarChar(32), result.status)
      .input("started", sql.DateTime2(7), new Date(first.startedAtUtc))
      .input("finished", sql.DateTime2(7), new Date(finishedAt))
      .input("elapsed", sql.Decimal(18, 3), result.elapsedMs)
      .query<{ turn_id: string }>(`
        INSERT agent.turns
          (agent_run_id,turn_ordinal,state_before,state_after,messages_sha256,
           sent_messages_sha256,prompt_bytes,status,started_at_utc,finished_at_utc,elapsed_ms)
        OUTPUT INSERTED.turn_id
        VALUES(@run,0,@before,@after,@messages,@messages,@bytes,@status,@started,@finished,@elapsed);
      `);
    const turnId = Number(insertedTurn.recordset[0]!.turn_id);

    await new sql.Request(transaction)
      .input("run", sql.BigInt, agentRunId)
      .input("turn", sql.BigInt, turnId)
      .input("span", sql.Char(16), result.rootSpanId)
      .input("status", sql.VarChar(32), result.status)
      .input("started", sql.DateTime2(7), new Date(first.startedAtUtc))
      .input("finished", sql.DateTime2(7), new Date(finishedAt))
      .input("duration", sql.Decimal(18, 3), result.elapsedMs)
      .input("attrs", sql.NVarChar(sql.MAX), canonicalJson({ operationId: result.operationId, terminalState: result.terminalState, identities: result.identities }))
      .input("error", sql.VarChar(80), result.errorClass)
      .input("detail", sql.NVarChar(sql.MAX), result.errorDetail)
      .query(`
        INSERT agent.agent_steps
          (agent_run_id,turn_id,step_ordinal,step_kind,span_id,parent_span_id,status,
           started_at_utc,finished_at_utc,duration_ms,attributes_json,error_class,error_detail)
        VALUES(@run,@turn,0,'model_operation',@span,NULL,@status,@started,@finished,@duration,@attrs,@error,@detail);
      `);

    const modelRequests: PersistedGatewayEvidence["modelRequests"] = [];
    for (const [index, attempt] of result.attempts.entries()) {
      const requestId = await insertAttempt(transaction, identity, result, agentRunId, turnId, index + 1, attempt);
      modelRequests.push(requestId);
    }

    let decisionId: number | null = null;
    if (result.value?.kind === "decision") {
      const decisionJson = canonicalJson(result.value);
      const insertedDecision = await new sql.Request(transaction)
        .input("run", sql.BigInt, agentRunId)
        .input("turn", sql.BigInt, turnId)
        .input("episode", sql.VarChar(120), identity.episodeId)
        .input("class", sql.VarChar(80), result.value.incidentClass)
        .input("severity", sql.VarChar(24), result.value.severity)
        .input("action", sql.VarChar(100), result.value.action)
        .input("confidence", sql.Decimal(9, 6), result.value.confidence)
        .input("abstained", sql.Bit, result.value.abstain)
        .input("summary", sql.NVarChar(2000), result.value.summary)
        .input("correlation", sql.VarChar(160), result.value.correlationKey)
        .input("json", sql.NVarChar(sql.MAX), decisionJson)
        .input("hash", sql.Char(64), hashJson(result.value))
        .query<{ decision_id: string }>(`
          INSERT agent.decisions
            (agent_run_id,turn_id,episode_id,incident_class,severity,action,confidence,
             abstained,summary,correlation_key,decision_json,decision_sha256,contract_valid,policy_valid)
          OUTPUT INSERTED.decision_id
          VALUES(@run,@turn,@episode,@class,@severity,@action,@confidence,@abstained,
            @summary,@correlation,@json,@hash,1,1);
        `);
      decisionId = Number(insertedDecision.recordset[0]!.decision_id);
      await new sql.Request(transaction)
        .input("run", sql.BigInt, agentRunId)
        .input("turn", sql.BigInt, turnId)
        .input("decision", sql.BigInt, decisionId)
        .input("hash", sql.Char(64), hashJson(result.value))
        .input("detail", sql.NVarChar(sql.MAX), canonicalJson({ action: result.value.action, allowed: true, gate: "model-client" }))
        .query(`
          INSERT agent.policy_events
            (agent_run_id,turn_id,decision_id,policy_version,policy_point,outcome,
             reason_code,input_sha256,detail_json)
          VALUES(@run,@turn,@decision,'policy-v1-building','decision_action','allow',
            'closed_action_enum',@hash,@detail);
        `);
    }

    const jobStatus = result.status === "success" ? "complete" : "failed";
    await new sql.Request(transaction)
      .input("job", sql.BigInt, identity.jobId)
      .input("attempt", sql.BigInt, identity.jobAttemptId)
      .input("status", sql.VarChar(24), jobStatus)
      .input("error", sql.VarChar(100), result.errorClass)
      .input("detail", sql.NVarChar(sql.MAX), result.errorDetail)
      .query(`
        UPDATE control.job_attempts
        SET status=@status,finished_at_utc=SYSUTCDATETIME(),error_class=@error,error_detail=@detail
        WHERE job_attempt_id=@attempt;
        UPDATE control.jobs
        SET status=@status,completed_at_utc=SYSUTCDATETIME(),error_class=@error,error_detail=@detail
        WHERE job_id=@job;
      `);
    await transaction.commit();
    return { agentRunId, turnId, decisionId, modelRequests };
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
}

async function insertAttempt(
  transaction: sql.Transaction,
  identity: GatewayEvidenceIdentity,
  result: GatewayCallResult,
  agentRunId: number,
  turnId: number,
  stepOrdinal: number,
  attempt: GatewayAttemptEvidence,
): Promise<{ modelRequestId: number; modelResponseId: number; clientRequestId: string; spanId: string }> {
  const finishedAt = attempt.parseFinishedAtUtc ?? attempt.bodyFinishedAtUtc ?? new Date().toISOString();
  await new sql.Request(transaction)
    .input("run", sql.BigInt, agentRunId)
    .input("turn", sql.BigInt, turnId)
    .input("ordinal", sql.Int, stepOrdinal)
    .input("span", sql.Char(16), attempt.spanId)
    .input("parent", sql.Char(16), result.rootSpanId)
    .input("status", sql.VarChar(32), attempt.status)
    .input("started", sql.DateTime2(7), new Date(attempt.startedAtUtc))
    .input("finished", sql.DateTime2(7), new Date(finishedAt))
    .input("duration", sql.Decimal(18, 3), attempt.clientElapsedMs)
    .input("attrs", sql.NVarChar(sql.MAX), canonicalJson({ clientRequestId: attempt.clientRequestId, retryOrdinal: attempt.retryOrdinal, httpStatus: attempt.httpStatus, timingAvailability: { connectWriteMs: "unavailable_from_fetch_api", serverTimings: "unavailable_without_request_correlated_service_export" } }))
    .input("error", sql.VarChar(80), attempt.errorClass)
    .input("detail", sql.NVarChar(sql.MAX), attempt.errorDetail)
    .query(`
      INSERT agent.agent_steps
        (agent_run_id,turn_id,step_ordinal,step_kind,span_id,parent_span_id,status,
         started_at_utc,finished_at_utc,duration_ms,attributes_json,error_class,error_detail)
      VALUES(@run,@turn,@ordinal,'model_request',@span,@parent,@status,@started,@finished,
        @duration,@attrs,@error,@detail);
    `);
  const insertedRequest = await new sql.Request(transaction)
    .input("turn", sql.BigInt, turnId)
    .input("client", sql.VarChar(160), attempt.clientRequestId)
    .input("endpoint", sql.NVarChar(1000), attempt.endpoint)
    .input("endpoint_hash", sql.Char(64), result.identities.endpointIdentitySha256)
    .input("messages", sql.Char(64), result.identities.messagesSha256)
    .input("prompt", sql.Char(64), result.identities.promptSha256)
    .input("tools", sql.Char(64), result.identities.toolRegistrySha256)
    .input("schema", sql.Char(64), result.identities.responseSchemaSha256)
    .input("decode", sql.Char(64), result.identities.decodeConfigSha256)
    .input("body", sql.VarBinary(sql.MAX), attempt.requestBody)
    .input("body_hash", sql.Char(64), attempt.requestBodySha256)
    .input("bytes", sql.Int, attempt.requestBody.length)
    .input("retry", sql.Int, attempt.retryOrdinal)
    .input("started", sql.DateTime2(7), new Date(attempt.startedAtUtc))
    .input("request_written", sql.DateTime2(7), dateOrNull(attempt.requestWriteFinishedAtUtc))
    .input("headers", sql.DateTime2(7), dateOrNull(attempt.responseHeadersAtUtc))
    .input("first", sql.DateTime2(7), dateOrNull(attempt.firstContentAtUtc))
    .input("body_finished", sql.DateTime2(7), dateOrNull(attempt.bodyFinishedAtUtc))
    .input("elapsed", sql.Decimal(18, 3), attempt.clientElapsedMs)
    .input("headers_ms", sql.Decimal(18, 3), attempt.headersWaitMs)
    .input("body_ms", sql.Decimal(18, 3), attempt.bodyReadMs)
    .input("status", sql.VarChar(32), attempt.status)
    .input("error", sql.VarChar(80), attempt.errorClass)
    .input("detail", sql.NVarChar(sql.MAX), attempt.errorDetail)
    .query<{ model_request_id: string }>(`
      INSERT agent.model_requests
        (turn_id,client_request_id,endpoint_uri,endpoint_identity_sha256,messages_sha256,
         prompt_sha256,tool_registry_sha256,response_schema_sha256,decode_config_sha256,
         request_body,request_body_sha256,request_bytes,retry_ordinal,started_at_utc,
         request_write_finished_at_utc,response_headers_at_utc,first_content_at_utc,
         body_finished_at_utc,client_elapsed_ms,
         headers_wait_ms,body_read_ms,status,error_class,error_detail)
      OUTPUT INSERTED.model_request_id
      VALUES(@turn,@client,@endpoint,@endpoint_hash,@messages,@prompt,@tools,@schema,@decode,
        @body,@body_hash,@bytes,@retry,@started,@request_written,@headers,@first,@body_finished,@elapsed,
        @headers_ms,@body_ms,@status,@error,@detail);
    `);
  const modelRequestId = Number(insertedRequest.recordset[0]!.model_request_id);
  const repairKind = attempt.repairKind === "fence_strip" ? "strip_code_fence" : attempt.repairKind === "leading_text_strip" ? "strip_leading_text" : attempt.repairKind;
  const insertedResponse = await new sql.Request(transaction)
    .input("request", sql.BigInt, modelRequestId)
    .input("http", sql.Int, attempt.httpStatus)
    .input("service", sql.NVarChar(300), attempt.serviceRequestId)
    .input("body", sql.VarBinary(sql.MAX), attempt.responseBody)
    .input("hash", sql.Char(64), attempt.responseBodySha256)
    .input("bytes", sql.Int, attempt.responseBody.length)
    .input("finish", sql.VarChar(80), attempt.finishReason)
    .input("prompt_tokens", sql.BigInt, attempt.promptTokens)
    .input("completion_tokens", sql.BigInt, attempt.completionTokens)
    .input("total_tokens", sql.BigInt, attempt.totalTokens)
    .input("provenance", sql.VarChar(32), attempt.totalTokens === null ? null : "server_reported")
    .input("parse_started", sql.DateTime2(7), dateOrNull(attempt.parseStartedAtUtc))
    .input("parse_finished", sql.DateTime2(7), dateOrNull(attempt.parseFinishedAtUtc))
    .input("parse_ms", sql.Decimal(18, 3), attempt.parseMs)
    .input("parse_status", sql.VarChar(32), attempt.parsedValue !== null ? "parsed" : attempt.parseMs === null ? "not_attempted" : "rejected")
    .input("repair", sql.VarChar(40), repairKind)
    .input("json", sql.NVarChar(sql.MAX), attempt.parsedValue === null ? null : canonicalJson(attempt.parsedValue))
    .input("error", sql.VarChar(80), attempt.errorClass)
    .input("detail", sql.NVarChar(sql.MAX), attempt.errorDetail)
    .query<{ model_response_id: string }>(`
      INSERT agent.model_responses
        (model_request_id,http_status,service_request_id,response_body,response_body_sha256,
         response_bytes,finish_reason,prompt_tokens,completion_tokens,total_tokens,
         token_count_provenance,parse_started_at_utc,parse_finished_at_utc,parse_ms,
         parse_status,repair_kind,structured_json,error_class,error_detail)
      OUTPUT INSERTED.model_response_id
      VALUES(@request,@http,@service,@body,@hash,@bytes,@finish,@prompt_tokens,
        @completion_tokens,@total_tokens,@provenance,@parse_started,@parse_finished,
        @parse_ms,@parse_status,@repair,@json,@error,@detail);
    `);
  const modelResponseId = Number(insertedResponse.recordset[0]!.model_response_id);
  for (const validation of validationsFor(attempt)) {
    await new sql.Request(transaction)
      .input("run", sql.BigInt, agentRunId)
      .input("turn", sql.BigInt, turnId)
      .input("response", sql.BigInt, modelResponseId)
      .input("layer", sql.VarChar(40), validation.layer)
      .input("rule", sql.VarChar(120), validation.rule)
      .input("outcome", sql.VarChar(24), validation.outcome)
      .input("repair", sql.VarChar(40), validation.repair)
      .input("detail", sql.NVarChar(sql.MAX), canonicalJson(validation.detail))
      .query(`
        INSERT agent.validation_events
          (agent_run_id,turn_id,model_response_id,layer,rule_id,outcome,repair_kind,detail_json)
        VALUES(@run,@turn,@response,@layer,@rule,@outcome,@repair,@detail);
      `);
  }
  return { modelRequestId, modelResponseId, clientRequestId: attempt.clientRequestId, spanId: attempt.spanId };
}

function validationsFor(attempt: GatewayAttemptEvidence): Array<{ layer: string; rule: string; outcome: string; repair: string | null; detail: Record<string, unknown> }> {
  const httpPass = attempt.httpStatus !== null && attempt.httpStatus >= 200 && attempt.httpStatus < 300;
  const bodyPass = attempt.responseBody.length > 0 && !attempt.truncated;
  const contractPass = attempt.parsedValue !== null;
  return [
    { layer: "http", rule: "http_status_2xx", outcome: httpPass ? "pass" : "fail", repair: null, detail: { httpStatus: attempt.httpStatus, errorClass: attempt.errorClass } },
    { layer: "response_body", rule: "nonempty_within_limit", outcome: bodyPass ? "pass" : "fail", repair: null, detail: { bytes: attempt.responseBody.length, truncated: attempt.truncated } },
    { layer: "contract", rule: "agent_response_v1", outcome: contractPass ? (attempt.repairKind === "none" ? "pass" : "repaired") : "fail", repair: attempt.repairKind, detail: { errorClass: attempt.errorClass, parsedKind: attempt.parsedValue?.kind ?? null } },
  ];
}

function agentStatusFor(result: GatewayCallResult): string {
  if (result.status === "success") return "complete";
  if (result.terminalState === "contract_rejected") return "contract_rejected";
  if (result.terminalState === "model_timeout") return "model_timeout";
  return "failed";
}

function dateOrNull(value: string | null): Date | null {
  return value === null ? null : new Date(value);
}

export interface ActiveAgentEvidenceIdentity extends Omit<GatewayEvidenceIdentity, "messages"> {
  traceId: string;
  promptContractSha256: string;
  toolRegistrySha256: string;
  startedAtUtc: string;
  loopBudget: {
    maxModelTurns: number;
    maxToolCalls: number;
    maxSameToolCalls: number;
    maxTotalToolResultChars: number;
    maxWallTimeSeconds: number;
  };
}

export interface PersistedTurnEvidence {
  turnId: number;
  nextStepOrdinal: number;
  modelRequests: PersistedGatewayEvidence["modelRequests"];
}

export interface ToolValidationEvidence {
  layer: string;
  ruleId: string;
  outcome: "pass" | "fail" | "warn" | "repaired";
  detail: Record<string, unknown>;
}

export interface AgentStepEvidenceInput {
  agentRunId: number;
  turnId: number | null;
  stepOrdinal: number;
  stepKind: string;
  spanId: string;
  parentSpanId: string | null;
  status: string;
  startedAtUtc: string;
  finishedAtUtc: string;
  durationMs: number;
  attributes: Record<string, unknown>;
  errorClass?: string | null;
  errorDetail?: string | null;
}

export interface PersistToolEvidenceInput {
  agentRunId: number;
  turnId: number;
  stepOrdinal: number;
  spanId: string;
  parentSpanId: string;
  toolCallId: string;
  toolName: string;
  canonicalArgs: Record<string, unknown>;
  canonicalArgsSha256: string;
  policyStatus: string;
  policyReason: string;
  executionMode: "replay" | "live" | "cache";
  cacheHit: boolean;
  snapshotMiss: boolean;
  startedAtUtc: string;
  finishedAtUtc: string;
  latencyMs: number;
  result: unknown;
  resultSha256: string;
  resultBytes: number;
  rowCount: number | null;
  truncated: boolean;
  status: string;
  errorClass: string | null;
  errorDetail: string | null;
  rawResultPath: string;
  rawResultSha256: string;
  rawResultBytes: number;
  retrievalRunId: number | null;
  validations: ToolValidationEvidence[];
}

type AgentDecision = Extract<AgentResponse, { kind: "decision" }>;

export async function createAgentRunEvidence(
  pool: sql.ConnectionPool,
  identity: ActiveAgentEvidenceIdentity,
): Promise<number> {
  const inserted = await pool.request()
    .input("run", sql.VarChar(120), identity.runId)
    .input("job", sql.BigInt, identity.jobId)
    .input("attempt", sql.BigInt, identity.jobAttemptId)
    .input("work", sql.BigInt, identity.workItemId)
    .input("episode", sql.VarChar(120), identity.episodeId)
    .input("model", sql.VarChar(80), identity.modelProfileId)
    .input("arm", sql.VarChar(80), identity.agentArmId)
    .input("decode", sql.VarChar(40), identity.decodeConfigId)
    .input("trace", sql.Char(32), identity.traceId)
    .input("contract", sql.Char(64), identity.promptContractSha256)
    .input("tools", sql.Char(64), identity.toolRegistrySha256)
    .input("started", sql.DateTime2(7), new Date(identity.startedAtUtc))
    .input("budget", sql.NVarChar(sql.MAX), canonicalJson(identity.loopBudget))
    .query<{ agent_run_id: string }>(`
      INSERT agent.agent_runs
        (run_id,job_id,job_attempt_id,work_item_id,episode_id,model_profile_id,
         agent_arm_id,decode_config_id,trace_id,prompt_contract_sha256,tool_registry_sha256,
         status,started_at_utc,loop_budget_json)
      OUTPUT INSERTED.agent_run_id
      VALUES(@run,@job,@attempt,@work,@episode,@model,@arm,@decode,@trace,@contract,@tools,
        'running',@started,@budget);
    `);
  return Number(inserted.recordset[0]!.agent_run_id);
}

export async function persistGatewayTurnEvidence(
  pool: sql.ConnectionPool,
  identity: GatewayEvidenceIdentity,
  agentRunId: number,
  turnOrdinal: number,
  stepOrdinal: number,
  parentSpanId: string,
  result: GatewayCallResult,
): Promise<PersistedTurnEvidence> {
  if (result.attempts.length === 0) throw new Error("Cannot persist a model turn without an attempt");
  const first = result.attempts[0]!;
  const last = result.attempts.at(-1)!;
  const finishedAt = last.parseFinishedAtUtc ?? last.bodyFinishedAtUtc ?? new Date().toISOString();
  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    const insertedTurn = await new sql.Request(transaction)
      .input("run", sql.BigInt, agentRunId)
      .input("ordinal", sql.Int, turnOrdinal)
      .input("after", sql.VarChar(40), result.terminalState)
      .input("messages", sql.Char(64), result.identities.messagesSha256)
      .input("bytes", sql.Int, Buffer.byteLength(canonicalJson(identity.messages)))
      .input("status", sql.VarChar(32), result.status)
      .input("started", sql.DateTime2(7), new Date(first.startedAtUtc))
      .input("finished", sql.DateTime2(7), new Date(finishedAt))
      .input("elapsed", sql.Decimal(18, 3), result.elapsedMs)
      .query<{ turn_id: string }>(`
        INSERT agent.turns
          (agent_run_id,turn_ordinal,state_before,state_after,messages_sha256,
           sent_messages_sha256,prompt_bytes,status,started_at_utc,finished_at_utc,elapsed_ms)
        OUTPUT INSERTED.turn_id
        VALUES(@run,@ordinal,'model_requested',@after,@messages,@messages,@bytes,@status,
          @started,@finished,@elapsed);
      `);
    const turnId = Number(insertedTurn.recordset[0]!.turn_id);
    await new sql.Request(transaction)
      .input("run", sql.BigInt, agentRunId)
      .input("turn", sql.BigInt, turnId)
      .input("ordinal", sql.Int, stepOrdinal)
      .input("span", sql.Char(16), result.rootSpanId)
      .input("parent", sql.Char(16), parentSpanId)
      .input("status", sql.VarChar(32), result.status)
      .input("started", sql.DateTime2(7), new Date(first.startedAtUtc))
      .input("finished", sql.DateTime2(7), new Date(finishedAt))
      .input("duration", sql.Decimal(18, 3), result.elapsedMs)
      .input("attrs", sql.NVarChar(sql.MAX), canonicalJson({ operationId: result.operationId, terminalState: result.terminalState, identities: result.identities }))
      .input("error", sql.VarChar(80), result.errorClass)
      .input("detail", sql.NVarChar(sql.MAX), result.errorDetail)
      .query(`
        INSERT agent.agent_steps
          (agent_run_id,turn_id,step_ordinal,step_kind,span_id,parent_span_id,status,
           started_at_utc,finished_at_utc,duration_ms,attributes_json,error_class,error_detail)
        VALUES(@run,@turn,@ordinal,'model_operation',@span,@parent,@status,@started,
          @finished,@duration,@attrs,@error,@detail);
      `);
    const modelRequests: PersistedGatewayEvidence["modelRequests"] = [];
    for (const [index, attempt] of result.attempts.entries()) {
      modelRequests.push(await insertAttempt(
        transaction,
        identity,
        result,
        agentRunId,
        turnId,
        stepOrdinal + index + 1,
        attempt,
      ));
    }
    await transaction.commit();
    return { turnId, nextStepOrdinal: stepOrdinal + result.attempts.length + 1, modelRequests };
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
}

export async function persistAgentStepEvidence(
  pool: sql.ConnectionPool,
  input: AgentStepEvidenceInput,
): Promise<number> {
  const inserted = await pool.request()
    .input("run", sql.BigInt, input.agentRunId)
    .input("turn", sql.BigInt, input.turnId)
    .input("ordinal", sql.Int, input.stepOrdinal)
    .input("kind", sql.VarChar(40), input.stepKind)
    .input("span", sql.Char(16), input.spanId)
    .input("parent", sql.Char(16), input.parentSpanId)
    .input("status", sql.VarChar(32), input.status)
    .input("started", sql.DateTime2(7), new Date(input.startedAtUtc))
    .input("finished", sql.DateTime2(7), new Date(input.finishedAtUtc))
    .input("duration", sql.Decimal(18, 3), input.durationMs)
    .input("attrs", sql.NVarChar(sql.MAX), canonicalJson(input.attributes))
    .input("error", sql.VarChar(80), input.errorClass ?? null)
    .input("detail", sql.NVarChar(sql.MAX), input.errorDetail ?? null)
    .query<{ agent_step_id: string }>(`
      INSERT agent.agent_steps
        (agent_run_id,turn_id,step_ordinal,step_kind,span_id,parent_span_id,status,
         started_at_utc,finished_at_utc,duration_ms,attributes_json,error_class,error_detail)
      OUTPUT INSERTED.agent_step_id
      VALUES(@run,@turn,@ordinal,@kind,@span,@parent,@status,@started,@finished,
        @duration,@attrs,@error,@detail);
    `);
  return Number(inserted.recordset[0]!.agent_step_id);
}

export async function persistToolEvidence(
  pool: sql.ConnectionPool,
  input: PersistToolEvidenceInput,
): Promise<number> {
  const transmittedResultJson = canonicalJson(input.result);
  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    await new sql.Request(transaction)
      .input("run", sql.BigInt, input.agentRunId)
      .input("turn", sql.BigInt, input.turnId)
      .input("ordinal", sql.Int, input.stepOrdinal)
      .input("span", sql.Char(16), input.spanId)
      .input("parent", sql.Char(16), input.parentSpanId)
      .input("status", sql.VarChar(32), input.status)
      .input("started", sql.DateTime2(7), new Date(input.startedAtUtc))
      .input("finished", sql.DateTime2(7), new Date(input.finishedAtUtc))
      .input("duration", sql.Decimal(18, 3), input.latencyMs)
      .input("attrs", sql.NVarChar(sql.MAX), canonicalJson({
        toolCallId: input.toolCallId,
        toolName: input.toolName,
        cacheHit: input.cacheHit,
        snapshotMiss: input.snapshotMiss,
        rawResultPath: input.rawResultPath,
        rawResultSha256: input.rawResultSha256,
        rawResultBytes: input.rawResultBytes,
        transmittedResultSha256: input.resultSha256,
        retrievalRunId: input.retrievalRunId,
      }))
      .input("error", sql.VarChar(80), input.errorClass)
      .input("detail", sql.NVarChar(sql.MAX), input.errorDetail)
      .query(`
        INSERT agent.agent_steps
          (agent_run_id,turn_id,step_ordinal,step_kind,span_id,parent_span_id,status,
           started_at_utc,finished_at_utc,duration_ms,attributes_json,error_class,error_detail)
        VALUES(@run,@turn,@ordinal,'tool_operation',@span,@parent,@status,@started,
          @finished,@duration,@attrs,@error,@detail);
      `);
    const inserted = await new sql.Request(transaction)
      .input("step", sql.BigInt, await stepId(transaction, input.agentRunId, input.stepOrdinal))
      .input("run", sql.BigInt, input.agentRunId)
      .input("turn", sql.BigInt, input.turnId)
      .input("call", sql.VarChar(160), input.toolCallId)
      .input("name", sql.VarChar(80), input.toolName)
      .input("args", sql.NVarChar(sql.MAX), canonicalJson(input.canonicalArgs))
      .input("args_hash", sql.Char(64), input.canonicalArgsSha256)
      .input("policy", sql.VarChar(32), input.policyStatus)
      .input("reason", sql.NVarChar(1000), input.policyReason)
      .input("mode", sql.VarChar(24), input.executionMode)
      .input("cache", sql.Bit, input.cacheHit)
      .input("miss", sql.Bit, input.snapshotMiss)
      .input("started", sql.DateTime2(7), new Date(input.startedAtUtc))
      .input("finished", sql.DateTime2(7), new Date(input.finishedAtUtc))
      .input("latency", sql.Decimal(18, 3), input.latencyMs)
      .input("result", sql.NVarChar(sql.MAX), transmittedResultJson)
      .input("hash", sql.Char(64), input.resultSha256)
      .input("bytes", sql.Int, input.resultBytes)
      .input("rows", sql.Int, input.rowCount)
      .input("truncated", sql.Bit, input.truncated)
      .input("status", sql.VarChar(32), input.status)
      .input("error", sql.VarChar(80), input.errorClass)
      .input("detail", sql.NVarChar(sql.MAX), input.errorDetail)
      .input("raw_path", sql.NVarChar(1000), input.rawResultPath)
      .input("raw_hash", sql.Char(64), input.rawResultSha256)
      .input("raw_bytes", sql.Int, input.rawResultBytes)
      .input("result_chars", sql.Int, transmittedResultJson.length)
      .input("retrieval", sql.BigInt, input.retrievalRunId)
      .query<{ tool_invocation_id: string }>(`
        INSERT agent.tool_invocations
          (agent_step_id,agent_run_id,turn_id,tool_call_id,tool_name,
           canonical_args_json,canonical_args_sha256,policy_status,policy_reason,
           execution_mode,cache_hit,snapshot_miss,started_at_utc,finished_at_utc,
           latency_ms,result_json,result_sha256,result_bytes,row_count,truncated,status,
           error_class,error_detail,raw_result_path,raw_result_sha256,raw_result_bytes,
           result_chars,retrieval_run_id)
        OUTPUT INSERTED.tool_invocation_id
        VALUES(@step,@run,@turn,@call,@name,@args,@args_hash,@policy,@reason,@mode,
          @cache,@miss,@started,@finished,@latency,@result,@hash,@bytes,@rows,@truncated,
          @status,@error,@detail,@raw_path,@raw_hash,@raw_bytes,@result_chars,@retrieval);
      `);
    const toolInvocationId = Number(inserted.recordset[0]!.tool_invocation_id);
    for (const validation of input.validations) {
      await new sql.Request(transaction)
        .input("run", sql.BigInt, input.agentRunId)
        .input("turn", sql.BigInt, input.turnId)
        .input("tool", sql.BigInt, toolInvocationId)
        .input("layer", sql.VarChar(40), validation.layer)
        .input("rule", sql.VarChar(120), validation.ruleId)
        .input("outcome", sql.VarChar(24), validation.outcome)
        .input("detail", sql.NVarChar(sql.MAX), canonicalJson(validation.detail))
        .query(`
          INSERT agent.validation_events
            (agent_run_id,turn_id,tool_invocation_id,layer,rule_id,outcome,detail_json)
          VALUES(@run,@turn,@tool,@layer,@rule,@outcome,@detail);
        `);
    }
    await transaction.commit();
    return toolInvocationId;
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
}

export async function persistDecisionEvidence(
  pool: sql.ConnectionPool,
  input: {
    agentRunId: number;
    turnId: number;
    episodeId: string;
    decision: AgentDecision;
    returnedChunks: ReadonlyMap<string, number | null>;
    policyVersion: string;
  },
): Promise<{ decisionId: number; actionProposalId: number }> {
  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    const decisionHash = hashJson(input.decision);
    const insertedDecision = await new sql.Request(transaction)
      .input("run", sql.BigInt, input.agentRunId)
      .input("turn", sql.BigInt, input.turnId)
      .input("episode", sql.VarChar(120), input.episodeId)
      .input("class", sql.VarChar(80), input.decision.incidentClass)
      .input("severity", sql.VarChar(24), input.decision.severity)
      .input("action", sql.VarChar(100), input.decision.action)
      .input("confidence", sql.Decimal(9, 6), input.decision.confidence)
      .input("abstained", sql.Bit, input.decision.abstain)
      .input("summary", sql.NVarChar(2000), input.decision.summary)
      .input("correlation", sql.VarChar(160), input.decision.correlationKey)
      .input("json", sql.NVarChar(sql.MAX), canonicalJson(input.decision))
      .input("hash", sql.Char(64), decisionHash)
      .query<{ decision_id: string }>(`
        INSERT agent.decisions
          (agent_run_id,turn_id,episode_id,incident_class,severity,action,confidence,
           abstained,summary,correlation_key,decision_json,decision_sha256,contract_valid,policy_valid)
        OUTPUT INSERTED.decision_id
        VALUES(@run,@turn,@episode,@class,@severity,@action,@confidence,@abstained,
          @summary,@correlation,@json,@hash,1,1);
      `);
    const decisionId = Number(insertedDecision.recordset[0]!.decision_id);
    for (const [ordinal, chunkId] of input.decision.citedChunkIds.entries()) {
      await new sql.Request(transaction)
        .input("decision", sql.BigInt, decisionId)
        .input("ordinal", sql.Int, ordinal + 1)
        .input("chunk", sql.VarChar(120), chunkId)
        .input("retrieval", sql.BigInt, input.returnedChunks.get(chunkId) ?? null)
        .query(`
          INSERT agent.decision_citations
            (decision_id,citation_ordinal,chunk_id,retrieval_run_id,returned_to_agent)
          VALUES(@decision,@ordinal,@chunk,@retrieval,1);
        `);
    }
    for (const policy of [
      { point: "decision_action", reason: "closed_action_enum", detail: { action: input.decision.action } },
      { point: "citation_resolution", reason: "all_citations_returned", detail: { citedChunkIds: input.decision.citedChunkIds } },
      { point: "unknown_abstention", reason: "unknown_requires_abstention", detail: { incidentClass: input.decision.incidentClass, abstain: input.decision.abstain } },
    ]) {
      await new sql.Request(transaction)
        .input("run", sql.BigInt, input.agentRunId)
        .input("turn", sql.BigInt, input.turnId)
        .input("decision", sql.BigInt, decisionId)
        .input("version", sql.VarChar(40), input.policyVersion)
        .input("point", sql.VarChar(80), policy.point)
        .input("reason", sql.VarChar(80), policy.reason)
        .input("hash", sql.Char(64), decisionHash)
        .input("detail", sql.NVarChar(sql.MAX), canonicalJson(policy.detail))
        .query(`
          INSERT agent.policy_events
            (agent_run_id,turn_id,decision_id,policy_version,policy_point,outcome,
             reason_code,input_sha256,detail_json)
          VALUES(@run,@turn,@decision,@version,@point,'allow',@reason,@hash,@detail);
        `);
    }
    for (const validation of [
      { layer: "decision", rule: "citation_resolution", detail: { citedChunkIds: input.decision.citedChunkIds, allReturned: true } },
      { layer: "decision", rule: "closed_action_allowlist", detail: { action: input.decision.action } },
      { layer: "decision", rule: "unknown_abstention_coherence", detail: { incidentClass: input.decision.incidentClass, abstain: input.decision.abstain } },
      { layer: "decision", rule: "summary_rationale_word_limits", detail: { summary: input.decision.summary, rationale: input.decision.rationale } },
    ]) {
      await new sql.Request(transaction)
        .input("run", sql.BigInt, input.agentRunId)
        .input("turn", sql.BigInt, input.turnId)
        .input("layer", sql.VarChar(40), validation.layer)
        .input("rule", sql.VarChar(120), validation.rule)
        .input("detail", sql.NVarChar(sql.MAX), canonicalJson(validation.detail))
        .query(`
          INSERT agent.validation_events(agent_run_id,turn_id,layer,rule_id,outcome,detail_json)
          VALUES(@run,@turn,@layer,@rule,'pass',@detail);
        `);
    }
    const actionKey = hashJson({ agentRunId: input.agentRunId, decisionHash, action: input.decision.action, arguments: input.decision.actionArguments });
    const insertedProposal = await new sql.Request(transaction)
      .input("decision", sql.BigInt, decisionId)
      .input("action", sql.VarChar(80), input.decision.action)
      .input("args", sql.NVarChar(sql.MAX), canonicalJson(input.decision.actionArguments))
      .input("key", sql.Char(64), actionKey)
      .query<{ action_proposal_id: string }>(`
        INSERT ops.action_proposals
          (decision_id,action_type,action_arguments_json,policy_status,policy_reason,
           caller_opted_in,execution_status,idempotency_key)
        OUTPUT INSERTED.action_proposal_id
        VALUES(@decision,@action,@args,'record_only','Lab 3 never executes remediation',
          0,'not_executed',@key);
      `);
    await transaction.commit();
    return { decisionId, actionProposalId: Number(insertedProposal.recordset[0]!.action_proposal_id) };
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
}

export async function finishAgentRunEvidence(
  pool: sql.ConnectionPool,
  input: {
    agentRunId: number;
    jobId: number;
    jobAttemptId: number;
    status: "complete" | "needs_human_review" | "contract_rejected" | "policy_rejected" | "model_timeout" | "tool_timeout" | "failed" | "stopped";
    terminalReason: string | null;
    elapsedMs: number;
    errorClass: string | null;
    errorDetail: string | null;
    modelTurnCount: number;
    toolCallCount: number;
    toolResultChars: number;
  },
): Promise<void> {
  const jobStatus = input.status === "complete" || input.status === "needs_human_review" ? "complete" : "failed";
  await pool.request()
    .input("run", sql.BigInt, input.agentRunId)
    .input("job", sql.BigInt, input.jobId)
    .input("attempt", sql.BigInt, input.jobAttemptId)
    .input("agent_status", sql.VarChar(32), input.status)
    .input("job_status", sql.VarChar(24), jobStatus)
    .input("reason", sql.NVarChar(1000), input.terminalReason)
    .input("elapsed", sql.Decimal(18, 3), input.elapsedMs)
    .input("error", sql.VarChar(100), input.errorClass)
    .input("detail", sql.NVarChar(sql.MAX), input.errorDetail)
    .input("model_turns", sql.Int, input.modelTurnCount)
    .input("tool_calls", sql.Int, input.toolCallCount)
    .input("tool_chars", sql.Int, input.toolResultChars)
    .query(`
      UPDATE agent.agent_runs
      SET status=@agent_status,terminal_reason=@reason,finished_at_utc=SYSUTCDATETIME(),elapsed_ms=@elapsed,
          model_turn_count=@model_turns,tool_call_count=@tool_calls,tool_result_chars=@tool_chars
      WHERE agent_run_id=@run AND status='running';
      UPDATE control.job_attempts
      SET status=@job_status,finished_at_utc=SYSUTCDATETIME(),error_class=@error,error_detail=@detail
      WHERE job_attempt_id=@attempt AND status='running';
      UPDATE control.jobs
      SET status=@job_status,completed_at_utc=SYSUTCDATETIME(),error_class=@error,error_detail=@detail
      WHERE job_id=@job AND status='running';
    `);
}

async function stepId(transaction: sql.Transaction, agentRunId: number, stepOrdinal: number): Promise<number> {
  const result = await new sql.Request(transaction)
    .input("run", sql.BigInt, agentRunId)
    .input("ordinal", sql.Int, stepOrdinal)
    .query<{ agent_step_id: string }>(`
      SELECT agent_step_id FROM agent.agent_steps
      WHERE agent_run_id=@run AND step_ordinal=@ordinal;
    `);
  const value = result.recordset[0]?.agent_step_id;
  if (value === undefined) throw new Error("Persisted agent step could not be resolved");
  return Number(value);
}
