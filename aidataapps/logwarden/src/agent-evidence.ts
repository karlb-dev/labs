import sql from "mssql";
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
         response_headers_at_utc,first_content_at_utc,body_finished_at_utc,client_elapsed_ms,
         headers_wait_ms,body_read_ms,status,error_class,error_detail)
      OUTPUT INSERTED.model_request_id
      VALUES(@turn,@client,@endpoint,@endpoint_hash,@messages,@prompt,@tools,@schema,@decode,
        @body,@body_hash,@bytes,@retry,@started,@headers,@first,@body_finished,@elapsed,
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
