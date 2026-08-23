import { readFile } from "node:fs/promises";
import sql from "mssql";
import { frozenArmIdentities } from "../src/campaign.js";
import { loadConfig } from "../src/config.js";
import { actions, incidentClasses, severities } from "../src/contracts.js";
import {
  loadDerivedBaselineConfig,
  majorityPrediction,
  oraclePrediction,
  retrievalOnlyPrediction,
  selectRouterCandidate,
  type DerivedPrediction,
} from "../src/derived-baselines.js";
import { canonicalJson, hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal } from "../src/telemetry.js";

const allowedRoles = ["dev", "calibration", "test_id", "test_variant_holdout", "test_unknown", "test_live_parity", "test_storm"] as const;
const allowedArms = ["B0-majority-no-action-v1", "B2-lexical-v1", "B2-vector-v1", "B2-hybrid-v1", "B3-oracle-packet-v1", "A-router"] as const;
type DerivedArm = typeof allowedArms[number];

interface TruthRow {
  episode_id: string;
  split_role: string;
  expected_class: string;
  expected_severity: string;
  expected_action: string;
  should_abstain: boolean;
  packet_sha256: string;
  truth_sha256: string;
}

interface StoredPrediction {
  prediction_id: string;
  agent_run_id: string | null;
  decision_id: string | null;
  model_profile_id: string | null;
  predicted_class: string | null;
  predicted_severity: string | null;
  predicted_action: string | null;
  confidence: number | null;
  abstained: boolean | null;
  outcome: "decision" | "abstention" | "failure";
  failure_stage: string | null;
  failure_class: string | null;
  complete_case_eligible: boolean;
  prediction_json: string;
}

interface CandidatePrediction {
  prediction: DerivedPrediction | null;
  outcome: "decision" | "abstention" | "failure";
  failureStage: string | null;
  failureClass: string | null;
  completeCaseEligible: boolean;
  agentRunId: number | null;
  decisionId: number | null;
  predictionJson: Record<string, unknown>;
}

const startedAtUtc = new Date().toISOString();
const roles = roleArgument();
const arms = armArgument();
const profiles = listArgument("--profiles", "");
if (arms.includes("A-router") && profiles.length === 0) throw new Error("A-router requires --profiles");
if (!arms.includes("A-router") && profiles.length > 0) throw new Error("--profiles is only valid when deriving A-router");

const baselineConfig = loadDerivedBaselineConfig();
const baselineConfigSha256 = hashJson(baselineConfig);
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const pool = await connect(config.databases.lab, config.databases.controlName, 600_000);
const created = await createComponentTelemetryJournal(runDirectory, run.runId, "derived-campaign-arms");
const root = await created.journal.startSpan("campaign.derive_arms", {}, { roles, arms, profiles, baselineConfigSha256 });

try {
  const campaign = await campaignState();
  if (roles.some((role) => role.startsWith("test_")) && !["frozen", "running", "complete"].includes(campaign.status)) {
    throw new Error("Test-role derived predictions are forbidden before campaign freeze");
  }
  await requireSchema();
  await registerArms(campaign.status);
  await validateProfiles();
  const truthRows = await loadTruth();
  if (truthRows.length === 0) throw new Error(`No valid packet truth exists for ${roles.join(",")}`);

  const outputs: Array<Record<string, unknown>> = [];
  let inserted = 0;
  for (const truth of truthRows) {
    for (const armId of arms.filter((arm) => arm !== "A-router")) {
      const candidate = await baselineCandidate(armId, truth);
      const persisted = await persistPrediction(campaign.campaignId, armId, null, truth, candidate);
      inserted += persisted.inserted ? 1 : 0;
      outputs.push(outputRow(armId, null, truth, candidate, persisted));
      await created.journal.record("point", "campaign.derived_prediction", {
        traceId: root.traceId, parentSpanId: root.spanId, episodeId: truth.episode_id, jobId: persisted.jobId,
      }, { armId, predictionId: persisted.predictionId, inserted: persisted.inserted, outcome: candidate.outcome }, { status: "success" });
    }
    if (arms.includes("A-router")) {
      for (const profile of profiles) {
        const candidate = await routerCandidate(truth, profile);
        const persisted = await persistPrediction(campaign.campaignId, "A-router", profile, truth, candidate);
        inserted += persisted.inserted ? 1 : 0;
        outputs.push(outputRow("A-router", profile, truth, candidate, persisted));
        await created.journal.record("point", "campaign.derived_prediction", {
          traceId: root.traceId, parentSpanId: root.spanId, episodeId: truth.episode_id, jobId: persisted.jobId,
        }, { armId: "A-router", modelProfileId: profile, predictionId: persisted.predictionId, inserted: persisted.inserted,
          outcome: candidate.outcome, derivedFromArm: candidate.predictionJson.derivedFromArm }, { status: "success" });
      }
    }
  }
  outputs.sort((left, right) => String(left.splitRole).localeCompare(String(right.splitRole))
    || String(left.modelProfileId).localeCompare(String(right.modelProfileId))
    || String(left.agentArmId).localeCompare(String(right.agentArmId))
    || String(left.episodeId).localeCompare(String(right.episodeId)));
  const orderedPredictionSetSha256 = hashJson(outputs.map((row) => ({
    episodeId: row.episodeId, modelProfileId: row.modelProfileId, agentArmId: row.agentArmId,
    predictionSha256: row.predictionSha256,
  })));
  const suffix = safeName([...roles, ...arms, ...profiles].join("-"));
  const tablePath = `${runDirectory}/tables/derived-campaign-arms-${suffix}.jsonl`;
  await atomicWrite(tablePath, outputs.map((row) => canonicalJson(row)).join("\n") + "\n", 0o600);
  await created.journal.endSpan(root, "success", { predictions: outputs.length, inserted, orderedPredictionSetSha256 });
  await created.journal.flush();
  const ingestion = await ingestTelemetryJournal(pool, run.runId, created.path, { batchSize: 100 });
  const counts = Object.fromEntries(arms.map((arm) => [arm, outputs.filter((row) => row.agentArmId === arm).length]));
  const receiptBody = {
    schemaVersion: 1, runId: run.runId, campaignId: campaign.campaignId, campaignStatus: campaign.status,
    startedAtUtc, finishedAtUtc: new Date().toISOString(), roles, arms, profiles,
    baselineConfigSha256, predictionCount: outputs.length, insertedThisInvocation: inserted,
    resumed: outputs.length - inserted, counts, orderedPredictionSetSha256, tablePath,
    telemetry: { journalPath: created.path, ingestion }, disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const receiptPath = `${runDirectory}/metrics/derived-campaign-arms-${suffix}.json`;
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  await appendExperimentLog(`Derived ${outputs.length} predictions for ${arms.join(",")} over ${roles.join(",")}${profiles.length > 0 ? ` and ${profiles.join(",")}` : ""}; receipt ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({ runId: run.runId, roles, arms, profiles, predictions: outputs.length, inserted, counts, receiptPath, receiptSha256: receipt.receiptSha256, disposition: "PASS" }, null, 2));
} catch (error) {
  const errorDetail = safeError(error);
  await created.journal.endSpan(root, "failed", { errorDetail }).catch(() => undefined);
  await created.journal.flush().catch(() => undefined);
  console.error(JSON.stringify({ runId: run.runId, roles, arms, profiles, errorDetail, disposition: "STOP_DATA" }, null, 2));
  process.exitCode = 1;
} finally {
  await pool.close();
}

async function campaignState(): Promise<{ campaignId: number; status: string }> {
  const result = await pool.request().input("run", sql.VarChar(120), run.runId).query<{ campaign_id: string; status: string }>(`
    SELECT campaign.campaign_id,campaign.status FROM control.campaigns campaign
    INNER JOIN control.runs run ON run.campaign_id=campaign.campaign_id WHERE run.run_id=@run;
  `);
  const row = result.recordset[0];
  if (row === undefined || !["building", "frozen", "running", "complete"].includes(row.status)) throw new Error("Run has no eligible campaign");
  return { campaignId: Number(row.campaign_id), status: row.status };
}

async function requireSchema(): Promise<void> {
  const result = await pool.request().query<{ migration_present: number }>(`
    SELECT CONVERT(int,CASE WHEN OBJECT_ID(N'eval.retrieval_benchmark_results',N'U') IS NULL THEN 0 ELSE 1 END) migration_present;
  `);
  if (Number(result.recordset[0]?.migration_present) !== 1) throw new Error("Migration 031 must be applied before derived campaign arms");
}

async function registerArms(campaignStatus: string): Promise<void> {
  const identities = frozenArmIdentities().filter((identity) => arms.includes(identity.armId as DerivedArm));
  if (identities.length !== arms.length) throw new Error("Derived arm registry is incomplete");
  for (const identity of identities) {
    const armHash = hashJson(identity);
    const existing = await pool.request().input("id", sql.VarChar(80), identity.armId)
      .query<{ arm_hash: string }>("SELECT arm_hash FROM control.agent_arms WHERE agent_arm_id=@id;");
    if (existing.recordset[0] !== undefined) {
      if (existing.recordset[0].arm_hash !== armHash) throw new Error(`Derived arm registry drift for ${identity.armId}`);
      continue;
    }
    if (campaignStatus !== "building") throw new Error(`Derived arm ${identity.armId} was not registered before campaign freeze`);
    await pool.request().input("id", sql.VarChar(80), identity.armId).input("hash", sql.Char(64), armHash)
      .input("prompt", sql.Char(64), identity.promptSha256).input("tools", sql.Char(64), identity.toolRegistrySha256)
      .input("policy", sql.Char(64), identity.policySha256).input("contract", sql.Char(64), identity.contractSha256)
      .input("packet", sql.VarChar(40), identity.packetVersion).input("retrieval", sql.VarChar(40), identity.retrievalMode)
      .input("correlation", sql.VarChar(40), identity.correlationMode).input("json", sql.NVarChar(sql.MAX), canonicalJson(identity)).query(`
        INSERT control.agent_arms(agent_arm_id,arm_hash,prompt_sha256,tool_registry_sha256,policy_sha256,
          contract_sha256,packet_version,retrieval_mode,correlation_mode,config_json)
        VALUES(@id,@hash,@prompt,@tools,@policy,@contract,@packet,@retrieval,@correlation,@json);
      `);
  }
}

async function validateProfiles(): Promise<void> {
  if (profiles.length === 0) return;
  const result = await pool.request().input("profiles", sql.NVarChar(sql.MAX), JSON.stringify(profiles)).query<{ model_profile_id: string }>(`
    SELECT profile.model_profile_id FROM control.model_profiles profile
    INNER JOIN OPENJSON(@profiles) selected ON CONVERT(varchar(80),selected.value)=profile.model_profile_id;
  `);
  const found = new Set(result.recordset.map((row) => row.model_profile_id));
  for (const profile of profiles) if (!found.has(profile)) throw new Error(`Unknown registered model profile ${profile}`);
}

async function loadTruth(): Promise<TruthRow[]> {
  const result = await pool.request().input("run", sql.VarChar(120), run.runId).input("roles", sql.NVarChar(sql.MAX), JSON.stringify(roles)).query<TruthRow>(`
    WITH selected_roles AS (SELECT CONVERT(varchar(40),value) value FROM OPENJSON(@roles))
    SELECT truth.episode_id,truth.split_role,truth.expected_class,truth.expected_severity,truth.expected_action,
      truth.should_abstain,packet.packet_sha256,truth.truth_sha256
    FROM eval.ground_truth_episodes truth INNER JOIN selected_roles role ON role.value=truth.split_role
    INNER JOIN ingest.incident_packets packet ON packet.episode_id=truth.episode_id AND packet.is_valid=1
    INNER JOIN workload.injection_executions execution ON execution.episode_id=truth.episode_id AND execution.run_id=@run
    ORDER BY truth.split_role,truth.scenario_group_id,truth.episode_id;
  `);
  return result.recordset;
}

async function baselineCandidate(armId: Exclude<DerivedArm, "A-router">, truth: TruthRow): Promise<CandidatePrediction> {
  if (armId === "B0-majority-no-action-v1") {
    const prediction = majorityPrediction(baselineConfig);
    return decisionCandidate(prediction, {
      schemaVersion: 1, predictionSource: "B0", baselineId: armId, episodeId: truth.episode_id,
      packetSha256: truth.packet_sha256, baselineConfigSha256, fitRole: baselineConfig.majority.fitRole,
      fitEpisodeCount: baselineConfig.majority.fitEpisodeCount, ...prediction,
    });
  }
  if (armId === "B3-oracle-packet-v1") {
    const prediction = oraclePrediction({
      expectedClass: truth.expected_class, expectedSeverity: truth.expected_severity,
      expectedAction: truth.expected_action, shouldAbstain: Boolean(truth.should_abstain),
    }, baselineConfig);
    return decisionCandidate(prediction, {
      schemaVersion: 1, predictionSource: "B3", baselineId: armId, episodeId: truth.episode_id,
      truthSha256: truth.truth_sha256, evaluatorOnly: true, fieldPolicy: baselineConfig.oracle.fieldPolicy, ...prediction,
    });
  }
  const arm = baselineConfig.retrievalOnly.arms.find((value) => value.armId === armId);
  if (arm === undefined) throw new Error(`Missing B2 config for ${armId}`);
  const benchmark = await pool.request().input("run", sql.VarChar(120), run.runId)
    .input("episode", sql.VarChar(120), truth.episode_id).input("mode", sql.VarChar(40), arm.retrievalMode).query<{
      retrieval_run_id: string; returned_runbooks_json: string; result_sha256: string; evaluator_only: boolean;
    }>(`
      SELECT retrieval_run_id,returned_runbooks_json,result_sha256,evaluator_only
      FROM eval.retrieval_benchmark_results WHERE run_id=@run AND episode_id=@episode AND retrieval_mode=@mode;
    `);
  const row = benchmark.recordset[0];
  if (row === undefined || !row.evaluator_only) throw new Error(`Missing evaluator-only retrieval result for ${truth.episode_id}/${arm.retrievalMode}`);
  const returned = parseRunbooks(row.returned_runbooks_json);
  const top = returned[0] === undefined ? null : await loadRunbookMetadata(returned[0]);
  const prediction = retrievalOnlyPrediction(top, baselineConfig);
  return decisionCandidate(prediction, {
    schemaVersion: 1, predictionSource: "B2", baselineId: armId, episodeId: truth.episode_id,
    packetSha256: truth.packet_sha256, baselineConfigSha256, retrievalMode: arm.retrievalMode,
    retrievalRunId: Number(row.retrieval_run_id), retrievalResultSha256: row.result_sha256,
    returnedRunbooks: returned, topRunbookMetadata: top, evaluatorOnlyRetrieval: true, ...prediction,
  });
}

async function loadRunbookMetadata(runbookId: string): Promise<{ runbookId: string; incidentClass: DerivedPrediction["incidentClass"]; severityFloor: DerivedPrediction["severity"] }> {
  const result = await pool.request().input("id", sql.VarChar(100), runbookId).query<{ runbook_id: string; incident_class: string; severity_floor: string }>(`
    SELECT runbook_id,incident_class,severity_floor FROM kb.runbooks WHERE runbook_id=@id AND enabled=1;
  `);
  const row = result.recordset[0];
  if (row === undefined) throw new Error(`Returned runbook metadata is missing for ${runbookId}`);
  return retrievalMetadata(runbookId, row.incident_class, row.severity_floor);
}

function retrievalMetadata(runbookId: string, incidentClass: string, severityFloor: string) {
  if (!(incidentClasses as readonly string[]).includes(incidentClass) || !(severities as readonly string[]).includes(severityFloor)) {
    throw new Error(`Runbook ${runbookId} has invalid class/severity metadata`);
  }
  return { runbookId, incidentClass: incidentClass as DerivedPrediction["incidentClass"], severityFloor: severityFloor as DerivedPrediction["severity"] };
}

async function routerCandidate(truth: TruthRow, profile: string): Promise<CandidatePrediction> {
  const result = await pool.request().input("episode", sql.VarChar(120), truth.episode_id).input("profile", sql.VarChar(80), profile)
    .input("b1", sql.VarChar(80), baselineConfig.router.rulesArmId).input("tools", sql.VarChar(80), baselineConfig.router.fallbackArmId)
    .query<StoredPrediction>(`
      SELECT prediction.prediction_id,prediction.agent_run_id,prediction.decision_id,prediction.model_profile_id,
        prediction.predicted_class,prediction.predicted_severity,prediction.predicted_action,prediction.confidence,
        prediction.abstained,prediction.outcome,prediction.failure_stage,prediction.failure_class,
        prediction.complete_case_eligible,prediction.prediction_json
      FROM eval.predictions prediction
      WHERE prediction.episode_id=@episode AND
        prediction.control_id IS NULL AND
        ((prediction.agent_arm_id=@b1 AND prediction.model_profile_id IS NULL)
          OR (prediction.agent_arm_id=@tools AND prediction.model_profile_id=@profile));
    `);
  const b1 = result.recordset.find((row) => row.model_profile_id === null);
  const tools = result.recordset.find((row) => row.model_profile_id === profile);
  if (b1 === undefined || tools === undefined) throw new Error(`Router source rows are incomplete for ${truth.episode_id}/${profile}`);
  const selected = selectRouterCandidate(
    { predictionId: Number(b1.prediction_id), outcome: b1.outcome, predictionJson: JSON.parse(b1.prediction_json) as Record<string, unknown> },
    { predictionId: Number(tools.prediction_id), outcome: tools.outcome, predictionJson: JSON.parse(tools.prediction_json) as Record<string, unknown> },
  );
  const source = selected.sourceArmId === "B1-rules-v1" ? b1 : tools;
  const costs = source.agent_run_id === null ? { modelCalls: 0, promptTokens: 0, completionTokens: 0, totalTokens: 0, elapsedMs: 0 }
    : await agentRunCosts(Number(source.agent_run_id));
  const prediction = source.outcome === "failure" ? null : storedPredictionValue(source);
  return {
    prediction,
    outcome: source.outcome,
    failureStage: source.failure_stage,
    failureClass: source.failure_class,
    completeCaseEligible: Boolean(source.complete_case_eligible),
    agentRunId: source.agent_run_id === null ? null : Number(source.agent_run_id),
    decisionId: source.decision_id === null ? null : Number(source.decision_id),
    predictionJson: {
      schemaVersion: 1, predictionSource: "derived_router", armId: "A-router", episodeId: truth.episode_id,
      modelProfileId: profile, packetSha256: truth.packet_sha256, baselineConfigSha256,
      derivedFromArm: selected.sourceArmId, sourcePredictionId: Number(source.prediction_id),
      b1PredictionId: Number(b1.prediction_id), b1Resolved: (JSON.parse(b1.prediction_json) as Record<string, unknown>).resolved === true,
      fallbackPredictionId: Number(tools.prediction_id), costAttribution: costs,
      outcome: source.outcome, failureStage: source.failure_stage, failureClass: source.failure_class,
      ...(prediction ?? { incidentClass: null, severity: null, action: null, confidence: null, abstain: null }),
    },
  };
}

async function agentRunCosts(agentRunId: number) {
  const result = await pool.request().input("id", sql.BigInt, agentRunId).query<{
    model_calls: number; prompt_tokens: string; completion_tokens: string; total_tokens: string; elapsed_ms: number;
  }>(`
    SELECT COUNT(response.model_response_id) model_calls,
      COALESCE(SUM(response.prompt_tokens),0) prompt_tokens,
      COALESCE(SUM(response.completion_tokens),0) completion_tokens,
      COALESCE(SUM(response.total_tokens),0) total_tokens,
      COALESCE(MAX(run.elapsed_ms),0) elapsed_ms
    FROM agent.agent_runs run LEFT JOIN agent.turns turn ON turn.agent_run_id=run.agent_run_id
    LEFT JOIN agent.model_requests request ON request.turn_id=turn.turn_id
    LEFT JOIN agent.model_responses response ON response.model_request_id=request.model_request_id
    WHERE run.agent_run_id=@id GROUP BY run.agent_run_id;
  `);
  const row = result.recordset[0];
  if (row === undefined) throw new Error(`Router source agent run ${agentRunId} is missing`);
  return { modelCalls: Number(row.model_calls), promptTokens: Number(row.prompt_tokens), completionTokens: Number(row.completion_tokens), totalTokens: Number(row.total_tokens), elapsedMs: Number(row.elapsed_ms) };
}

function decisionCandidate(prediction: DerivedPrediction, predictionJson: Record<string, unknown>): CandidatePrediction {
  return {
    prediction,
    outcome: prediction.abstain ? "abstention" : "decision",
    failureStage: null,
    failureClass: null,
    completeCaseEligible: true,
    agentRunId: null,
    decisionId: null,
    predictionJson,
  };
}

async function persistPrediction(
  campaignId: number, armId: DerivedArm, profile: string | null, truth: TruthRow, candidate: CandidatePrediction,
): Promise<{ jobId: number; predictionId: number; inserted: boolean }> {
  const predictionSource = armId === "B0-majority-no-action-v1" ? "B0" : armId.startsWith("B2-") ? "B2"
    : armId === "B3-oracle-packet-v1" ? "B3" : "derived_router";
  const jobKey = hashJson({ schemaVersion: 1, campaignId, runKind: predictionSource === "derived_router" ? "derived" : "baseline",
    episodeId: truth.episode_id, modelProfileId: profile, agentArmId: armId, predictionSource, sampleIndex: 0,
    baselineConfigSha256 });
  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    const jobResult = await new sql.Request(transaction).input("key", sql.Char(64), jobKey)
      .input("campaign", sql.BigInt, campaignId).input("kind", sql.VarChar(40), predictionSource === "derived_router" ? "derived" : "baseline")
      .input("episode", sql.VarChar(120), truth.episode_id).input("profile", sql.VarChar(80), profile)
      .input("arm", sql.VarChar(80), armId).query<{ job_id: string }>(`
        IF NOT EXISTS(SELECT 1 FROM control.jobs WHERE job_key=@key)
          INSERT control.jobs(job_key,campaign_id,run_kind,episode_id,model_profile_id,agent_arm_id,status,started_at_utc,completed_at_utc)
          VALUES(@key,@campaign,@kind,@episode,@profile,@arm,'complete',SYSUTCDATETIME(),SYSUTCDATETIME());
        SELECT job_id FROM control.jobs WHERE job_key=@key;
      `);
    const jobId = Number(jobResult.recordset[0]!.job_id);
    const prior = await new sql.Request(transaction).input("job", sql.BigInt, jobId)
      .query<{ prediction_id: string; prediction_json: string }>("SELECT prediction_id,prediction_json FROM eval.predictions WHERE job_id=@job;");
    if (prior.recordset[0] !== undefined) {
      if (canonicalJson(JSON.parse(prior.recordset[0].prediction_json)) !== canonicalJson(candidate.predictionJson)) throw new Error(`Derived prediction drift for ${truth.episode_id}/${profile ?? "baseline"}/${armId}`);
      await transaction.commit();
      return { jobId, predictionId: Number(prior.recordset[0].prediction_id), inserted: false };
    }
    const value = candidate.prediction;
    const inserted = await new sql.Request(transaction).input("job", sql.BigInt, jobId)
      .input("agent", sql.BigInt, candidate.agentRunId).input("decision", sql.BigInt, candidate.decisionId)
      .input("episode", sql.VarChar(120), truth.episode_id).input("profile", sql.VarChar(80), profile)
      .input("arm", sql.VarChar(80), armId).input("source", sql.VarChar(32), predictionSource)
      .input("class", sql.VarChar(80), value?.incidentClass ?? null).input("severity", sql.VarChar(24), value?.severity ?? null)
      .input("action", sql.VarChar(100), value?.action ?? null).input("confidence", sql.Decimal(9, 6), value?.confidence ?? null)
      .input("abstain", sql.Bit, value?.abstain ?? null).input("outcome", sql.VarChar(32), candidate.outcome)
      .input("stage", sql.VarChar(80), candidate.failureStage).input("failure", sql.VarChar(80), candidate.failureClass)
      .input("complete", sql.Bit, candidate.completeCaseEligible).input("json", sql.NVarChar(sql.MAX), canonicalJson(candidate.predictionJson))
      .query<{ prediction_id: string }>(`
        INSERT eval.predictions(job_id,agent_run_id,decision_id,episode_id,model_profile_id,agent_arm_id,prediction_source,
          predicted_class,predicted_severity,predicted_action,confidence,abstained,outcome,failure_stage,failure_class,
          eligible,complete_case_eligible,prediction_json)
        OUTPUT inserted.prediction_id VALUES(@job,@agent,@decision,@episode,@profile,@arm,@source,@class,@severity,@action,
          @confidence,@abstain,@outcome,@stage,@failure,1,@complete,@json);
      `);
    await transaction.commit();
    return { jobId, predictionId: Number(inserted.recordset[0]!.prediction_id), inserted: true };
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
}

function storedPredictionValue(source: StoredPrediction): DerivedPrediction {
  if (source.predicted_class === null || !(incidentClasses as readonly string[]).includes(source.predicted_class)
      || source.predicted_severity === null || !(severities as readonly string[]).includes(source.predicted_severity)
      || source.predicted_action === null || !(actions as readonly string[]).includes(source.predicted_action)
      || source.confidence === null || source.confidence < 0 || source.confidence > 1 || source.abstained === null) {
    throw new Error(`Router source prediction ${source.prediction_id} is incomplete or violates the decision contract`);
  }
  return {
    incidentClass: source.predicted_class as DerivedPrediction["incidentClass"],
    severity: source.predicted_severity as DerivedPrediction["severity"],
    action: source.predicted_action as DerivedPrediction["action"],
    confidence: Number(source.confidence),
    abstain: Boolean(source.abstained),
  };
}

function outputRow(
  armId: DerivedArm, profile: string | null, truth: TruthRow, candidate: CandidatePrediction,
  persisted: { jobId: number; predictionId: number; inserted: boolean },
) {
  return {
    episodeId: truth.episode_id, splitRole: truth.split_role, modelProfileId: profile, agentArmId: armId,
    jobId: persisted.jobId, predictionId: persisted.predictionId, inserted: persisted.inserted,
    outcome: candidate.outcome, predictedClass: candidate.prediction?.incidentClass ?? null,
    predictedSeverity: candidate.prediction?.severity ?? null, predictedAction: candidate.prediction?.action ?? null,
    abstained: candidate.prediction?.abstain ?? null, predictionSha256: hashJson(candidate.predictionJson),
  };
}

function parseRunbooks(source: string): string[] {
  const parsed = JSON.parse(source) as unknown;
  if (!Array.isArray(parsed) || !parsed.every((value) => typeof value === "string" && /^TSG-[A-Z]{3,5}-\d{2}$/.test(value))) {
    throw new Error("Retrieval benchmark returned invalid runbook identifiers");
  }
  return [...new Set(parsed)];
}

function roleArgument(): Array<typeof allowedRoles[number]> {
  const source = argument("--roles") ?? "dev";
  const values = [...new Set(source.split(",").map((value) => value.trim()).filter(Boolean))];
  if (values.length === 0 || !values.every((value) => (allowedRoles as readonly string[]).includes(value))) throw new Error(`Invalid roles: ${source}`);
  return values as Array<typeof allowedRoles[number]>;
}

function armArgument(): DerivedArm[] {
  const source = argument("--arms") ?? "B0-majority-no-action-v1,B2-lexical-v1,B2-vector-v1,B2-hybrid-v1,B3-oracle-packet-v1";
  const values = [...new Set(source.split(",").map((value) => value.trim()).filter(Boolean))];
  if (values.length === 0 || !values.every((value) => (allowedArms as readonly string[]).includes(value))) throw new Error(`Invalid derived arms: ${source}`);
  return values as DerivedArm[];
}

function listArgument(name: string, fallback: string): string[] {
  const source = argument(name) ?? fallback;
  return [...new Set(source.split(",").map((value) => value.trim()).filter(Boolean))];
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}

function safeName(value: string): string { return value.replace(/[^a-z0-9_.-]+/gi, "-").toLowerCase(); }
function safeError(error: unknown): string { return (error as { message?: string }).message ?? String(error); }
