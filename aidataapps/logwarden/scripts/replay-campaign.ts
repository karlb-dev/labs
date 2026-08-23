import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import sql from "mssql";
import { runAgentLoop, type AgentLoopOptions, type AgentLoopResult, type PrimaryAgentArm } from "../src/agent-loop.js";
import { frozenArmIdentities, loadAgentArmRegistry, loadDecodeRegistry, loadStandardCampaign } from "../src/campaign.js";
import { summarizeChatMetrics, validateChatMetricDelta } from "../src/chat-metrics.js";
import { loadConfig } from "../src/config.js";
import { loadTier1ControlPolicy, maskErrorNumbersAndSignatures, parseInferenceControlId, type InferenceControlId } from "../src/controls.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { resolveEmbeddingProfile, resolveModelProfile } from "../src/models.js";
import { connect } from "../src/repository.js";
import {
  buildReplayCells,
  inferenceDecode,
  parseReplayArms,
  parseReplayRoles,
  replayJobKey,
  replayPrediction,
  type ReplayCell,
  type ReplayEpisode,
  type ReplayRole,
} from "../src/replay.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal, type TelemetryIngestionResult } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal } from "../src/telemetry.js";

interface RunManifest { runId: string }
interface CampaignRow { campaignId: number; status: string; campaignHash: string }
interface PacketRow {
  episode_id: string;
  split_role: string;
  packet_json: string;
  packet_sha256: string;
  expected_runbooks_json: string;
}
interface JobIdentity {
  jobId: number;
  workItemId: number;
  cell: ReplayCell;
}
interface ClaimedWork {
  work_item_id: string;
  job_id: string;
  episode_id: string;
  lease_token: string;
}
interface AttemptIdentity {
  jobAttemptId: number;
  attemptNumber: number;
}
interface SpanLink {
  traceId: string;
  spanId: string;
  turnId?: number;
  modelRequestId?: number;
  toolInvocationId?: number;
}
interface FrozenControlRow { episodeId: string; splitRole: ReplayRole; family: string; regime: string }
interface ControlAssignmentEvidence {
  controlId: InferenceControlId;
  episodeId: string;
  assignmentOrdinal: number;
  sourcePacketSha256: string;
  transformedPacketSha256: string;
  replacementCount: number;
  manifestSha256: string;
}

const startedAtUtc = new Date().toISOString();
const invocationId = randomUUID();
const profileKey = argument("--profile") ?? process.env.MODEL_PROFILE ?? "qwen-smoke";
const profile = resolveModelProfile(profileKey);
const campaignConfig = loadStandardCampaign();
const controlPolicy = loadTier1ControlPolicy();
const controlPolicySha256 = hashJson(controlPolicy);
const controlId = controlArgument();
const armRegistry = loadAgentArmRegistry();
const decodeRegistry = loadDecodeRegistry();
const decodeConfigId = armRegistry.decodeConfigId;
const decodeConfig = requiredDecodeConfig();
const roles = parseReplayRoles(argument("--roles") ?? (profileKey === "qwen-smoke" ? "dev" : controlId === null ? campaignConfig.qualityRoles.join(",") : "test_id,test_unknown"));
const arms = parseReplayArms(argument("--arms") ?? (controlId === null ? "A-direct,A-rag,A-tools" : "A-tools"));
const defaultWorkers = controlId === "batching-sequential-v1" ? controlPolicy.controls[controlId].workers : campaignConfig.workerCount;
const workerCount = numberArgument("--workers", defaultWorkers, 1, 64);
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const runKind = profileKey === "qwen-smoke" ? "qwen_smoke_replay" : controlId === null ? "agent_replay" : `control_${controlId.replace(/-v\d+$/, "").replaceAll("-", "_")}`;
const controlSuffix = controlId === null ? "" : `-${safeName(controlId)}`;
const receiptPath = `${runDirectory}/metrics/replay-${safeName(profileKey)}-${safeName(roles.join("-"))}${controlSuffix}.json`;
const rawRoot = `${runDirectory}/raw/replay/${safeName(profileKey)}/${controlId ?? "primary"}/${invocationId}`;
const embeddingProfile = resolveEmbeddingProfile(campaignConfig.embeddingProfile);
const control = await connect(config.databases.lab, config.databases.controlName, 600_000);
const agent = await connect(config.databases.agent, config.databases.controlName, 600_000);
let controlAssignments: ControlAssignmentEvidence[] = [];

try {
  const campaign = await loadCampaign();
  await assertGovernance(campaign);
  await requirePortGate();
  await requireServedModel();
  await registerDecode(campaign.status);
  await registerArms(campaign.status);
  const episodes = await loadEpisodes();
  await assertControlSources(episodes);
  const cells = buildReplayCells(episodes, arms);
  if (cells.length === 0) throw new Error("Replay selection produced no cells");
  const jobs = await ensureJobs(campaign, cells);
  await recoverInterruptedSelectedJobs(jobs);
  for (const job of jobs) await materializePrediction(job, null);
  await assertQueueIsolation(jobs);

  if (profileKey !== "qwen-smoke") await markCampaignRunning(campaign);
  const modelRequestsBefore = await selectedModelRequestCount(jobs);
  const metricsBefore = await retainChatMetrics("before");
  const settledWorkers = await Promise.allSettled(Array.from({ length: workerCount }, (_, index) => runWorker(index, campaign, jobs, episodes)));
  const rejectedWorkers = settledWorkers.filter((result): result is PromiseRejectedResult => result.status === "rejected");
  if (rejectedWorkers.length > 0) throw new AggregateError(rejectedWorkers.map((result) => result.reason), `${rejectedWorkers.length} replay workers failed`);
  const workerResults = settledWorkers.map((result) => (result as PromiseFulfilledResult<Record<string, unknown>>).value);
  await linkTelemetryByEvidence(jobs);
  for (const job of jobs) await materializePrediction(job, null);
  const verification = await verifyReplay(jobs);
  const metricsAfter = await retainChatMetrics("after");
  const modelRequestCount = Number(verification.model_request_count);
  const newModelRequestCount = modelRequestCount - modelRequestsBefore;
  if (newModelRequestCount < 0) throw new Error("Replay model-request count regressed");
  const metricDelta = newModelRequestCount === 0
    ? zeroDelta(metricsBefore.summary, metricsAfter.summary)
    : validateChatMetricDelta(metricsBefore.summary, metricsAfter.summary, newModelRequestCount);
  const exports = await exportRawResponses(jobs);
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    invocationId,
    profileKey,
    model: { modelId: profile.modelId, revision: profile.revision, image: profile.vllmImage, profileSha256: hashJson(profile) },
    roles,
    arms,
    controlId,
    controlPolicySha256: controlId === null ? null : controlPolicySha256,
    workerCount,
    runKind,
    decodeConfigId,
    decodeConfigSha256: hashJson(decodeConfig),
    agentBudget: campaignConfig.agentBudget,
    campaignId: campaign.campaignId,
    campaignStatusAtStart: campaign.status,
    campaignHash: campaign.campaignHash,
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    selectedEpisodes: episodes.length,
    selectedCells: cells.length,
    selectedCellSetSha256: hashJson(cells),
    controlAssignments: controlId === null ? null : {
      count: controlAssignments.length,
      orderedSetSha256: hashJson(controlAssignments),
      rows: controlAssignments,
    },
    workers: workerResults,
    verification,
    modelRequestsBefore,
    newModelRequestCount,
    modelMetrics: { before: metricsBefore, after: metricsAfter, delta: metricDelta },
    rawExports: exports,
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  await persistEvidence("PASS", receipt);
  await appendExperimentLog(`${profileKey} ${controlId === null ? "primary replay" : `control ${controlId}`} retained ${cells.length} cells across ${roles.join(",")} and ${arms.join(",")} with ${String(verification.decision_count)} decisions, ${String(verification.failure_count)} failures, and ${modelRequestCount} model requests; receipt ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({
    runId: run.runId,
    profileKey,
    roles,
    arms,
    controlId,
    selectedCells: cells.length,
    decisions: verification.decision_count,
    failures: verification.failure_count,
    modelRequests: modelRequestCount,
    metricDelta,
    receiptPath,
    receiptSha256: receipt.receiptSha256,
    disposition: "PASS",
  }, null, 2));
} catch (error) {
  const errorDetail = safeError(error);
  const failureBody = {
    schemaVersion: 1,
    runId: run.runId,
    invocationId,
    profileKey,
    roles,
    arms,
    controlId,
    workerCount,
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    errorDetail,
    disposition: "FAIL",
  };
  const failure = { ...failureBody, receiptSha256: hashJson(failureBody) };
  await atomicWrite(`${rawRoot}/failure-receipt.json`, `${JSON.stringify(failure, null, 2)}\n`, 0o600).catch(() => undefined);
  await persistEvidence("FAIL", failure).catch(() => undefined);
  console.error(JSON.stringify(failure, null, 2));
  process.exitCode = 1;
} finally {
  await Promise.all([control.close(), agent.close()]);
}

async function loadCampaign(): Promise<CampaignRow> {
  const result = await control.request().input("run", sql.VarChar(120), run.runId).query<{
    campaign_id: string; status: string; campaign_hash: string;
  }>(`
    SELECT campaign.campaign_id,campaign.status,campaign.campaign_hash
    FROM control.campaigns campaign INNER JOIN control.runs run ON run.campaign_id=campaign.campaign_id
    WHERE run.run_id=@run;
  `);
  const row = result.recordset[0];
  if (row === undefined) throw new Error("Current run has no campaign");
  return { campaignId: Number(row.campaign_id), status: row.status, campaignHash: row.campaign_hash };
}

async function assertGovernance(campaign: CampaignRow): Promise<void> {
  if (profileKey === "qwen-smoke") {
    if (controlId !== null) throw new Error("qwen-smoke cannot run frozen target controls");
    if (roles.some((role) => role !== "dev")) throw new Error("qwen-smoke is restricted to the development role");
    if (campaign.status !== "building") throw new Error(`qwen-smoke is a pre-freeze development gate and cannot run while campaign status is ${campaign.status}`);
    return;
  }
  if (!campaignConfig.targetProfiles.includes(profileKey)) throw new Error(`${profileKey} is not a governed target profile`);
  if (!(["frozen", "running"] as string[]).includes(campaign.status)) throw new Error(`Target replay requires a frozen campaign, not ${campaign.status}`);
  const freeze = await validatedReceipt(`${runDirectory}/manifests/freeze.json`, "FROZEN");
  const targets = freeze.targetProfiles;
  if (!Array.isArray(targets) || !targets.includes(profileKey)) throw new Error(`Campaign freeze does not authorize ${profileKey}`);
  if (controlId === null) {
    const configured = [...campaignConfig.qualityRoles].sort();
    if (canonicalJson([...roles].sort()) !== canonicalJson(configured)) throw new Error("Primary target replay must cover every frozen quality role in one cell set");
  } else {
    if (canonicalJson(arms) !== canonicalJson(["A-tools"])) throw new Error("Tier 1 inference controls are restricted to A-tools");
    if (canonicalJson([...roles].sort()) !== canonicalJson(["test_id", "test_unknown"])) throw new Error("Tier 1 inference controls require the frozen test_id/test_unknown subset");
    if (controlId === "batching-sequential-v1" && workerCount !== 1) throw new Error("Batching-invariance sequential control requires exactly one worker");
    if (controlId !== "batching-sequential-v1" && workerCount !== campaignConfig.workerCount) throw new Error("Masking and shuffled controls use the frozen primary worker count");
  }
}

async function markCampaignRunning(campaign: CampaignRow): Promise<void> {
  await control.request().input("campaign", sql.BigInt, campaign.campaignId)
    .input("run", sql.VarChar(120), run.runId).query(`
      UPDATE control.campaigns SET status='running' WHERE campaign_id=@campaign AND status='frozen';
      UPDATE control.runs SET status='running' WHERE run_id=@run AND status='initialized';
    `);
}

async function requirePortGate(): Promise<void> {
  const gate = await validatedReceipt(`${runDirectory}/metrics/chat-port-gate-${profileKey}.json`, "PASS");
  if (gate.profileKey !== profileKey) throw new Error(`Port receipt belongs to ${String(gate.profileKey)}`);
  const model = gate.model as Record<string, unknown> | undefined;
  if (model?.modelId !== profile.modelId || model.revision !== profile.revision || model.image !== profile.vllmImage || model.profileHash !== hashJson(profile)) {
    throw new Error("Port receipt model identity drifted from the registry");
  }
}

async function requireServedModel(): Promise<void> {
  const response = await fetch(`${config.inference.chatBaseUrl}/models`, { signal: AbortSignal.timeout(30_000) });
  const body = await response.text();
  if (!response.ok) throw new Error(`Chat model listing returned HTTP ${response.status}`);
  const parsed = JSON.parse(body) as { data?: Array<{ id?: string }> };
  if (!parsed.data?.some((entry) => entry.id === profile.modelId)) throw new Error(`Chat service does not expose ${profile.modelId}`);
}

async function registerDecode(campaignStatus: string): Promise<void> {
  const requested = canonicalJson(decodeConfig);
  const hash = hashJson(decodeConfig);
  const existing = await control.request().input("id", sql.VarChar(40), decodeConfigId)
    .query<{ decode_hash: string; requested_json: string }>("SELECT decode_hash,requested_json FROM control.decode_configs WHERE decode_config_id=@id;");
  const row = existing.recordset[0];
  if (row !== undefined) {
    if (row.decode_hash !== hash || canonicalJson(JSON.parse(row.requested_json)) !== requested) throw new Error(`Decode registry drift for ${decodeConfigId}`);
    return;
  }
  if (campaignStatus !== "building") throw new Error(`Decode ${decodeConfigId} was not registered before freeze`);
  await control.request().input("id", sql.VarChar(40), decodeConfigId).input("hash", sql.Char(64), hash)
    .input("json", sql.NVarChar(sql.MAX), requested)
    .query("INSERT control.decode_configs(decode_config_id,decode_hash,requested_json) VALUES(@id,@hash,@json);");
}

async function registerArms(campaignStatus: string): Promise<void> {
  const selected = frozenArmIdentities(armRegistry).filter((identity) => arms.includes(identity.armId as PrimaryAgentArm));
  if (selected.length !== arms.length) throw new Error("Frozen arm registry does not cover the selected replay arms");
  for (const identity of selected) {
    const armHash = hashJson(identity);
    const existing = await control.request().input("id", sql.VarChar(80), identity.armId)
      .query<{ arm_hash: string }>("SELECT arm_hash FROM control.agent_arms WHERE agent_arm_id=@id;");
    if (existing.recordset[0] !== undefined) {
      if (existing.recordset[0].arm_hash !== armHash) throw new Error(`Agent arm drift for ${identity.armId}`);
      continue;
    }
    if (campaignStatus !== "building") throw new Error(`Agent arm ${identity.armId} was not registered before freeze`);
    await control.request().input("id", sql.VarChar(80), identity.armId).input("hash", sql.Char(64), armHash)
      .input("prompt", sql.Char(64), identity.promptSha256).input("tools", sql.Char(64), identity.toolRegistrySha256)
      .input("policy", sql.Char(64), identity.policySha256).input("contract", sql.Char(64), identity.contractSha256)
      .input("packet", sql.VarChar(40), identity.packetVersion).input("retrieval", sql.VarChar(40), identity.retrievalMode)
      .input("correlation", sql.VarChar(40), identity.correlationMode).input("json", sql.NVarChar(sql.MAX), canonicalJson(identity))
      .query(`
        INSERT control.agent_arms(agent_arm_id,arm_hash,prompt_sha256,tool_registry_sha256,policy_sha256,
          contract_sha256,packet_version,retrieval_mode,correlation_mode,config_json)
        VALUES(@id,@hash,@prompt,@tools,@policy,@contract,@packet,@retrieval,@correlation,@json);
      `);
  }
}

async function loadEpisodes(): Promise<ReplayEpisode[]> {
  const result = await control.request().input("run", sql.VarChar(120), run.runId).query<PacketRow>(`
    SELECT packet.episode_id,packet.split_role,packet.packet_json,packet.packet_sha256,truth.expected_runbooks_json
    FROM ingest.incident_packets packet INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=packet.episode_id
    INNER JOIN workload.injection_executions execution ON execution.episode_id=packet.episode_id AND execution.run_id=@run
    WHERE packet.is_valid=1 ORDER BY packet.split_role,packet.episode_id;
  `);
  const available = new Map(result.recordset.map((row) => [row.episode_id, row]));
  const selectedRows = controlId === null
    ? result.recordset.filter((row) => roles.includes(row.split_role as ReplayRole))
    : (await frozenControlRows()).map((assignment) => {
        const row = available.get(assignment.episodeId);
        if (row === undefined || row.split_role !== assignment.splitRole) throw new Error(`Frozen control packet is missing or role-drifted: ${assignment.episodeId}`);
        return row;
      });
  const selected: ReplayEpisode[] = [];
  controlAssignments = [];
  for (const [ordinal, row] of selectedRows.entries()) {
    const packet = JSON.parse(row.packet_json) as unknown;
    if (hashJson(packet) !== row.packet_sha256) throw new Error(`Packet hash drift for ${row.episode_id}`);
    const expectedRunbooks = JSON.parse(row.expected_runbooks_json) as unknown;
    if (!Array.isArray(expectedRunbooks) || !expectedRunbooks.every((value) => typeof value === "string")) throw new Error(`Invalid retrieval eligibility for ${row.episode_id}`);
    const transformed = controlId === "error-number-mask-v1" ? maskErrorNumbersAndSignatures(packet, controlPolicy.controls[controlId])
      : { value: packet, replacementCount: 0, originalSha256: row.packet_sha256, transformedSha256: row.packet_sha256 };
    if (transformed.originalSha256 !== row.packet_sha256) throw new Error(`Control source hash drift for ${row.episode_id}`);
    if (controlId !== null) {
      const definition = controlPolicy.controls[controlId];
      const manifest = {
        schemaVersion: 1, controlId, controlPolicySha256, controlDefinitionSha256: hashJson(definition),
        episodeId: row.episode_id, assignmentOrdinal: ordinal, sourcePacketSha256: row.packet_sha256,
        transformedPacketSha256: transformed.transformedSha256, replacementCount: transformed.replacementCount,
      };
      await persistControlAssignment(manifest);
      controlAssignments.push({
        controlId, episodeId: row.episode_id, assignmentOrdinal: ordinal, sourcePacketSha256: row.packet_sha256,
        transformedPacketSha256: transformed.transformedSha256, replacementCount: transformed.replacementCount,
        manifestSha256: hashJson(manifest),
      });
    }
    selected.push({
      episodeId: row.episode_id,
      splitRole: row.split_role as ReplayRole,
      packetSha256: transformed.transformedSha256,
      packet: transformed.value,
      expectedRunbooks,
    });
  }
  const expected = roles.reduce((sum, role) => sum + Number(campaignRoleCounts[role] ?? 0), 0);
  if (profileKey !== "qwen-smoke" && controlId === null && selected.length !== expected) throw new Error(`Target replay selected ${selected.length}/${expected} frozen episodes`);
  if (profileKey === "qwen-smoke" && selected.length !== 60) throw new Error(`qwen-smoke requires all 60 dev episodes, found ${selected.length}`);
  const controlExpected = controlId === "batching-sequential-v1" ? 48 : 96;
  if (controlId !== null && selected.length !== controlExpected) throw new Error(`Control ${controlId} selected ${selected.length}/${controlExpected} frozen episodes`);
  return selected;
}

async function frozenControlRows(): Promise<FrozenControlRow[]> {
  if (controlId === null) return [];
  const [freeze, subset] = await Promise.all([
    validatedReceipt(`${runDirectory}/manifests/freeze.json`, "FROZEN"),
    readFile(`${runDirectory}/manifests/control-subset.json`, "utf8").then((value) => JSON.parse(value) as Record<string, unknown>),
  ]);
  const rows = subset.rows;
  if (!Array.isArray(rows)) throw new Error("Frozen control subset has no rows");
  const parsed = rows.map((value): FrozenControlRow => {
    const row = value as Record<string, unknown>;
    if (typeof row.episodeId !== "string" || (row.splitRole !== "test_id" && row.splitRole !== "test_unknown")
        || typeof row.family !== "string" || typeof row.regime !== "string") throw new Error("Frozen control subset row violates its contract");
    return { episodeId: row.episodeId, splitRole: row.splitRole, family: row.family, regime: row.regime };
  });
  if (parsed.length !== campaignConfig.controls.subsetEpisodes || hashJson(parsed) !== subset.subsetSha256
      || subset.subsetSha256 !== freeze.controlSubsetSha256) throw new Error("Frozen control subset hash or cardinality drift");
  return controlId === "batching-sequential-v1" ? parsed.slice(0, 48) : parsed;
}

async function persistControlAssignment(manifest: {
  controlId: InferenceControlId; episodeId: string; assignmentOrdinal: number; sourcePacketSha256: string;
  transformedPacketSha256: string; replacementCount: number;
} & Record<string, unknown>): Promise<void> {
  const manifestJson = canonicalJson(manifest);
  const manifestSha256 = hashJson(manifest);
  const prior = await control.request().input("run", sql.VarChar(120), run.runId).input("control", sql.VarChar(80), manifest.controlId)
    .input("episode", sql.VarChar(120), manifest.episodeId).query<{ transform_manifest_sha256: string }>(`
      SELECT transform_manifest_sha256 FROM eval.control_assignments WHERE run_id=@run AND control_id=@control AND episode_id=@episode;
    `);
  if (prior.recordset[0] !== undefined) {
    if (prior.recordset[0].transform_manifest_sha256 !== manifestSha256) throw new Error(`Control assignment drift for ${manifest.controlId}/${manifest.episodeId}`);
    return;
  }
  await control.request().input("run", sql.VarChar(120), run.runId).input("control", sql.VarChar(80), manifest.controlId)
    .input("episode", sql.VarChar(120), manifest.episodeId).input("ordinal", sql.Int, manifest.assignmentOrdinal)
    .input("source", sql.Char(64), manifest.sourcePacketSha256).input("transformed", sql.Char(64), manifest.transformedPacketSha256)
    .input("json", sql.NVarChar(sql.MAX), manifestJson).input("hash", sql.Char(64), manifestSha256).query(`
      INSERT eval.control_assignments(run_id,control_id,episode_id,assignment_ordinal,source_packet_sha256,
        transformed_packet_sha256,transform_manifest_json,transform_manifest_sha256)
      VALUES(@run,@control,@episode,@ordinal,@source,@transformed,@json,@hash);
    `);
}

async function assertControlSources(episodes: ReplayEpisode[]): Promise<void> {
  if (controlId === null) return;
  const ids = JSON.stringify(episodes.map((episode) => episode.episodeId));
  const result = await control.request().input("ids", sql.NVarChar(sql.MAX), ids).input("profile", sql.VarChar(80), profileKey)
    .input("run", sql.VarChar(120), run.runId).query<{ primary_count: number; shuffled_count: number }>(`
      WITH selected AS (SELECT CONVERT(varchar(120),value) episode_id FROM OPENJSON(@ids))
      SELECT
        (SELECT COUNT(*) FROM selected INNER JOIN eval.predictions prediction ON prediction.episode_id=selected.episode_id
          WHERE prediction.model_profile_id=@profile AND prediction.agent_arm_id='A-tools' AND prediction.control_id IS NULL) primary_count,
        (SELECT COUNT(*) FROM selected INNER JOIN eval.retrieval_benchmark_results benchmark ON benchmark.episode_id=selected.episode_id
          WHERE benchmark.run_id=@run AND benchmark.retrieval_mode='shuffled_runbook' AND benchmark.evaluator_only=1) shuffled_count;
    `);
  const row = result.recordset[0]!;
  if (Number(row.primary_count) !== episodes.length) throw new Error(`Control ${controlId} requires ${episodes.length} completed primary A-tools source predictions`);
  if (controlId === "shuffled-runbooks-v1" && Number(row.shuffled_count) !== episodes.length) throw new Error("Shuffled control source rows are incomplete");
}

const campaignRoleCounts: Partial<Record<ReplayRole, number>> = {
  dev: 60, calibration: 60, test_id: 300, test_variant_holdout: 120, test_unknown: 60,
};

async function ensureJobs(campaign: CampaignRow, cells: ReplayCell[]): Promise<JobIdentity[]> {
  const jobs: JobIdentity[] = [];
  for (const [ordinal, cell] of cells.entries()) {
    const key = replayJobKey({ campaignId: campaign.campaignId, runKind, episodeId: cell.episodeId, modelProfileId: profileKey, agentArmId: cell.arm, decodeConfigId });
    const transaction = new sql.Transaction(control);
    await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
    try {
      const jobResult = await new sql.Request(transaction).input("key", sql.Char(64), key)
        .input("campaign", sql.BigInt, campaign.campaignId).input("kind", sql.VarChar(40), runKind)
        .input("episode", sql.VarChar(120), cell.episodeId).input("model", sql.VarChar(80), profileKey)
        .input("arm", sql.VarChar(80), cell.arm).input("decode", sql.VarChar(40), decodeConfigId)
        .input("priority", sql.Int, 1_000_000 - ordinal).query<{
          job_id: string; run_kind: string; episode_id: string; model_profile_id: string; agent_arm_id: string; decode_config_id: string;
        }>(`
          IF NOT EXISTS(SELECT 1 FROM control.jobs WHERE job_key=@key)
            INSERT control.jobs(job_key,campaign_id,run_kind,episode_id,model_profile_id,agent_arm_id,
              decode_config_id,sample_index,priority,status)
            VALUES(@key,@campaign,@kind,@episode,@model,@arm,@decode,0,@priority,'pending');
          SELECT job_id,run_kind,episode_id,model_profile_id,agent_arm_id,decode_config_id
          FROM control.jobs WHERE job_key=@key;
        `);
      const job = jobResult.recordset[0]!;
      if (job.run_kind !== runKind || job.episode_id !== cell.episodeId || job.model_profile_id !== profileKey || job.agent_arm_id !== cell.arm || job.decode_config_id !== decodeConfigId) {
        throw new Error(`Replay job identity drift for ${cell.arm}/${cell.episodeId}`);
      }
      const jobId = Number(job.job_id);
      const workResult = await new sql.Request(transaction).input("run", sql.VarChar(120), run.runId)
        .input("job", sql.BigInt, jobId).input("episode", sql.VarChar(120), cell.episodeId)
        .input("priority", sql.Int, 1_000_000 - ordinal).query<{ work_item_id: string }>(`
          IF NOT EXISTS(SELECT 1 FROM ops.work_items WHERE job_id=@job)
            INSERT ops.work_items(run_id,job_id,episode_id,status,priority)
            VALUES(@run,@job,@episode,'pending',@priority);
          SELECT work_item_id FROM ops.work_items WHERE job_id=@job;
        `);
      jobs.push({ jobId, workItemId: Number(workResult.recordset[0]!.work_item_id), cell });
      await transaction.commit();
    } catch (error) {
      await transaction.rollback().catch(() => undefined);
      throw error;
    }
  }
  return jobs;
}

async function recoverInterruptedSelectedJobs(jobs: JobIdentity[]): Promise<void> {
  const ids = JSON.stringify(jobs.map((job) => job.jobId));
  const active = await control.request().input("ids", sql.NVarChar(sql.MAX), ids).query<{ active_count: number }>(`
    WITH selected AS (SELECT CONVERT(bigint,value) job_id FROM OPENJSON(@ids))
    SELECT COUNT(*) active_count FROM ops.work_items item INNER JOIN selected ON selected.job_id=item.job_id
    WHERE item.lease_token IS NOT NULL AND item.leased_until_utc>=SYSUTCDATETIME();
  `);
  if (Number(active.recordset[0]?.active_count ?? 0) > 0) throw new Error("Selected replay cells still have active leases; another worker may be running");
  await control.request().input("ids", sql.NVarChar(sql.MAX), ids).query(`
    WITH selected AS (SELECT CONVERT(bigint,value) job_id FROM OPENJSON(@ids))
    UPDATE agent_run SET status='failed',terminal_reason='Recovered an interrupted replay without scientific retry',
      finished_at_utc=COALESCE(finished_at_utc,SYSUTCDATETIME())
    FROM agent.agent_runs agent_run INNER JOIN selected ON selected.job_id=agent_run.job_id
    WHERE agent_run.status='running';
    WITH selected AS (SELECT CONVERT(bigint,value) job_id FROM OPENJSON(@ids))
    UPDATE attempt SET status='failed',finished_at_utc=COALESCE(finished_at_utc,SYSUTCDATETIME()),
      error_class='worker_interrupted',error_detail='Recovered before replay resume'
    FROM control.job_attempts attempt INNER JOIN selected ON selected.job_id=attempt.job_id
    WHERE attempt.status='running';
    WITH selected AS (SELECT CONVERT(bigint,value) job_id FROM OPENJSON(@ids))
    UPDATE job SET status='failed',completed_at_utc=COALESCE(completed_at_utc,SYSUTCDATETIME()),
      error_class='worker_interrupted',error_detail='Recovered before replay resume'
    FROM control.jobs job INNER JOIN selected ON selected.job_id=job.job_id WHERE job.status='running';
    DECLARE @stopped TABLE(work_item_id bigint,from_state varchar(40));
    WITH selected AS (SELECT CONVERT(bigint,value) job_id FROM OPENJSON(@ids))
    UPDATE item SET status='stopped',lease_owner=NULL,lease_token=NULL,leased_until_utc=NULL,
      completed_at_utc=COALESCE(completed_at_utc,SYSUTCDATETIME())
    OUTPUT INSERTED.work_item_id,DELETED.status INTO @stopped
    FROM ops.work_items item INNER JOIN selected ON selected.job_id=item.job_id
    INNER JOIN control.jobs job ON job.job_id=item.job_id
    WHERE job.status='failed' AND item.status NOT IN ('complete','contract_rejected','policy_rejected','model_timeout','tool_timeout','stopped');
    INSERT ops.transitions(entity_kind,entity_id,from_state,to_state,actor,reason)
    SELECT 'work_item',work_item_id,from_state,'stopped','replay-resume','interrupted cell retained as a scientific failure' FROM @stopped;
  `);
}

async function assertQueueIsolation(jobs: JobIdentity[]): Promise<void> {
  const ids = JSON.stringify(jobs.map((job) => job.jobId));
  const result = await control.request().input("ids", sql.NVarChar(sql.MAX), ids).query<{ unexpected: number }>(`
    WITH selected AS (SELECT CONVERT(bigint,value) job_id FROM OPENJSON(@ids))
    SELECT COUNT(*) unexpected FROM ops.work_items item
    WHERE NOT EXISTS(SELECT 1 FROM selected WHERE selected.job_id=item.job_id)
      AND (item.status IN ('pending','retryable_failure','lease_expired') OR
        (item.status IN ('leased','packet_loaded','model_requested','tool_requested','tool_completed','decision_received','validated','persisted','proposed_action_recorded')
         AND item.leased_until_utc<SYSUTCDATETIME()));
  `);
  if (Number(result.recordset[0]?.unexpected ?? 0) !== 0) throw new Error("Replay queue contains claimable work outside the selected cell set");
}

async function runWorker(index: number, campaign: CampaignRow, jobs: JobIdentity[], episodes: ReplayEpisode[]): Promise<Record<string, unknown>> {
  const workerId = `replay-${safeName(profileKey)}-${invocationId.slice(0, 8)}-${String(index).padStart(2, "0")}`.slice(0, 120);
  const created = await createComponentTelemetryJournal(runDirectory, run.runId, `replay-${profileKey}-${controlId ?? "primary"}-worker-${index}`);
  const links: SpanLink[] = [];
  const episodeMap = new Map(episodes.map((episode) => [episode.episodeId, episode]));
  const jobMap = new Map(jobs.map((job) => [job.jobId, job]));
  const agentControl = agentControlOptions();
  let claimed = 0;
  let decisions = 0;
  let failures = 0;
  await created.journal.record("point", "replay.worker.started", {}, { invocationId, workerId, profileKey, roles, arms, controlId, runKind });
  while (true) {
    const leaseToken = randomUUID();
    const claim = await control.request().input("worker_id", sql.VarChar(120), workerId)
      .input("lease_token", sql.UniqueIdentifier, leaseToken).input("lease_seconds", sql.Int, 300)
      .execute<ClaimedWork>("ops.usp_claim_work_item");
    const work = claim.recordset[0];
    if (work === undefined) break;
    claimed += 1;
    const job = jobMap.get(Number(work.job_id));
    if (job === undefined) throw new Error(`Queue isolation failed: worker claimed job ${work.job_id}`);
    const episode = episodeMap.get(job.cell.episodeId);
    if (episode === undefined) throw new Error(`Missing packet for claimed episode ${job.cell.episodeId}`);
    let attempt: AttemptIdentity | null = null;
    let result: AgentLoopResult | null = null;
    try {
      attempt = await startAttempt(job, workerId, leaseToken);
      result = await runAgentLoop({
        identity: {
          runId: run.runId,
          campaignId: campaign.campaignId,
          jobId: job.jobId,
          jobAttemptId: attempt.jobAttemptId,
          workItemId: job.workItemId,
          leaseToken,
          episodeId: job.cell.episodeId,
          modelProfileId: profileKey,
          agentArmId: job.cell.arm,
          decodeConfigId,
          workerId,
        },
        controlPool: control,
        toolPool: agent,
        packet: episode.packet,
        arm: job.cell.arm,
        endpoint: `${config.inference.chatBaseUrl}/chat/completions`,
        modelProfile: profile,
        decode: inferenceDecode(decodeConfig),
        embeddingEndpoint: `${config.inference.qwenEmbeddingBaseUrl}/embeddings`,
        embeddingProfile,
        retrievalMode: job.cell.arm === "A-direct" ? "none" : "hybrid_rrf",
        runDirectory,
        journal: created.journal,
        policyVersion: armRegistry.policyVersion,
        budget: campaignConfig.agentBudget,
        timeoutMs: campaignConfig.agentBudget.maxWallTimeSeconds * 1_000,
        maxRetries: 0,
        ...(agentControl === undefined ? {} : { control: agentControl }),
      });
      links.push(...result.spanLinks);
      const prediction = await materializePrediction(job, result);
      if (prediction === "decision") decisions += 1;
      else failures += 1;
      await created.journal.record("point", "replay.cell.terminal", { jobId: job.jobId, episodeId: job.cell.episodeId, attemptId: attempt.jobAttemptId }, {
        invocationId, workerId, arm: job.cell.arm, controlId, status: result.status, terminalReason: result.terminalReason, prediction: prediction,
      }, { status: result.status === "complete" ? "success" : "failed" });
    } catch (error) {
      failures += 1;
      const detail = safeError(error);
      await terminalizeUnexpected(job, attempt, detail);
      await materializePrediction(job, result).catch(() => undefined);
      await created.journal.record("point", "replay.cell.exception", { jobId: job.jobId, episodeId: job.cell.episodeId, ...(attempt === null ? {} : { attemptId: attempt.jobAttemptId }) }, {
        invocationId, workerId, arm: job.cell.arm, controlId, errorDetail: detail,
      }, { status: "failed" });
    }
  }
  await created.journal.record("point", "replay.worker.finished", {}, { invocationId, workerId, controlId, claimed, decisions, failures }, { status: "success" });
  await created.journal.flush();
  const ingestion = await ingestTelemetryJournal(control, run.runId, created.path, { batchSize: 250 });
  await linkTelemetry(links);
  return { workerId, journalPath: created.path, processEpochId: created.processEpochId, claimed, decisions, failures, ingestion };
}

async function startAttempt(job: JobIdentity, workerId: string, leaseToken: string): Promise<AttemptIdentity> {
  const manifest = {
    schemaVersion: 1,
    invocationId,
    workerId,
    leaseTokenSha256: sha256(leaseToken),
    profileKey,
    modelId: profile.modelId,
    revision: profile.revision,
    image: profile.vllmImage,
    arm: job.cell.arm,
    role: job.cell.splitRole,
    episodeId: job.cell.episodeId,
    packetSha256: job.cell.packetSha256,
    controlId,
    controlPolicySha256: controlId === null ? null : controlPolicySha256,
    runKind,
    decodeConfigId,
    decodeConfigSha256: hashJson(decodeConfig),
    endpointIdentitySha256: sha256(new URL(config.inference.chatBaseUrl).origin),
    maxRetries: 0,
    agentBudget: campaignConfig.agentBudget,
  };
  const transaction = new sql.Transaction(control);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    const updated = await new sql.Request(transaction).input("job", sql.BigInt, job.jobId)
      .query<{ attempt_count: number }>(`
        UPDATE control.jobs SET status='running',started_at_utc=COALESCE(started_at_utc,SYSUTCDATETIME()),
          attempt_count=attempt_count+1,error_class=NULL,error_detail=NULL
        OUTPUT INSERTED.attempt_count WHERE job_id=@job AND status='pending';
      `);
    const attemptNumber = Number(updated.recordset[0]?.attempt_count);
    if (!Number.isSafeInteger(attemptNumber) || attemptNumber < 1) throw new Error(`Job ${job.jobId} was not pending when claimed`);
    const inserted = await new sql.Request(transaction).input("job", sql.BigInt, job.jobId)
      .input("number", sql.Int, attemptNumber).input("worker", sql.VarChar(120), workerId)
      .input("manifest", sql.NVarChar(sql.MAX), canonicalJson(manifest)).query<{ job_attempt_id: string }>(`
        INSERT control.job_attempts(job_id,attempt_number,worker_id,request_manifest_json,status)
        OUTPUT INSERTED.job_attempt_id VALUES(@job,@number,@worker,@manifest,'running');
      `);
    await transaction.commit();
    return { jobAttemptId: Number(inserted.recordset[0]!.job_attempt_id), attemptNumber };
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
}

async function terminalizeUnexpected(job: JobIdentity, attempt: AttemptIdentity | null, detail: string): Promise<void> {
  await control.request().input("job", sql.BigInt, job.jobId).input("work", sql.BigInt, job.workItemId)
    .input("attempt", sql.BigInt, attempt?.jobAttemptId ?? null).input("detail", sql.NVarChar(sql.MAX), detail).query(`
      UPDATE agent.agent_runs SET status='failed',terminal_reason=@detail,
        finished_at_utc=COALESCE(finished_at_utc,SYSUTCDATETIME()) WHERE job_id=@job AND status='running';
      UPDATE control.job_attempts SET status='failed',finished_at_utc=COALESCE(finished_at_utc,SYSUTCDATETIME()),
        error_class='worker_exception',error_detail=@detail WHERE job_attempt_id=@attempt AND status='running';
      UPDATE control.jobs SET status='failed',completed_at_utc=COALESCE(completed_at_utc,SYSUTCDATETIME()),
        error_class='worker_exception',error_detail=@detail WHERE job_id=@job AND status IN ('pending','running');
      DECLARE @prior varchar(40)=(SELECT status FROM ops.work_items WHERE work_item_id=@work);
      IF @prior NOT IN ('complete','contract_rejected','policy_rejected','model_timeout','tool_timeout','stopped')
      BEGIN
        UPDATE ops.work_items SET status='stopped',lease_owner=NULL,lease_token=NULL,leased_until_utc=NULL,
          completed_at_utc=COALESCE(completed_at_utc,SYSUTCDATETIME()) WHERE work_item_id=@work;
        INSERT ops.transitions(entity_kind,entity_id,from_state,to_state,actor,reason)
        VALUES('work_item',@work,@prior,'stopped','replay-worker','unexpected worker exception retained as failure');
      END;
    `);
}

async function materializePrediction(job: JobIdentity, immediate: AgentLoopResult | null): Promise<"decision" | "failure" | "existing" | "pending"> {
  const result = await control.request().input("job", sql.BigInt, job.jobId).query<{
    job_status: string; job_attempt_id: string | null; job_error_class: string | null; job_error_detail: string | null;
    agent_run_id: string | null; agent_status: AgentLoopResult["status"] | null; terminal_reason: string | null;
    trace_id: string | null; elapsed_ms: number | null; model_turn_count: number | null; tool_call_count: number | null;
    tool_result_chars: number | null; decision_id: string | null; incident_class: string | null; severity: string | null;
    action: string | null; confidence: number | null; abstained: boolean | null; decision_sha256: string | null;
    cache_hit_count: number; snapshot_miss_count: number;
  }>(`
    SELECT job.status job_status,attempt.job_attempt_id,job.error_class job_error_class,job.error_detail job_error_detail,
      agent_run.agent_run_id,agent_run.status agent_status,agent_run.terminal_reason,agent_run.trace_id,agent_run.elapsed_ms,
      agent_run.model_turn_count,agent_run.tool_call_count,agent_run.tool_result_chars,
      decision.decision_id,decision.incident_class,decision.severity,decision.action,decision.confidence,
      decision.abstained,decision.decision_sha256,
      (SELECT COUNT(*) FROM agent.tool_invocations tool WHERE tool.agent_run_id=agent_run.agent_run_id AND tool.cache_hit=1) cache_hit_count,
      (SELECT COUNT(*) FROM agent.tool_invocations tool WHERE tool.agent_run_id=agent_run.agent_run_id AND tool.snapshot_miss=1) snapshot_miss_count
    FROM control.jobs job
    OUTER APPLY(SELECT TOP(1) * FROM control.job_attempts value WHERE value.job_id=job.job_id ORDER BY value.attempt_number DESC) attempt
    OUTER APPLY(SELECT TOP(1) * FROM agent.agent_runs value WHERE value.job_id=job.job_id ORDER BY value.agent_run_id DESC) agent_run
    OUTER APPLY(SELECT TOP(1) * FROM agent.decisions value WHERE value.agent_run_id=agent_run.agent_run_id ORDER BY value.decision_id DESC) decision
    WHERE job.job_id=@job;
  `);
  const row = result.recordset[0];
  if (row === undefined) throw new Error(`Missing job ${job.jobId}`);
  const prior = await control.request().input("job", sql.BigInt, job.jobId)
    .query<{ prediction_id: string }>("SELECT prediction_id FROM eval.predictions WHERE job_id=@job;");
  if (prior.recordset[0] !== undefined) return "existing";
  if (!(["complete", "failed"] as string[]).includes(row.job_status)) return "pending";
  const decision = row.job_status !== "complete" || row.agent_status !== "complete" || row.decision_id === null ? null : {
    incidentClass: row.incident_class!,
    severity: row.severity!,
    action: row.action!,
    confidence: Number(row.confidence),
    abstained: Boolean(row.abstained),
    decisionSha256: row.decision_sha256!,
  };
  const reconstructed = immediate ?? (row.agent_run_id === null ? null : {
    status: row.agent_status ?? "failed",
    terminalReason: row.terminal_reason ?? row.job_error_detail ?? "terminal agent run",
    traceId: row.trace_id!,
    modelTurnCount: Number(row.model_turn_count ?? 0),
    toolCallCount: Number(row.tool_call_count ?? 0),
    toolResultChars: Number(row.tool_result_chars ?? 0),
    cacheHitCount: Number(row.cache_hit_count ?? 0),
    snapshotMissCount: Number(row.snapshot_miss_count ?? 0),
    elapsedMs: Number(row.elapsed_ms ?? 0),
  });
  const sourcePredictionId = await primarySourcePredictionId(job.cell.episodeId);
  const prediction = replayPrediction({
    episodeId: job.cell.episodeId,
    splitRole: job.cell.splitRole,
    packetSha256: job.cell.packetSha256,
    modelProfileId: profileKey,
    agentArmId: job.cell.arm,
    decodeConfigId,
    jobId: job.jobId,
    jobAttemptId: row.job_attempt_id === null ? null : Number(row.job_attempt_id),
    agentRunId: row.agent_run_id === null ? null : Number(row.agent_run_id),
    decisionId: row.decision_id === null ? null : Number(row.decision_id),
    result: reconstructed,
    decision,
    failureClass: row.job_error_class,
    failureDetail: row.job_error_detail ?? row.terminal_reason,
    controlId,
    sourcePredictionId,
  });
  await control.request().input("job", sql.BigInt, job.jobId).input("run", sql.BigInt, row.agent_run_id === null ? null : Number(row.agent_run_id))
    .input("decision", sql.BigInt, row.decision_id === null ? null : Number(row.decision_id)).input("episode", sql.VarChar(120), job.cell.episodeId)
    .input("model", sql.VarChar(80), profileKey).input("arm", sql.VarChar(80), job.cell.arm)
    .input("class", sql.VarChar(80), prediction.predictedClass).input("severity", sql.VarChar(24), prediction.predictedSeverity)
    .input("action", sql.VarChar(100), prediction.predictedAction).input("confidence", sql.Decimal(9, 6), prediction.confidence)
    .input("abstain", sql.Bit, prediction.abstained).input("outcome", sql.VarChar(32), prediction.outcome)
    .input("stage", sql.VarChar(80), prediction.failureStage).input("failure", sql.VarChar(80), row.job_error_class)
    .input("complete", sql.Bit, prediction.completeCaseEligible).input("control_id", sql.VarChar(80), controlId)
    .input("source_prediction", sql.BigInt, sourcePredictionId)
    .input("json", sql.NVarChar(sql.MAX), canonicalJson(prediction.envelope)).query(`
      IF NOT EXISTS(SELECT 1 FROM eval.predictions WHERE job_id=@job)
        INSERT eval.predictions(job_id,agent_run_id,decision_id,episode_id,model_profile_id,agent_arm_id,
          prediction_source,predicted_class,predicted_severity,predicted_action,confidence,abstained,outcome,
          failure_stage,failure_class,eligible,complete_case_eligible,control_id,source_prediction_id,prediction_json)
        VALUES(@job,@run,@decision,@episode,@model,@arm,'agent',@class,@severity,@action,@confidence,@abstain,
          @outcome,@stage,@failure,1,@complete,@control_id,@source_prediction,@json);
    `);
  return decision === null ? "failure" : "decision";
}

async function primarySourcePredictionId(episodeId: string): Promise<number | null> {
  if (controlId === null) return null;
  const result = await control.request().input("run", sql.VarChar(120), run.runId)
    .input("episode", sql.VarChar(120), episodeId).input("profile", sql.VarChar(80), profileKey)
    .query<{ prediction_id: string }>(`
      SELECT prediction.prediction_id
      FROM eval.predictions prediction
      INNER JOIN ops.work_items item ON item.job_id=prediction.job_id AND item.run_id=@run
      WHERE prediction.episode_id=@episode AND prediction.model_profile_id=@profile
        AND prediction.agent_arm_id='A-tools' AND prediction.prediction_source='agent'
        AND prediction.control_id IS NULL;
    `);
  if (result.recordset.length !== 1) throw new Error(`Expected exactly one primary A-tools prediction for ${profileKey}/${episodeId}, found ${result.recordset.length}`);
  return Number(result.recordset[0]!.prediction_id);
}

async function linkTelemetry(links: SpanLink[]): Promise<void> {
  for (const link of links) await control.request().input("trace", sql.Char(32), link.traceId)
    .input("span", sql.Char(16), link.spanId).input("turn", sql.BigInt, link.turnId ?? null)
    .input("request", sql.BigInt, link.modelRequestId ?? null).input("tool", sql.BigInt, link.toolInvocationId ?? null).query(`
      UPDATE telemetry.spans SET turn_id=COALESCE(@turn,turn_id),model_request_id=COALESCE(@request,model_request_id),
        tool_invocation_id=COALESCE(@tool,tool_invocation_id) WHERE trace_id=@trace AND span_id=@span;
    `);
}

async function linkTelemetryByEvidence(jobs: JobIdentity[]): Promise<void> {
  const ids = JSON.stringify(jobs.map((job) => job.jobId));
  await control.request().input("ids", sql.NVarChar(sql.MAX), ids).query(`
    WITH selected AS (SELECT CONVERT(bigint,value) job_id FROM OPENJSON(@ids))
    UPDATE span SET turn_id=COALESCE(span.turn_id,step.turn_id),
      model_request_id=COALESCE(span.model_request_id,request.model_request_id),
      tool_invocation_id=COALESCE(span.tool_invocation_id,tool.tool_invocation_id)
    FROM telemetry.spans span
    INNER JOIN agent.agent_runs agent_run ON agent_run.trace_id=span.trace_id
    INNER JOIN selected ON selected.job_id=agent_run.job_id
    LEFT JOIN agent.agent_steps step ON step.agent_run_id=agent_run.agent_run_id AND step.span_id=span.span_id
    LEFT JOIN agent.model_requests request ON request.client_request_id=JSON_VALUE(span.attributes_json,'$.clientRequestId')
    LEFT JOIN agent.tool_invocations tool ON tool.agent_run_id=agent_run.agent_run_id AND tool.agent_step_id=step.agent_step_id;
  `);
}

async function selectedModelRequestCount(jobs: JobIdentity[]): Promise<number> {
  const ids = JSON.stringify(jobs.map((job) => job.jobId));
  const result = await control.request().input("ids", sql.NVarChar(sql.MAX), ids).query<{ request_count: number }>(`
    WITH selected AS (SELECT CONVERT(bigint,value) job_id FROM OPENJSON(@ids))
    SELECT COUNT(*) request_count FROM agent.model_requests request
    INNER JOIN agent.turns turn ON turn.turn_id=request.turn_id
    INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=turn.agent_run_id
    INNER JOIN selected ON selected.job_id=agent_run.job_id;
  `);
  return Number(result.recordset[0]?.request_count ?? 0);
}

async function verifyReplay(jobs: JobIdentity[]): Promise<Record<string, unknown>> {
  const ids = JSON.stringify(jobs.map((job) => job.jobId));
  const result = await control.request().input("ids", sql.NVarChar(sql.MAX), ids).query<Record<string, unknown>>(`
    WITH selected AS (SELECT CONVERT(bigint,value) job_id FROM OPENJSON(@ids))
    SELECT
      (SELECT COUNT(*) FROM selected) selected_count,
      (SELECT COUNT(*) FROM control.jobs job INNER JOIN selected ON selected.job_id=job.job_id WHERE job.status IN ('complete','failed')) terminal_job_count,
      (SELECT COUNT(*) FROM eval.predictions prediction INNER JOIN selected ON selected.job_id=prediction.job_id) prediction_count,
      (SELECT COUNT(*) FROM eval.predictions prediction INNER JOIN selected ON selected.job_id=prediction.job_id WHERE prediction.outcome IN ('decision','abstention')) decision_count,
      (SELECT COUNT(*) FROM eval.predictions prediction INNER JOIN selected ON selected.job_id=prediction.job_id WHERE prediction.outcome='failure') failure_count,
      (SELECT COUNT(*) FROM agent.agent_runs agent_run INNER JOIN selected ON selected.job_id=agent_run.job_id) agent_run_count,
      (SELECT COUNT(*) FROM agent.turns turn INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=turn.agent_run_id INNER JOIN selected ON selected.job_id=agent_run.job_id) model_turn_count,
      (SELECT COUNT(*) FROM agent.model_requests request INNER JOIN agent.turns turn ON turn.turn_id=request.turn_id INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=turn.agent_run_id INNER JOIN selected ON selected.job_id=agent_run.job_id) model_request_count,
      (SELECT COUNT(*) FROM agent.tool_invocations tool INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=tool.agent_run_id INNER JOIN selected ON selected.job_id=agent_run.job_id) tool_call_count,
      (SELECT COUNT(*) FROM ops.work_items item INNER JOIN selected ON selected.job_id=item.job_id WHERE item.lease_token IS NOT NULL OR item.lease_owner IS NOT NULL OR item.leased_until_utc IS NOT NULL) retained_lease_count,
      (SELECT COUNT(*) FROM agent.model_requests request INNER JOIN agent.turns turn ON turn.turn_id=request.turn_id INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=turn.agent_run_id INNER JOIN selected ON selected.job_id=agent_run.job_id WHERE LOWER(CONVERT(varchar(64),HASHBYTES('SHA2_256',request.request_body),2))<>request.request_body_sha256) bad_request_hash_count,
      (SELECT COUNT(*) FROM agent.model_responses response INNER JOIN agent.model_requests request ON request.model_request_id=response.model_request_id INNER JOIN agent.turns turn ON turn.turn_id=request.turn_id INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=turn.agent_run_id INNER JOIN selected ON selected.job_id=agent_run.job_id WHERE LOWER(CONVERT(varchar(64),HASHBYTES('SHA2_256',response.response_body),2))<>response.response_body_sha256) bad_response_hash_count,
      (SELECT COUNT(*) FROM agent.model_requests request INNER JOIN agent.turns turn ON turn.turn_id=request.turn_id INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=turn.agent_run_id INNER JOIN selected ON selected.job_id=agent_run.job_id WHERE JSON_QUERY(CONVERT(nvarchar(max),request.request_body),'$.response_format') IS NOT NULL OR JSON_QUERY(CONVERT(nvarchar(max),request.request_body),'$.tools') IS NOT NULL) constrained_transport_count,
      (SELECT COUNT(*) FROM ops.action_proposals proposal INNER JOIN agent.decisions decision ON decision.decision_id=proposal.decision_id INNER JOIN agent.agent_runs agent_run ON agent_run.agent_run_id=decision.agent_run_id INNER JOIN selected ON selected.job_id=agent_run.job_id WHERE proposal.caller_opted_in=1 OR proposal.executed_at_utc IS NOT NULL OR proposal.execution_status<>'not_executed') unsafe_action_count,
      (SELECT COUNT(*) FROM telemetry.spans span INNER JOIN telemetry.traces trace ON trace.trace_id=span.trace_id INNER JOIN selected ON selected.job_id=trace.job_id WHERE span.finished_at_utc IS NULL) open_span_count,
      (SELECT COUNT(*) FROM telemetry.traces trace INNER JOIN selected ON selected.job_id=trace.job_id WHERE trace.finished_at_utc IS NULL) open_trace_count,
      (SELECT COUNT(*) FROM telemetry.spans span INNER JOIN telemetry.traces trace ON trace.trace_id=span.trace_id INNER JOIN selected ON selected.job_id=trace.job_id WHERE span.span_name='model.request' AND span.model_request_id IS NULL) unlinked_model_span_count,
      (SELECT COUNT(*) FROM telemetry.spans span INNER JOIN telemetry.traces trace ON trace.trace_id=span.trace_id INNER JOIN selected ON selected.job_id=trace.job_id WHERE span.span_name='tool.operation' AND span.tool_invocation_id IS NULL) unlinked_tool_span_count;
  `);
  const row = result.recordset[0]!;
  const exact = ["terminal_job_count", "prediction_count"].every((key) => Number(row[key]) === jobs.length);
  const zero = ["retained_lease_count", "bad_request_hash_count", "bad_response_hash_count", "constrained_transport_count", "unsafe_action_count", "open_span_count", "open_trace_count", "unlinked_model_span_count", "unlinked_tool_span_count"]
    .every((key) => Number(row[key]) === 0);
  if (!exact || !zero || Number(row.decision_count) + Number(row.failure_count) !== jobs.length) throw new Error(`Replay verification failed: ${JSON.stringify(row)}`);
  return { ...row, selectedCellSetSha256: hashJson(jobs.map((job) => job.cell)), disposition: "PASS" };
}

async function exportRawResponses(jobs: JobIdentity[]): Promise<Array<Record<string, unknown>>> {
  const ids = JSON.stringify(jobs.map((job) => job.jobId));
  const output: Array<Record<string, unknown>> = [];
  for (const arm of arms) {
    const result = await control.request().input("ids", sql.NVarChar(sql.MAX), ids).input("arm", sql.VarChar(80), arm).query<{
      job_id: string; episode_id: string; split_role: string; agent_run_id: string; turn_ordinal: number; retry_ordinal: number;
      client_request_id: string; request_body: Buffer; request_body_sha256: string; response_body: Buffer; response_body_sha256: string;
      http_status: number | null; finish_reason: string | null; prompt_tokens: string | null; completion_tokens: string | null;
      total_tokens: string | null; reasoning_sha256: string | null; reasoning_bytes: number | null; parse_status: string; repair_kind: string;
      error_class: string | null; client_elapsed_ms: number | null; headers_wait_ms: number | null; body_read_ms: number | null; parse_ms: number | null;
    }>(`
      WITH selected AS (SELECT CONVERT(bigint,value) job_id FROM OPENJSON(@ids))
      SELECT job.job_id,job.episode_id,truth.split_role,agent_run.agent_run_id,turn.turn_ordinal,request.retry_ordinal,
        request.client_request_id,request.request_body,request.request_body_sha256,response.response_body,response.response_body_sha256,
        response.http_status,response.finish_reason,response.prompt_tokens,response.completion_tokens,response.total_tokens,
        response.reasoning_sha256,response.reasoning_bytes,response.parse_status,response.repair_kind,response.error_class,
        request.client_elapsed_ms,request.headers_wait_ms,request.body_read_ms,response.parse_ms
      FROM selected INNER JOIN control.jobs job ON job.job_id=selected.job_id
      INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=job.episode_id
      INNER JOIN agent.agent_runs agent_run ON agent_run.job_id=job.job_id
      INNER JOIN agent.turns turn ON turn.agent_run_id=agent_run.agent_run_id
      INNER JOIN agent.model_requests request ON request.turn_id=turn.turn_id
      INNER JOIN agent.model_responses response ON response.model_request_id=request.model_request_id
      WHERE job.agent_arm_id=@arm ORDER BY truth.split_role,job.episode_id,turn.turn_ordinal,request.retry_ordinal;
    `);
    const lines = result.recordset.map((row) => canonicalJson({
      schemaVersion: 1,
      profileKey,
      arm,
      controlId,
      jobId: Number(row.job_id),
      episodeId: row.episode_id,
      splitRole: row.split_role,
      agentRunId: Number(row.agent_run_id),
      turnOrdinal: row.turn_ordinal,
      retryOrdinal: row.retry_ordinal,
      clientRequestId: row.client_request_id,
      requestBodySha256: row.request_body_sha256,
      requestBody: JSON.parse(row.request_body.toString("utf8")),
      responseBodySha256: row.response_body_sha256,
      responseBodyUtf8: row.response_body.toString("utf8"),
      httpStatus: row.http_status,
      finishReason: row.finish_reason,
      promptTokens: row.prompt_tokens === null ? null : Number(row.prompt_tokens),
      completionTokens: row.completion_tokens === null ? null : Number(row.completion_tokens),
      totalTokens: row.total_tokens === null ? null : Number(row.total_tokens),
      reasoningSha256: row.reasoning_sha256,
      reasoningBytes: row.reasoning_bytes,
      parseStatus: row.parse_status,
      repairKind: row.repair_kind,
      errorClass: row.error_class,
      timingMs: { client: row.client_elapsed_ms, headersWait: row.headers_wait_ms, bodyRead: row.body_read_ms, parse: row.parse_ms },
    }));
    const body = lines.length === 0 ? "" : `${lines.join("\n")}\n`;
    const path = `${runDirectory}/raw/model-responses-${safeName(profileKey)}-${safeName(arm)}${controlSuffix}.jsonl`;
    await atomicWrite(path, body, 0o600);
    output.push({ arm, path, rows: lines.length, bytes: Buffer.byteLength(body), sha256: sha256(body) });
  }
  return output;
}

async function retainChatMetrics(label: string) {
  const endpoint = new URL(config.inference.chatBaseUrl);
  endpoint.pathname = "/metrics";
  endpoint.search = "";
  const response = await fetch(endpoint, { signal: AbortSignal.timeout(30_000) });
  const body = await response.text();
  if (!response.ok) throw new Error(`Chat metrics ${label} returned HTTP ${response.status}`);
  const path = `${rawRoot}/chat-metrics-${label}.prom`;
  await atomicWrite(path, body, 0o600);
  return { label, path, bytes: Buffer.byteLength(body), sha256: sha256(body), summary: summarizeChatMetrics(body, profile.modelId) };
}

function zeroDelta(before: ReturnType<typeof summarizeChatMetrics>, after: ReturnType<typeof summarizeChatMetrics>) {
  const fields = ["chatHttpRequests", "successfulRequests", "stopRequests", "lengthRequests", "erroredRequests", "promptTokens", "generationTokens", "latencyObservations", "preemptions"] as const;
  const delta = Object.fromEntries(fields.map((field) => [field, after[field] - before[field]]));
  if (Object.values(delta).some((value) => value !== 0)) throw new Error("A no-op replay observed unexpected chat counter traffic");
  return delta;
}

async function persistEvidence(disposition: string, detail: unknown): Promise<void> {
  const detailJson = canonicalJson(detail);
  await control.request().input("key", sql.VarChar(120), `replay-${safeName(profileKey)}-${safeName(controlId ?? "primary")}-${invocationId}`)
    .input("run", sql.VarChar(120), run.runId).input("disposition", sql.VarChar(40), disposition)
    .input("stage", sql.VarChar(80), controlId === null ? "agent_replay" : "negative_control")
    .input("detail", sql.NVarChar(sql.MAX), detailJson).input("hash", sql.Char(64), sha256(detailJson)).query(`
      INSERT control.evidence_events(event_key,run_id,stage,scientific_tier,disposition,detail_json,detail_sha256)
      VALUES(@key,@run,@stage,'tier1',@disposition,@detail,@hash);
    `);
}

async function validatedReceipt(path: string, disposition: string): Promise<Record<string, unknown>> {
  const parsed = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
  const hash = parsed.receiptSha256;
  if (typeof hash !== "string") throw new Error(`Receipt has no hash: ${path}`);
  const { receiptSha256: _ignored, ...body } = parsed;
  if (hashJson(body) !== hash || parsed.disposition !== disposition) throw new Error(`Invalid ${disposition} receipt: ${path}`);
  return parsed;
}

function numberArgument(name: string, fallback: number, minimum: number, maximum: number): number {
  const value = Number(argument(name) ?? fallback);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(`${name} must be an integer from ${minimum} through ${maximum}`);
  return value;
}

function requiredDecodeConfig(): NonNullable<typeof decodeRegistry.configs[string]> {
  const value = decodeRegistry.configs[decodeConfigId];
  if (value === undefined) throw new Error(`Missing decode configuration ${decodeConfigId}`);
  return value;
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}

function controlArgument(): InferenceControlId | null {
  const value = argument("--control");
  return value === undefined || value === "primary" ? null : parseInferenceControlId(value);
}

function agentControlOptions(): AgentLoopOptions["control"] {
  if (controlId === null) return undefined;
  if (controlId === "shuffled-runbooks-v1") return { controlId, retrievalOverride: "shuffled_runbook" };
  if (controlId === "error-number-mask-v1") return {
    controlId,
    transformToolResult: ({ result }) => {
      const transformed = maskErrorNumbersAndSignatures(result, controlPolicy.controls[controlId]);
      return {
        result: transformed.value,
        metadata: {
          replacementCount: transformed.replacementCount,
          originalSha256: transformed.originalSha256,
          transformedSha256: transformed.transformedSha256,
        },
      };
    },
  };
  return { controlId };
}

function safeName(value: string): string { return value.replace(/[^a-z0-9_.-]+/gi, "-").toLowerCase(); }
function safeError(error: unknown): string { return (error as { message?: string }).message ?? String(error); }
