import sql from "mssql";
import { readFile } from "node:fs/promises";
import { frozenArmIdentities, loadAgentArmRegistry } from "../src/campaign.js";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";

interface RunManifest { runId: string }
interface CampaignRow { campaign_id: string; status: string }
interface ArmRow {
  agent_arm_id: string; arm_hash: string; prompt_sha256: string; tool_registry_sha256: string;
  policy_sha256: string; contract_sha256: string; packet_version: string; retrieval_mode: string;
  correlation_mode: string; config_json: string;
}

const selectedArms = ["A-direct", "A-rag", "A-tools"] as const;
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const pool = await connect(config.databases.lab, config.databases.controlName, 600_000);
const retryId = new Date().toISOString().replace(/[-:.]/g, "");
const archivePath = `${runDirectory}/manifests/qwen-smoke-pilot-retained-${retryId}.json`;
const receiptPath = `${runDirectory}/manifests/qwen-smoke-pilot-retry-${retryId}.json`;

try {
  const campaign = await loadCampaign();
  if (campaign.status !== "building") throw new Error(`Qwen pilot retry is forbidden after campaign freeze (${campaign.status})`);
  const currentIdentities = frozenArmIdentities(loadAgentArmRegistry())
    .filter((identity) => (selectedArms as readonly string[]).includes(identity.armId));
  if (currentIdentities.length !== selectedArms.length) throw new Error("Current registry does not contain all qwen pilot arms");

  const priorArms = await pool.request().input("arms", sql.NVarChar(sql.MAX), JSON.stringify(selectedArms)).query<ArmRow>(`
    WITH selected AS (SELECT CONVERT(varchar(80),value) agent_arm_id FROM OPENJSON(@arms))
    SELECT arm.agent_arm_id,arm.arm_hash,arm.prompt_sha256,arm.tool_registry_sha256,arm.policy_sha256,
      arm.contract_sha256,arm.packet_version,arm.retrieval_mode,arm.correlation_mode,arm.config_json
    FROM control.agent_arms arm INNER JOIN selected ON selected.agent_arm_id=arm.agent_arm_id
    ORDER BY arm.agent_arm_id;
  `);
  if (priorArms.recordset.length !== selectedArms.length) throw new Error("Registered qwen pilot arm identities are incomplete");
  const priorById = new Map(priorArms.recordset.map((row) => [row.agent_arm_id, row]));
  if (currentIdentities.every((identity) => priorById.get(identity.armId)?.arm_hash === hashJson(identity))) {
    throw new Error("Qwen pilot retry requires an audited pre-freeze arm-identity change");
  }

  const cells = await pool.request().input("campaign", sql.BigInt, Number(campaign.campaign_id))
    .input("run", sql.VarChar(120), run.runId).query<Record<string, unknown>>(`
    SELECT job.job_id,job.episode_id,job.agent_arm_id,job.status job_status,job.attempt_count,
      job.error_class,job.error_detail,item.work_item_id,item.status work_status,item.lease_token,
      prediction.prediction_id,prediction.agent_run_id,prediction.outcome,prediction.failure_stage,
      prediction.failure_class,prediction.prediction_json
    FROM control.jobs job
    INNER JOIN ops.work_items item ON item.job_id=job.job_id AND item.run_id=@run
    LEFT JOIN eval.predictions prediction ON prediction.job_id=job.job_id
    WHERE job.campaign_id=@campaign AND job.run_kind='qwen_smoke_replay' AND job.model_profile_id='qwen-smoke'
    ORDER BY job.agent_arm_id,job.episode_id;
  `);
  const counts = summarizeCells(cells.recordset);
  if (cells.recordset.length !== 180 || counts.jobsFailed !== 180 || counts.predictions !== 180
      || counts.failurePredictions !== 180 || counts.activeLeases !== 0
      || selectedArms.some((arm) => counts.byArm[arm] !== 60)) {
    throw new Error(`Qwen pilot cells are not the exact retryable failed grid: ${canonicalJson(counts)}`);
  }

  const scoreRefs = await pool.request().input("campaign", sql.BigInt, Number(campaign.campaign_id)).query<Record<string, number>>(`
    WITH selected AS
    (
      SELECT prediction.prediction_id FROM eval.predictions prediction
      INNER JOIN control.jobs job ON job.job_id=prediction.job_id
      WHERE job.campaign_id=@campaign AND job.run_kind='qwen_smoke_replay' AND job.model_profile_id='qwen-smoke'
    )
    SELECT
      (SELECT COUNT(*) FROM eval.decision_scores score INNER JOIN selected ON selected.prediction_id=score.prediction_id) decision_scores,
      (SELECT COUNT(*) FROM eval.tool_scores score INNER JOIN selected ON selected.prediction_id=score.prediction_id) tool_scores,
      (SELECT COUNT(*) FROM eval.retrieval_scores score INNER JOIN selected ON selected.prediction_id=score.prediction_id) retrieval_scores,
      (SELECT COUNT(*) FROM eval.predictions prediction INNER JOIN selected ON selected.prediction_id=prediction.source_prediction_id) derived_dependents;
  `);
  const references = scoreRefs.recordset[0]!;
  if (Object.values(references).some((value) => Number(value) !== 0)) {
    throw new Error(`Qwen pilot predictions already have scientific dependents: ${canonicalJson(references)}`);
  }

  const requests = await pool.request().input("campaign", sql.BigInt, Number(campaign.campaign_id)).query<Record<string, unknown>>(`
    SELECT job.job_id,agent_run.agent_run_id,agent_run.trace_id,agent_run.status agent_status,
      request.model_request_id,request.request_body_sha256,response.response_body_sha256,
      response.parse_status,response.repair_kind,response.error_class response_error_class
    FROM control.jobs job
    INNER JOIN agent.agent_runs agent_run ON agent_run.job_id=job.job_id
    INNER JOIN agent.turns turn ON turn.agent_run_id=agent_run.agent_run_id
    INNER JOIN agent.model_requests request ON request.turn_id=turn.turn_id
    INNER JOIN agent.model_responses response ON response.model_request_id=request.model_request_id
    WHERE job.campaign_id=@campaign AND job.run_kind='qwen_smoke_replay' AND job.model_profile_id='qwen-smoke'
    ORDER BY job.job_id,agent_run.agent_run_id,turn.turn_ordinal,request.retry_ordinal;
  `);
  const archiveBody = {
    schemaVersion: 1,
    runId: run.runId,
    retryId,
    reason: "Pre-freeze qwen smoke pilot exposed underspecified arm/tool prompt behavior; retain attempt 1 and authorize a corrected development attempt.",
    campaignStatus: campaign.status,
    cellCounts: counts,
    scoreReferences: references,
    priorArmIdentities: priorArms.recordset.map((row) => ({ ...row, config_json: JSON.parse(row.config_json) })),
    replacementArmIdentities: currentIdentities,
    cells: cells.recordset.map((row) => ({
      ...row,
      prediction_json: typeof row.prediction_json === "string" ? JSON.parse(row.prediction_json) : null,
      lease_token: row.lease_token === null ? null : "REDACTED_PRESENT",
    })),
    modelRequestResponseHashes: requests.recordset,
  };
  const archive = { ...archiveBody, archiveSha256: hashJson(archiveBody) };
  await atomicWrite(archivePath, `${JSON.stringify(archive, null, 2)}\n`, 0o600);

  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    for (const identity of currentIdentities) {
      await new sql.Request(transaction)
        .input("id", sql.VarChar(80), identity.armId)
        .input("arm_hash", sql.Char(64), hashJson(identity))
        .input("prompt", sql.Char(64), identity.promptSha256)
        .input("tools", sql.Char(64), identity.toolRegistrySha256)
        .input("policy", sql.Char(64), identity.policySha256)
        .input("contract", sql.Char(64), identity.contractSha256)
        .input("packet", sql.VarChar(40), identity.packetVersion)
        .input("retrieval", sql.VarChar(40), identity.retrievalMode)
        .input("correlation", sql.VarChar(40), identity.correlationMode)
        .input("json", sql.NVarChar(sql.MAX), canonicalJson(identity))
        .query(`
          UPDATE control.agent_arms SET arm_hash=@arm_hash,prompt_sha256=@prompt,tool_registry_sha256=@tools,
            policy_sha256=@policy,contract_sha256=@contract,packet_version=@packet,retrieval_mode=@retrieval,
            correlation_mode=@correlation,config_json=@json WHERE agent_arm_id=@id;
        `);
    }
    const reset = await new sql.Request(transaction)
      .input("campaign", sql.BigInt, Number(campaign.campaign_id))
      .input("run", sql.VarChar(120), run.runId)
      .query<{ deleted_predictions: number; reset_jobs: number; reset_work_items: number }>(`
        DECLARE @selected TABLE(job_id bigint PRIMARY KEY);
        INSERT @selected(job_id)
        SELECT job_id FROM control.jobs
        WHERE campaign_id=@campaign AND run_kind='qwen_smoke_replay' AND model_profile_id='qwen-smoke';
        DELETE prediction FROM eval.predictions prediction INNER JOIN @selected selected ON selected.job_id=prediction.job_id;
        DECLARE @deleted int=@@ROWCOUNT;
        DECLARE @transitions TABLE(work_item_id bigint,from_state varchar(40));
        UPDATE item SET status='pending',lease_owner=NULL,lease_token=NULL,leased_until_utc=NULL,
          next_attempt_at_utc=SYSUTCDATETIME(),completed_at_utc=NULL
        OUTPUT INSERTED.work_item_id,DELETED.status INTO @transitions
        FROM ops.work_items item INNER JOIN @selected selected ON selected.job_id=item.job_id
        WHERE item.run_id=@run;
        DECLARE @work int=@@ROWCOUNT;
        INSERT ops.transitions(entity_kind,entity_id,from_state,to_state,actor,reason)
        SELECT 'work_item',work_item_id,from_state,'pending','qwen-smoke-pilot-retry',
          'pre-freeze prompt-contract correction; prior attempt retained' FROM @transitions;
        UPDATE job SET status='pending',completed_at_utc=NULL,error_class=NULL,error_detail=NULL
        FROM control.jobs job INNER JOIN @selected selected ON selected.job_id=job.job_id;
        DECLARE @jobs int=@@ROWCOUNT;
        SELECT @deleted deleted_predictions,@jobs reset_jobs,@work reset_work_items;
      `);
    if (Number(reset.recordset[0]?.deleted_predictions) !== 180 || Number(reset.recordset[0]?.reset_jobs) !== 180
        || Number(reset.recordset[0]?.reset_work_items) !== 180) throw new Error("Qwen pilot reset cardinality drifted inside the transaction");
    await transaction.commit();
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }

  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    retryId,
    archivePath,
    archiveSha256: archive.archiveSha256,
    retainedModelRequests: requests.recordset.length,
    resetCells: 180,
    priorAttemptNumber: 1,
    nextAttemptNumber: 2,
    replacementArmSetSha256: hashJson(currentIdentities),
    disposition: "RETAINED_DEVELOPMENT_RETRY",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  const detailJson = canonicalJson(receipt);
  await pool.request().input("key", sql.VarChar(120), `qwen-pilot-retry-${retryId}`)
    .input("run", sql.VarChar(120), run.runId).input("detail", sql.NVarChar(sql.MAX), detailJson)
    .input("hash", sql.Char(64), sha256(detailJson)).query(`
      INSERT control.evidence_events(event_key,run_id,stage,scientific_tier,disposition,detail_json,detail_sha256)
      VALUES(@key,@run,'qwen_smoke_pilot_retry','development','RETAINED_RETRY',@detail,@hash);
    `);
  await appendExperimentLog(`Retained the failed qwen-smoke attempt-1 grid and its ${requests.recordset.length} raw model request/response hashes at ${archivePath}; reset exactly 180 development-only cells for prompt-contract v2 attempt 2, receipt ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({ runId: run.runId, resetCells: 180, retainedModelRequests: requests.recordset.length, archivePath, archiveSha256: archive.archiveSha256, receiptPath, receiptSha256: receipt.receiptSha256, disposition: receipt.disposition }, null, 2));
} finally {
  await pool.close();
}

async function loadCampaign(): Promise<CampaignRow> {
  const result = await pool.request().input("run", sql.VarChar(120), run.runId).query<CampaignRow>(`
    SELECT campaign.campaign_id,campaign.status FROM control.campaigns campaign
    INNER JOIN control.runs run ON run.campaign_id=campaign.campaign_id WHERE run.run_id=@run;
  `);
  if (result.recordset[0] === undefined) throw new Error("Current run has no campaign");
  return result.recordset[0];
}

function summarizeCells(rows: Array<Record<string, unknown>>) {
  return {
    jobsFailed: rows.filter((row) => row.job_status === "failed").length,
    predictions: rows.filter((row) => row.prediction_id !== null).length,
    failurePredictions: rows.filter((row) => row.outcome === "failure").length,
    activeLeases: rows.filter((row) => row.lease_token !== null).length,
    byArm: Object.fromEntries(selectedArms.map((arm) => [arm, rows.filter((row) => row.agent_arm_id === arm).length])) as Record<typeof selectedArms[number], number>,
  };
}
