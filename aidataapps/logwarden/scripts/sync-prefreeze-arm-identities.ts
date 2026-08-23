import { readFile } from "node:fs/promises";
import sql from "mssql";
import {
  assertAuditableNonModelMetadataDrift,
  differingArmIdentityFields,
} from "../src/arm-identity-sync.js";
import { frozenArmIdentities, type FrozenArmIdentity } from "../src/campaign.js";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";

interface ArmRow {
  agent_arm_id: string;
  arm_hash: string;
  config_json: string;
}

const expectedDrift = new Map<string, ReadonlyArray<keyof FrozenArmIdentity>>([
  ["B0-majority-no-action-v1", ["contractSha256"]],
  ["B1-rules-v1", ["contractSha256"]],
  ["B2-hybrid-v1", ["toolRegistrySha256", "contractSha256"]],
  ["B2-lexical-v1", ["toolRegistrySha256", "contractSha256"]],
  ["B2-vector-v1", ["toolRegistrySha256", "contractSha256"]],
  ["B3-oracle-packet-v1", ["contractSha256"]],
]);

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const pool = await connect(config.databases.lab, config.databases.controlName, 600_000);
const syncId = new Date().toISOString().replace(/[-:.]/g, "");
const archivePath = `${runDirectory}/manifests/prefreeze-arm-identity-sync-${syncId}.json`;

try {
  const campaign = await pool.request().input("run", sql.VarChar(120), run.runId).query<{
    campaign_id: string;
    status: string;
    freeze_count: number;
  }>(`
    SELECT campaign.campaign_id,campaign.status,
      (SELECT COUNT(*) FROM control.campaign_freezes freeze WHERE freeze.campaign_id=campaign.campaign_id) freeze_count
    FROM control.campaigns campaign
    INNER JOIN control.runs run ON run.campaign_id=campaign.campaign_id
    WHERE run.run_id=@run;
  `);
  const campaignRow = campaign.recordset[0];
  if (campaignRow === undefined || campaignRow.status !== "building" || Number(campaignRow.freeze_count) !== 0) {
    throw new Error("Arm identity synchronization is allowed only for an unfrozen building campaign");
  }

  const replacements = frozenArmIdentities();
  const registered = await pool.request().query<ArmRow>(`
    SELECT agent_arm_id,arm_hash,config_json FROM control.agent_arms ORDER BY agent_arm_id;
  `);
  const registeredById = new Map(registered.recordset.map((row) => [row.agent_arm_id, row]));
  const drift = replacements.flatMap((replacement) => {
    const stored = registeredById.get(replacement.armId);
    if (stored === undefined) throw new Error(`Campaign arm ${replacement.armId} is not registered`);
    if (stored.arm_hash === hashJson(replacement)) return [];
    const prior = JSON.parse(stored.config_json) as FrozenArmIdentity;
    const differences = assertAuditableNonModelMetadataDrift(prior, replacement);
    return [{ prior, replacement, storedHash: stored.arm_hash, replacementHash: hashJson(replacement), differences }];
  });
  const actualIds = drift.map((entry) => entry.replacement.armId).sort();
  const expectedIds = [...expectedDrift.keys()].sort();
  if (canonicalJson(actualIds) !== canonicalJson(expectedIds)) {
    throw new Error(`Unexpected pre-freeze arm drift set: ${canonicalJson(actualIds)}`);
  }
  for (const entry of drift) {
    const expected = expectedDrift.get(entry.replacement.armId)!;
    if (canonicalJson(entry.differences) !== canonicalJson(expected)) {
      throw new Error(`Unexpected drift fields for ${entry.replacement.armId}: ${entry.differences.join(",")}`);
    }
  }

  const inventory = await pool.request()
    .input("campaign", sql.BigInt, Number(campaignRow.campaign_id))
    .input("arms", sql.NVarChar(sql.MAX), JSON.stringify(actualIds))
    .query<Record<string, unknown>>(`
      WITH selected AS (SELECT CONVERT(varchar(80),value) agent_arm_id FROM OPENJSON(@arms))
      SELECT prediction.agent_arm_id,truth.split_role,COUNT(*) prediction_count,
        SUM(CASE WHEN score.prediction_id IS NULL THEN 0 ELSE 1 END) decision_score_count,
        SUM(CASE WHEN tool.prediction_id IS NULL THEN 0 ELSE 1 END) tool_score_count,
        SUM(CASE WHEN retrieval.prediction_id IS NULL THEN 0 ELSE 1 END) retrieval_score_count
      FROM eval.predictions prediction
      INNER JOIN selected ON selected.agent_arm_id=prediction.agent_arm_id
      INNER JOIN control.jobs job ON job.job_id=prediction.job_id AND job.campaign_id=@campaign
      INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=prediction.episode_id
      LEFT JOIN eval.decision_scores score ON score.prediction_id=prediction.prediction_id
      LEFT JOIN (SELECT DISTINCT prediction_id FROM eval.tool_scores) tool ON tool.prediction_id=prediction.prediction_id
      LEFT JOIN (SELECT DISTINCT prediction_id FROM eval.retrieval_scores) retrieval ON retrieval.prediction_id=prediction.prediction_id
      GROUP BY prediction.agent_arm_id,truth.split_role
      ORDER BY prediction.agent_arm_id,truth.split_role;
    `);
  const badInventory = inventory.recordset.filter((row) =>
    !["dev", "calibration"].includes(String(row.split_role)) || Number(row.prediction_count) !== 60,
  );
  if (inventory.recordset.length !== expectedIds.length * 2 || badInventory.length > 0) {
    throw new Error(`Baseline prediction inventory is not the exact dev/calibration grid: ${canonicalJson(inventory.recordset)}`);
  }

  const blockers = await pool.request()
    .input("campaign", sql.BigInt, Number(campaignRow.campaign_id))
    .input("arms", sql.NVarChar(sql.MAX), JSON.stringify(actualIds))
    .input("targets", sql.NVarChar(sql.MAX), JSON.stringify([
      "muse-glimmer-30b",
      "gemma-4-31b",
      "olmo-3.1-32b-instruct",
      "qwen-3.8-27b",
    ]))
    .query<Record<string, number>>(`
      WITH selected AS (SELECT CONVERT(varchar(80),value) agent_arm_id FROM OPENJSON(@arms)),
      targets AS (SELECT CONVERT(varchar(80),value) model_profile_id FROM OPENJSON(@targets))
      SELECT
        (SELECT COUNT(*) FROM eval.predictions prediction INNER JOIN control.jobs job ON job.job_id=prediction.job_id
          INNER JOIN selected ON selected.agent_arm_id=prediction.agent_arm_id
          INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=prediction.episode_id
          WHERE job.campaign_id=@campaign AND truth.split_role LIKE 'test[_]%') protected_predictions,
        (SELECT COUNT(*) FROM eval.predictions prediction INNER JOIN control.jobs job ON job.job_id=prediction.job_id
          INNER JOIN targets ON targets.model_profile_id=prediction.model_profile_id
          WHERE job.campaign_id=@campaign) target_predictions,
        (SELECT COUNT(*) FROM telemetry.model_service_samples sample INNER JOIN targets ON targets.model_profile_id=sample.model_profile_id) target_samples,
        (SELECT COUNT(*) FROM control.jobs job INNER JOIN selected ON selected.agent_arm_id=job.agent_arm_id
          WHERE job.campaign_id=@campaign AND job.status NOT IN ('completed','failed','stopped')) active_jobs,
        (SELECT COUNT(*) FROM ops.work_items item INNER JOIN control.jobs job ON job.job_id=item.job_id
          INNER JOIN selected ON selected.agent_arm_id=job.agent_arm_id
          WHERE job.campaign_id=@campaign AND item.status NOT IN ('completed','failed','stopped')) active_work_items;
    `);
  const blockerCounts = blockers.recordset[0]!;
  if (Object.values(blockerCounts).some((value) => Number(value) !== 0)) {
    throw new Error(`Protected or active evidence forbids arm synchronization: ${canonicalJson(blockerCounts)}`);
  }

  const archiveBody = {
    schemaVersion: 1,
    runId: run.runId,
    syncId,
    reason: "Prompt-contract v2 changed global contract metadata and prompt-facing runbook_search schemas after non-model dev/calibration baselines were registered; their executable policies and outputs did not change.",
    campaignStatus: campaignRow.status,
    drift: drift.map((entry) => ({
      armId: entry.replacement.armId,
      differingFields: differingArmIdentityFields(entry.prior, entry.replacement),
      storedHash: entry.storedHash,
      replacementHash: entry.replacementHash,
      prior: entry.prior,
      replacement: entry.replacement,
    })),
    predictionInventory: inventory.recordset,
    blockers: blockerCounts,
    resultImpact: "Identity metadata synchronization only; 720 dev/calibration predictions and all scores remain byte-for-byte unchanged, with no protected-role or governed-target evidence present.",
  };
  const archive = { ...archiveBody, archiveSha256: hashJson(archiveBody) };
  await atomicWrite(archivePath, `${JSON.stringify(archive, null, 2)}\n`, 0o600);

  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    for (const entry of drift) {
      const identity = entry.replacement;
      const result = await new sql.Request(transaction)
        .input("id", sql.VarChar(80), identity.armId)
        .input("prior", sql.Char(64), entry.storedHash)
        .input("arm_hash", sql.Char(64), entry.replacementHash)
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
            correlation_mode=@correlation,config_json=@json
          WHERE agent_arm_id=@id AND arm_hash=@prior;
          SELECT @@ROWCOUNT updated;
        `);
      if (Number(result.recordset[0]?.updated) !== 1) throw new Error(`Concurrent arm identity change for ${identity.armId}`);
    }
    const detailJson = canonicalJson({ archivePath, archiveSha256: archive.archiveSha256, drift: archiveBody.drift });
    await new sql.Request(transaction)
      .input("key", sql.VarChar(120), `prefreeze-arm-sync-${syncId}`)
      .input("run", sql.VarChar(120), run.runId)
      .input("detail", sql.NVarChar(sql.MAX), detailJson)
      .input("hash", sql.Char(64), sha256(detailJson))
      .query(`
        INSERT control.evidence_events(event_key,run_id,stage,scientific_tier,disposition,detail_json,detail_sha256)
        VALUES(@key,@run,'prefreeze_arm_identity_sync','development','METADATA_SYNC',@detail,@hash);
      `);
    await transaction.commit();
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }

  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    syncId,
    archivePath,
    archiveSha256: archive.archiveSha256,
    updatedArms: actualIds,
    predictionCount: inventory.recordset.reduce((sum, row) => sum + Number(row.prediction_count), 0),
    replacementArmSetSha256: hashJson(drift.map((entry) => entry.replacement)),
    disposition: "PREFREEZE_METADATA_SYNC",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const receiptPath = `${runDirectory}/manifests/prefreeze-arm-identity-sync-receipt-${syncId}.json`;
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  await appendExperimentLog(`Audited pre-freeze metadata synchronization updated ${actualIds.length} non-model arm identities after prompt-contract v2; retained 720 unchanged dev/calibration predictions, archive ${archive.archiveSha256}, receipt ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({ ...receipt, receiptPath }, null, 2));
} finally {
  await pool.close();
}
