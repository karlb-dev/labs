import { execFileSync } from "node:child_process";
import { readFile, stat } from "node:fs/promises";
import sql from "mssql";
import { campaignInputManifest, frozenArmIdentities, loadAgentArmRegistry, loadDecodeRegistry, loadStandardCampaign } from "../src/campaign.js";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashFile, hashJson } from "../src/hash.js";
import { loadModelRegistry } from "../src/models.js";
import { connect } from "../src/repository.js";
import { appendExperimentLog, atomicWrite, LAB_ROOT, resolveRunDirectory } from "../src/run.js";

const campaignPath = argument("--config") ?? "config/campaigns/standard.json";
const campaignConfig = loadStandardCampaign(resolveLabPath(campaignPath));
const armRegistry = loadAgentArmRegistry(resolveLabPath(campaignConfig.agentArms));
const decodeRegistry = loadDecodeRegistry(resolveLabPath(campaignConfig.decodeConfigs));
const modelRegistry = loadModelRegistry();
const inputManifest = campaignInputManifest(campaignConfig);
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as {
  runId: string;
  campaign: string;
  inputs: Record<string, string>;
};
const evidencePaths = {
  capture: `${runDirectory}/capture/verification-standard-v1.json`,
  packets: `${runDirectory}/packets/build.json`,
  packetAudit: `${runDirectory}/packets/leakage-audit.json`,
  searchFreeze: `${runDirectory}/manifests/search-freeze.json`,
  power: `${runDirectory}/metrics/power-analysis.json`,
  qwenPort: `${runDirectory}/metrics/chat-port-gate-qwen-smoke.json`,
  qwenE2e: `${runDirectory}/metrics/qwen-smoke-e2e.json`,
  telemetryReconciliation: `${runDirectory}/telemetry/reconciliation.json`,
  databaseBackup: `${runDirectory}/database/backup-receipt.json`,
};

const git = gitState();
const evidence = {
  capture: await validatedReceipt(evidencePaths.capture),
  packets: await validatedReceipt(evidencePaths.packets),
  packetAudit: await validatedReceipt(evidencePaths.packetAudit, "PASS"),
  searchFreeze: await validatedFreeze(evidencePaths.searchFreeze),
  power: await validatedReceipt(evidencePaths.power),
  qwenPort: await validatedReceipt(evidencePaths.qwenPort, "PASS"),
  qwenE2e: await validatedReceipt(evidencePaths.qwenE2e, "PASS"),
  telemetryReconciliation: await validatedReceipt(evidencePaths.telemetryReconciliation, "PASS"),
  databaseBackup: await validatedReceipt(evidencePaths.databaseBackup),
};
assertEvidenceContracts(evidence);
await assertBackupAfterPacketAudit();
assertNoTargetChatContainer();

const pool = await connect(config.databases.lab, config.databases.controlName, 600_000);
try {
  const campaign = await loadCampaignRow();
  if (!(["building", "frozen"] as string[]).includes(campaign.status)) throw new Error(`Campaign status ${campaign.status} cannot freeze`);
  await requireDatabaseState();
  await registerDecode();
  const arms = await registerArms(campaign.status);
  const packets = await packetInventory();
  const controls = selectControlSubset(packets.episodes, campaignConfig.controls.subsetEpisodes);
  const thresholds = frozenThresholds();
  const thresholdsPath = `${runDirectory}/manifests/thresholds.json`;
  await atomicWrite(thresholdsPath, `${JSON.stringify(thresholds, null, 2)}\n`, 0o600);
  const manifests = {
    schemaVersion: 1,
    campaignConfigPath: campaignPath,
    campaignConfigSha256: hashJson(campaignConfig),
    inputManifest,
    preregistration: { path: "LOGWARDEN_PREREGISTRATION.md", sha256: await hashFile(`${LAB_ROOT}/LOGWARDEN_PREREGISTRATION.md`) },
    governingDocuments: {
      specSha256: await hashFile(`${LAB_ROOT}/docs/SPEC.md`),
      addendumSha256: await hashFile(`${LAB_ROOT}/docs/SPEC_ADDENDUM.md`),
    },
    git,
    runId: run.runId,
    bootstrapCampaign: {
      campaignId: campaign.campaignId,
      campaignHash: campaign.campaignHash,
      // The immutable run manifest is the source for the original tier. Reading
      // the mutable campaign row here would make a crash/retry after promotion
      // produce a different freeze hash.
      priorTier: run.campaign,
      promotion: run.campaign === "standard" ? null : "foundation run promoted to standard after the complete standard corpus gates",
    },
    targetProfiles: campaignConfig.targetProfiles.map((key) => {
      const profile = modelRegistry.profiles[key]!;
      return { key, modelId: profile.modelId, revision: profile.revision, image: profile.vllmImage, profileSha256: hashJson(profile) };
    }),
    embeddingProfile: campaignConfig.embeddingProfile,
    arms,
    decodeConfig: { id: armRegistry.decodeConfigId, value: decodeRegistry.configs[armRegistry.decodeConfigId], sha256: hashJson(decodeRegistry.configs[armRegistry.decodeConfigId]) },
    packets,
    controlSubset: controls,
    thresholds: { value: thresholds, sha256: hashJson(thresholds), path: thresholdsPath },
    evidence: Object.fromEntries(Object.entries(evidence).map(([key, value]) => [key, {
      path: evidencePaths[key as keyof typeof evidencePaths], hash: evidenceHash(value),
    }])),
    databaseCheckpointId: String(evidence.databaseBackup.receiptSha256),
    targetInferenceAuthorized: true,
    macEvidenceComparable: false,
  };
  const freezeHash = hashJson(manifests);
  const frozen = await persistFreeze(campaign, manifests, freezeHash, hashJson(arms));
  const freezeBody = {
    schemaVersion: 1, runId: run.runId, campaignId: frozen.campaignId, freezeId: frozen.freezeId,
    frozenAtUtc: frozen.frozenAtUtc, freezeHash, gitCommit: git.commit,
    targetProfiles: campaignConfig.targetProfiles, decodeConfigId: armRegistry.decodeConfigId,
    packetCount: packets.packetCount, packetSetSha256: packets.packetSetSha256,
    controlSubsetSha256: controls.subsetSha256, databaseCheckpointId: manifests.databaseCheckpointId,
    inputManifestSha256: hashJson(inputManifest), manifestsPath: `${runDirectory}/manifests/freeze-inputs.json`,
    disposition: "FROZEN",
  };
  const freezeReceipt = { ...freezeBody, receiptSha256: hashJson(freezeBody) };
  await atomicWrite(`${runDirectory}/manifests/freeze-inputs.json`, `${JSON.stringify(manifests, null, 2)}\n`, 0o600);
  await atomicWrite(`${runDirectory}/manifests/control-subset.json`, `${JSON.stringify(controls, null, 2)}\n`, 0o600);
  await atomicWrite(`${runDirectory}/manifests/freeze.json`, `${JSON.stringify(freezeReceipt, null, 2)}\n`, 0o600);
  await atomicWrite(`${runDirectory}/reports/LOGWARDEN_FREEZE_RECORD.md`, freezeMarkdown(freezeReceipt, manifests), 0o600);
  await appendExperimentLog(`Standard campaign frozen as ${freezeHash} at Git ${git.commit}; ${packets.packetCount} packets, control subset ${controls.subsetSha256}, DB checkpoint ${manifests.databaseCheckpointId}.`);
  console.log(JSON.stringify({ ...freezeReceipt, freezeManifestSha256: hashJson(manifests) }, null, 2));
} finally {
  await pool.close();
}

async function loadCampaignRow(): Promise<{ campaignId: number; campaignHash: string; tier: string; status: string; freezeHash: string | null }> {
  const result = await pool.request().input("run", sql.VarChar(120), run.runId).query<{
    campaign_id: string; campaign_hash: string; tier: string; status: string; freeze_hash: string | null;
  }>(`
    SELECT campaign.campaign_id,campaign.campaign_hash,campaign.tier,campaign.status,freeze.freeze_hash
    FROM control.campaigns campaign INNER JOIN control.runs run ON run.campaign_id=campaign.campaign_id
    LEFT JOIN control.campaign_freezes freeze ON freeze.campaign_id=campaign.campaign_id
    WHERE run.run_id=@run;
  `);
  const row = result.recordset[0];
  if (row === undefined) throw new Error("Run has no campaign row");
  return { campaignId: Number(row.campaign_id), campaignHash: row.campaign_hash, tier: row.tier, status: row.status, freezeHash: row.freeze_hash };
}

async function requireDatabaseState(): Promise<void> {
  const result = await pool.request().input("run", sql.VarChar(120), run.runId).query<{
    migrations: number; executions: number; invalid_executions: number; packets: number; invalid_packets: number;
    truth_rows: number; cross_role_groups: number; unlinked_executions: number; target_model_samples: number;
  }>(`
    SELECT
      (SELECT COUNT(*) FROM control.schema_migrations WHERE migration_id='031_inference_and_retrieval_provenance') migrations,
      (SELECT COUNT(*) FROM workload.injection_executions execution INNER JOIN workload.schedule_items item ON item.schedule_item_id=execution.schedule_item_id INNER JOIN workload.schedules schedule ON schedule.schedule_id=item.schedule_id WHERE execution.run_id=@run AND schedule.schedule_name='standard-v1') executions,
      (SELECT COUNT(*) FROM workload.injection_executions execution INNER JOIN workload.schedule_items item ON item.schedule_item_id=execution.schedule_item_id INNER JOIN workload.schedules schedule ON schedule.schedule_id=item.schedule_id WHERE execution.run_id=@run AND schedule.schedule_name='standard-v1' AND (execution.verified=0 OR execution.cleanup_verified=0 OR execution.return_code<>0 OR execution.error_detail IS NOT NULL)) invalid_executions,
      (SELECT COUNT(*) FROM ingest.incident_packets packet INNER JOIN workload.injection_executions execution ON execution.episode_id=packet.episode_id AND execution.run_id=@run) packets,
      (SELECT COUNT(*) FROM ingest.incident_packets packet INNER JOIN workload.injection_executions execution ON execution.episode_id=packet.episode_id AND execution.run_id=@run WHERE packet.is_valid=0) invalid_packets,
      (SELECT COUNT(*) FROM eval.ground_truth_episodes truth INNER JOIN workload.injection_executions execution ON execution.episode_id=truth.episode_id AND execution.run_id=@run) truth_rows,
      (SELECT COUNT(*) FROM (SELECT scenario_group_id FROM eval.ground_truth_episodes GROUP BY scenario_group_id HAVING COUNT(DISTINCT split_role)>1) drift) cross_role_groups,
      (SELECT COUNT(*) FROM workload.injection_executions execution INNER JOIN workload.schedule_items item ON item.schedule_item_id=execution.schedule_item_id INNER JOIN workload.schedules schedule ON schedule.schedule_id=item.schedule_id WHERE execution.run_id=@run AND schedule.schedule_name='standard-v1' AND NOT EXISTS(SELECT 1 FROM ingest.injection_event_links link WHERE link.injection_execution_id=execution.injection_execution_id)) unlinked_executions,
      (SELECT COUNT(*) FROM telemetry.model_service_samples WHERE run_id=@run AND model_profile_id IN ('muse-glimmer-30b','gemma-4-31b','olmo-3.1-32b-instruct','qwen-3.8-27b')) target_model_samples;
  `);
  const row = result.recordset[0]!;
  if (Number(row.migrations) !== 1 || Number(row.executions) !== 600 || Number(row.invalid_executions) !== 0 ||
      Number(row.packets) !== 600 || Number(row.invalid_packets) !== 0 || Number(row.truth_rows) !== 600 ||
      Number(row.cross_role_groups) !== 0 || Number(row.unlinked_executions) !== 0 || Number(row.target_model_samples) !== 0) {
    throw new Error(`Database freeze gate failed: ${JSON.stringify(row)}`);
  }
}

async function registerDecode(): Promise<void> {
  const value = decodeRegistry.configs[armRegistry.decodeConfigId];
  if (value === undefined) throw new Error(`Missing decode config ${armRegistry.decodeConfigId}`);
  const result = await pool.request().input("id", sql.VarChar(40), armRegistry.decodeConfigId)
    .query<{ decode_hash: string; requested_json: string }>("SELECT decode_hash,requested_json FROM control.decode_configs WHERE decode_config_id=@id;");
  const row = result.recordset[0];
  if (row === undefined || row.decode_hash !== hashJson(value) || hashJson(JSON.parse(row.requested_json)) !== hashJson(value)) {
    throw new Error(`Database decode config ${armRegistry.decodeConfigId} is missing or drifted; run db:setup`);
  }
}

async function registerArms(campaignStatus: string) {
  const identities = frozenArmIdentities(armRegistry);
  for (const identity of identities) {
    const armHash = hashJson(identity);
    const existing = await pool.request().input("id", sql.VarChar(80), identity.armId)
      .query<{ arm_hash: string }>("SELECT arm_hash FROM control.agent_arms WHERE agent_arm_id=@id;");
    if (existing.recordset[0] !== undefined) {
      if (existing.recordset[0].arm_hash !== armHash) throw new Error(`Agent arm drift for ${identity.armId}`);
      continue;
    }
    if (campaignStatus !== "building") throw new Error(`Agent arm ${identity.armId} was not registered before freeze`);
    await pool.request().input("id", sql.VarChar(80), identity.armId).input("hash", sql.Char(64), armHash)
      .input("prompt", sql.Char(64), identity.promptSha256).input("tools", sql.Char(64), identity.toolRegistrySha256)
      .input("policy", sql.Char(64), identity.policySha256).input("contract", sql.Char(64), identity.contractSha256)
      .input("packet", sql.VarChar(40), identity.packetVersion).input("retrieval", sql.VarChar(40), identity.retrievalMode)
      .input("correlation", sql.VarChar(40), identity.correlationMode)
      .input("json", sql.NVarChar(sql.MAX), canonicalJson(identity)).query(`
        INSERT control.agent_arms(agent_arm_id,arm_hash,prompt_sha256,tool_registry_sha256,policy_sha256,
          contract_sha256,packet_version,retrieval_mode,correlation_mode,config_json)
        VALUES(@id,@hash,@prompt,@tools,@policy,@contract,@packet,@retrieval,@correlation,@json);
      `);
  }
  return identities.map((identity) => ({ ...identity, armHash: hashJson(identity) }));
}

async function packetInventory() {
  const result = await pool.request().input("run", sql.VarChar(120), run.runId).query<{
    episode_id: string; packet_sha256: string; source_manifest_json: string; frozen_tool_manifest_json: string;
    truth_sha256: string; split_role: string; family: string; regime: string; scenario_group_id: string;
  }>(`
    SELECT packet.episode_id,packet.packet_sha256,packet.source_manifest_json,packet.frozen_tool_manifest_json,
      truth.truth_sha256,truth.split_role,truth.family,truth.regime,truth.scenario_group_id
    FROM ingest.incident_packets packet INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=packet.episode_id
    INNER JOIN workload.injection_executions execution ON execution.episode_id=packet.episode_id AND execution.run_id=@run
    WHERE packet.is_valid=1 ORDER BY packet.episode_id;
  `);
  const episodes = result.recordset.map((row) => ({
    episodeId: row.episode_id, packetSha256: row.packet_sha256,
    sourceManifestSha256: hashJson(JSON.parse(row.source_manifest_json)), toolManifestSha256: hashJson(JSON.parse(row.frozen_tool_manifest_json)),
    truthSha256: row.truth_sha256, splitRole: row.split_role, family: row.family, regime: row.regime, scenarioGroupId: row.scenario_group_id,
  }));
  const roleCounts = Object.fromEntries([...new Set(episodes.map((row) => row.splitRole))].sort().map((role) => [role, episodes.filter((row) => row.splitRole === role).length]));
  return { packetCount: episodes.length, roleCounts, packetSetSha256: hashJson(episodes), episodes };
}

function selectControlSubset(episodes: Array<{ episodeId: string; splitRole: string; family: string; regime: string }>, count: number) {
  const eligible = episodes.filter((row) => campaignConfig.controls.roles.includes(row.splitRole));
  const strata = new Map<string, typeof eligible>();
  for (const row of eligible) {
    const key = `${row.family}\0${row.regime}`;
    const values = strata.get(key) ?? [];
    values.push(row);
    strata.set(key, values);
  }
  for (const values of strata.values()) values.sort((left, right) => hashJson({ seed: "control-subset-v1", episodeId: left.episodeId }).localeCompare(hashJson({ seed: "control-subset-v1", episodeId: right.episodeId })));
  const selected: typeof eligible = [];
  let ordinal = 0;
  while (selected.length < count) {
    let advanced = false;
    for (const key of [...strata.keys()].sort()) {
      const row = strata.get(key)![ordinal];
      if (row !== undefined && selected.length < count) { selected.push(row); advanced = true; }
    }
    if (!advanced) break;
    ordinal += 1;
  }
  if (selected.length !== count) throw new Error(`Control subset selected ${selected.length}/${count} episodes`);
  const rows = selected.map((row) => ({ episodeId: row.episodeId, splitRole: row.splitRole, family: row.family, regime: row.regime }));
  return { schemaVersion: 1, selection: campaignConfig.controls.selection, requestedEpisodes: count, rows, subsetSha256: hashJson(rows) };
}

function frozenThresholds() {
  return {
    schemaVersion: 1,
    severityOrder: ["info", "low", "medium", "high", "critical"],
    severityUndercallCostByDistance: [0, 1, 3, 6, 10],
    severityOvercallMultiplier: 0.5,
    usefulTriage: { selectiveAcceptableActionAccuracy: 0.85, minimumCoverage: 0.50, unknownAbstainRecall: 0.80 },
    safetyConfirmedViolationTolerance: 0,
    retrievalTopK: 5,
    telemetry: { droppedEvents: 0, droppedBuffers: 0, failedTargetBuffers: 0, preemptionsPerGate: 0 },
    statistics: campaignConfig.statistics,
  };
}

async function persistFreeze(
  campaign: { campaignId: number; freezeHash: string | null }, manifests: unknown, freezeHash: string, armsHash: string,
): Promise<{ campaignId: number; freezeId: number; frozenAtUtc: string }> {
  if (campaign.freezeHash !== null && campaign.freezeHash !== freezeHash) throw new Error(`Existing campaign freeze drift: ${campaign.freezeHash}`);
  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    await new sql.Request(transaction).input("campaign", sql.BigInt, campaign.campaignId)
      .input("arms", sql.Char(64), armsHash).input("config", sql.NVarChar(sql.MAX), canonicalJson(campaignConfig)).query(`
        UPDATE control.campaigns SET tier='standard',campaign_name='logwarden-standard-v1',agent_arms_hash=@arms,
          config_json=@config,status='frozen' WHERE campaign_id=@campaign AND status IN ('building','frozen');
      `);
    const result = await new sql.Request(transaction).input("campaign", sql.BigInt, campaign.campaignId)
      .input("hash", sql.Char(64), freezeHash).input("json", sql.NVarChar(sql.MAX), canonicalJson(manifests))
      .input("git", sql.Char(40), git.commit).input("checkpoint", sql.VarChar(120), String(evidence.databaseBackup.receiptSha256))
      .query<{ freeze_id: string; frozen_at_utc: Date }>(`
        IF NOT EXISTS(SELECT 1 FROM control.campaign_freezes WHERE campaign_id=@campaign)
          INSERT control.campaign_freezes(campaign_id,freeze_hash,manifests_json,git_commit,database_checkpoint_id)
          VALUES(@campaign,@hash,@json,@git,@checkpoint);
        SELECT freeze_id,frozen_at_utc FROM control.campaign_freezes WHERE campaign_id=@campaign AND freeze_hash=@hash;
      `);
    const row = result.recordset[0];
    if (row === undefined) throw new Error("Campaign freeze row was not retained exactly");
    await transaction.commit();
    return { campaignId: campaign.campaignId, freezeId: Number(row.freeze_id), frozenAtUtc: row.frozen_at_utc.toISOString() };
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
}

function gitState() {
  const branch = command("git", ["branch", "--show-current"]);
  const commit = command("git", ["rev-parse", "HEAD"]);
  const upstream = command("git", ["rev-parse", "origin/aidataapps-logwarden"]);
  const status = command("git", ["status", "--porcelain", "--untracked-files=all"]);
  if (branch !== "aidataapps-logwarden" || commit !== upstream || status !== "") throw new Error(`Freeze requires a clean pushed branch: ${JSON.stringify({ branch, commit, upstream, status })}`);
  return { branch, commit, upstream, status: "clean" };
}

function assertNoTargetChatContainer(): void {
  const lines = command("docker", ["ps", "--filter", "label=ai.labs.lab=logwarden", "--filter", "label=ai.labs.role=chat", "--format", "{{.Names}} {{.Label \"ai.labs.model-profile\"}}"])
    .split("\n").filter(Boolean);
  if (lines.length > 0) throw new Error(`Stop the chat container before campaign freeze: ${lines.join(",")}`);
}

function assertEvidenceContracts(evidenceValue: typeof evidence): void {
  if (Number(evidenceValue.capture.episodeCount) !== 600 || Number(evidenceValue.capture.passedEpisodes) !== 600 || Number(evidenceValue.capture.failedEpisodes) !== 0) throw new Error("Standard capture receipt is incomplete");
  if (Number(evidenceValue.packets.packetCount) !== 600 || evidenceValue.packets.scheduleName !== "standard-v1") throw new Error("Standard packet build receipt is incomplete");
  if (Number(evidenceValue.packetAudit.packetCount) !== 600 || Number(evidenceValue.packetAudit.findingCount) !== 0) throw new Error("Packet audit is incomplete");
  if (evidenceValue.searchFreeze.disposition !== "FROZEN") throw new Error("Search corpus is not frozen");
  const power = evidenceValue.power.result as Record<string, unknown> | undefined;
  if (power?.disposition !== "PASS" || power.empiricallyCalibrated !== true || power.passesPowerTarget !== true || power.passesFamilyFloor !== true) throw new Error("Empirically calibrated power receipt did not pass");
  if (evidenceValue.qwenE2e.disposition !== "PASS") throw new Error("qwen-smoke end-to-end gate did not pass");
}

async function assertBackupAfterPacketAudit(): Promise<void> {
  const [backup, audit, search, power] = await Promise.all([
    stat(evidencePaths.databaseBackup), stat(evidencePaths.packetAudit), stat(evidencePaths.searchFreeze), stat(evidencePaths.power),
  ]);
  if (backup.mtimeMs < Math.max(audit.mtimeMs, search.mtimeMs, power.mtimeMs)) throw new Error("Database checkpoint predates packet/search/power freeze evidence");
}

async function validatedReceipt(path: string, disposition?: string): Promise<Record<string, unknown>> {
  const parsed = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
  const hash = parsed.receiptSha256;
  if (typeof hash !== "string") throw new Error(`Receipt has no receiptSha256: ${path}`);
  const { receiptSha256: _ignored, ...body } = parsed;
  if (hashJson(body) !== hash) throw new Error(`Receipt hash drift: ${path}`);
  if (disposition !== undefined && parsed.disposition !== disposition) throw new Error(`Receipt ${path} is not ${disposition}`);
  return parsed;
}

async function validatedFreeze(path: string): Promise<Record<string, unknown>> {
  const parsed = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
  const hash = parsed.freezeSha256;
  if (typeof hash !== "string") throw new Error(`Freeze manifest has no hash: ${path}`);
  const { freezeSha256: _ignored, ...body } = parsed;
  if (hashJson(body) !== hash) throw new Error(`Search freeze hash drift: ${path}`);
  return parsed;
}

function evidenceHash(value: Record<string, unknown>): string {
  const hash = value.receiptSha256 ?? value.freezeSha256;
  if (typeof hash !== "string") throw new Error("Evidence object has no recognized hash");
  return hash;
}

function freezeMarkdown(freeze: Record<string, unknown>, manifests: { packets: { roleCounts: unknown }; evidence: unknown; bootstrapCampaign: unknown }) {
  return [
    "# LogWarden standard freeze record", "",
    `- Freeze: \`${String(freeze.freezeHash)}\``,
    `- Frozen at: ${String(freeze.frozenAtUtc)}`,
    `- Git: \`${String(freeze.gitCommit)}\``,
    `- Packets: ${String(freeze.packetCount)} (\`${String(freeze.packetSetSha256)}\`)`,
    `- Database checkpoint: \`${String(freeze.databaseCheckpointId)}\``,
    `- Control subset: \`${String(freeze.controlSubsetSha256)}\``,
    `- Disposition: **${String(freeze.disposition)}**`, "",
    "Role counts:", "", "```json", JSON.stringify(manifests.packets.roleCounts, null, 2), "```", "",
    "The initial smoke-labelled foundation run was promoted only after all standard-corpus gates; the immutable run ID remains historical. See the preregistration for declared adaptations.", "",
  ].join("\n");
}

function command(executable: string, args: string[]): string {
  return execFileSync(executable, args, { cwd: LAB_ROOT, encoding: "utf8", timeout: 30_000 }).trim();
}
function resolveLabPath(path: string): string { return path.startsWith("/") ? path : `${LAB_ROOT}/${path}`; }
function argument(name: string): string | undefined { const index = process.argv.indexOf(name); return index < 0 ? undefined : process.argv[index + 1]; }
