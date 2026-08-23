import { readFile } from "node:fs/promises";
import sql from "mssql";
import { frozenArmIdentities } from "../src/campaign.js";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { applyRulesBaseline, loadRulesBaseline } from "../src/rules-baseline.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";
import { ingestTelemetryJournal } from "../src/telemetry-ingest.js";
import { createComponentTelemetryJournal } from "../src/telemetry.js";

const allowedRoles = ["dev", "calibration", "test_id", "test_variant_holdout", "test_unknown", "test_live_parity", "test_storm"] as const;
const roles = roleArgument();
const rulesPath = argument("--rules") ?? "config/baselines/rules-v1.json";
const ruleset = loadRulesBaseline(rulesPath);
const rulesetSha256 = hashJson(ruleset);
const armId = ruleset.baselineId;
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const pool = await connect(config.databases.lab, config.databases.controlName, 300_000);
const created = await createComponentTelemetryJournal(runDirectory, run.runId, "baseline-b1-rules");
const root = await created.journal.startSpan("baseline.B1", {}, { armId, rulesPath, rulesetSha256, roles });

try {
  const campaign = await campaignState();
  if (campaign.status !== "frozen" && roles.some((role) => role.startsWith("test_"))) {
    throw new Error("Test-role B1 predictions are forbidden before campaign freeze");
  }
  await registerArm(campaign.status);
  const packets = await loadPackets();
  if (packets.length === 0) throw new Error(`No valid packets exist for ${roles.join(",")}`);
  const outputs: Array<Record<string, unknown>> = [];
  let inserted = 0;
  for (const packet of packets) {
    const parsedPacket = JSON.parse(packet.packet_json) as unknown;
    if (hashJson(parsedPacket) !== packet.packet_sha256) throw new Error(`Packet hash drift for ${packet.episode_id}`);
    const result = applyRulesBaseline(parsedPacket, ruleset);
    const prediction = {
      schemaVersion: 1,
      predictionSource: "B1",
      baselineId: armId,
      episodeId: packet.episode_id,
      splitRole: packet.split_role,
      packetSha256: packet.packet_sha256,
      rulesetSha256,
      resolved: result.resolved,
      matchedRuleId: result.matchedRuleId,
      matchedEventOrdinals: result.matchedEventOrdinals,
      ...result.prediction,
    };
    const persisted = await persistPrediction(campaign.campaignId, packet, prediction);
    inserted += persisted.inserted ? 1 : 0;
    outputs.push({ ...prediction, jobId: persisted.jobId, predictionId: persisted.predictionId, predictionSha256: hashJson(prediction) });
    await created.journal.record("point", "baseline.B1.prediction", {
      traceId: root.traceId, parentSpanId: root.spanId, episodeId: packet.episode_id, jobId: persisted.jobId,
    }, {
      armId, splitRole: packet.split_role, resolved: result.resolved, matchedRuleId: result.matchedRuleId,
      predictionId: persisted.predictionId, predictionSha256: hashJson(prediction), inserted: persisted.inserted,
    }, { status: "success" });
  }
  const coverage = Object.fromEntries(roles.map((role) => {
    const selected = outputs.filter((row) => row.splitRole === role);
    const resolved = selected.filter((row) => row.resolved === true).length;
    return [role, { episodes: selected.length, resolved, coverage: selected.length === 0 ? 0 : resolved / selected.length }];
  }));
  const matchedRules = Object.fromEntries([...new Set(outputs.map((row) => String(row.matchedRuleId ?? "unresolved")))].sort()
    .map((rule) => [rule, outputs.filter((row) => String(row.matchedRuleId ?? "unresolved") === rule).length]));
  const orderedPredictionSetSha256 = hashJson(outputs.map((row) => ({ episodeId: row.episodeId, predictionSha256: row.predictionSha256 })));
  const tablePath = `${runDirectory}/tables/baseline-B1-${safeName(roles.join("-"))}.jsonl`;
  await atomicWrite(tablePath, outputs.map((row) => canonicalJson(row)).join("\n") + "\n", 0o600);
  await created.journal.endSpan(root, "success", { episodes: outputs.length, inserted, coverage, orderedPredictionSetSha256 });
  await created.journal.flush();
  const ingestion = await ingestTelemetryJournal(pool, run.runId, created.path, { batchSize: 100 });
  const receiptBody = {
    schemaVersion: 1, runId: run.runId, campaignId: campaign.campaignId, campaignStatus: campaign.status,
    baselineId: armId, rulesPath, rulesetSha256, roles, episodeCount: outputs.length,
    insertedThisInvocation: inserted, resumed: outputs.length - inserted, coverage, matchedRules,
    tablePath, orderedPredictionSetSha256, telemetry: { journalPath: created.path, ingestion }, disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const receiptPath = `${runDirectory}/metrics/baseline-B1-${safeName(roles.join("-"))}.json`;
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  await appendExperimentLog(`B1 ${armId} produced ${outputs.length} ${roles.join(",")} predictions at ${JSON.stringify(coverage)}; receipt ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({ runId: run.runId, baselineId: armId, roles, episodes: outputs.length, inserted, coverage, matchedRules, receiptPath, receiptSha256: receipt.receiptSha256 }, null, 2));
} catch (error) {
  const errorDetail = safeError(error);
  await created.journal.endSpan(root, "failed", { errorDetail }).catch(() => undefined);
  await created.journal.flush().catch(() => undefined);
  console.error(JSON.stringify({ runId: run.runId, baselineId: armId, roles, errorDetail, disposition: "STOP_DATA" }, null, 2));
  process.exitCode = 1;
} finally {
  await pool.close();
}

interface PacketRow { episode_id: string; split_role: string; packet_json: string; packet_sha256: string }

async function campaignState(): Promise<{ campaignId: number; status: string }> {
  const result = await pool.request().input("run", sql.VarChar(120), run.runId).query<{ campaign_id: string; status: string }>(`
    SELECT campaign.campaign_id,campaign.status FROM control.campaigns campaign
    INNER JOIN control.runs run ON run.campaign_id=campaign.campaign_id WHERE run.run_id=@run;
  `);
  const row = result.recordset[0];
  if (row === undefined || !["building", "frozen", "running"].includes(row.status)) throw new Error("Run has no eligible campaign");
  return { campaignId: Number(row.campaign_id), status: row.status };
}

async function registerArm(campaignStatus: string): Promise<void> {
  const identity = frozenArmIdentities().find((value) => value.armId === armId);
  if (identity === undefined) throw new Error(`Frozen arm registry omits ${armId}`);
  const configJson = canonicalJson(identity);
  const armHash = hashJson(identity);
  const existing = await pool.request().input("id", sql.VarChar(80), armId)
    .query<{ arm_hash: string }>("SELECT arm_hash FROM control.agent_arms WHERE agent_arm_id=@id;");
  if (existing.recordset[0] !== undefined) {
    if (existing.recordset[0].arm_hash !== armHash) throw new Error(`B1 arm registry drift for ${armId}`);
    return;
  }
  if (campaignStatus !== "building") throw new Error(`B1 arm ${armId} was not registered before campaign freeze`);
  await pool.request().input("id", sql.VarChar(80), armId).input("arm_hash", sql.Char(64), armHash)
    .input("prompt", sql.Char(64), identity.promptSha256).input("tools", sql.Char(64), identity.toolRegistrySha256)
    .input("policy", sql.Char(64), identity.policySha256).input("contract", sql.Char(64), identity.contractSha256)
    .input("packet", sql.VarChar(40), identity.packetVersion).input("retrieval", sql.VarChar(40), identity.retrievalMode)
    .input("correlation", sql.VarChar(40), identity.correlationMode).input("config", sql.NVarChar(sql.MAX), configJson)
    .query(`INSERT control.agent_arms(agent_arm_id,arm_hash,prompt_sha256,tool_registry_sha256,policy_sha256,
      contract_sha256,packet_version,retrieval_mode,correlation_mode,config_json)
      VALUES(@id,@arm_hash,@prompt,@tools,@policy,@contract,@packet,@retrieval,@correlation,@config);`);
}

async function loadPackets(): Promise<PacketRow[]> {
  const result = await pool.request().input("run", sql.VarChar(120), run.runId).query<PacketRow>(`
    SELECT packet.episode_id,packet.split_role,packet.packet_json,packet.packet_sha256
    FROM ingest.incident_packets packet
    INNER JOIN workload.injection_executions execution ON execution.episode_id=packet.episode_id AND execution.run_id=@run
    WHERE packet.is_valid=1 ORDER BY packet.split_role,packet.episode_id;
  `);
  return result.recordset.filter((row) => roles.includes(row.split_role as typeof roles[number]));
}

async function persistPrediction(campaignId: number, packet: PacketRow, prediction: Record<string, unknown>): Promise<{ jobId: number; predictionId: number; inserted: boolean }> {
  const jobKey = hashJson({ schemaVersion: 1, campaignId, runKind: "baseline", episodeId: packet.episode_id, agentArmId: armId, predictionSource: "B1", sampleIndex: 0 });
  const transaction = new sql.Transaction(pool);
  await transaction.begin(sql.ISOLATION_LEVEL.SERIALIZABLE);
  try {
    const jobResult = await new sql.Request(transaction).input("key", sql.Char(64), jobKey)
      .input("campaign", sql.BigInt, campaignId).input("episode", sql.VarChar(120), packet.episode_id)
      .input("arm", sql.VarChar(80), armId).query<{ job_id: string }>(`
        IF NOT EXISTS(SELECT 1 FROM control.jobs WHERE job_key=@key)
          INSERT control.jobs(job_key,campaign_id,run_kind,episode_id,agent_arm_id,status,started_at_utc,completed_at_utc)
          VALUES(@key,@campaign,'baseline',@episode,@arm,'complete',SYSUTCDATETIME(),SYSUTCDATETIME());
        SELECT job_id FROM control.jobs WHERE job_key=@key;
      `);
    const jobId = Number(jobResult.recordset[0]!.job_id);
    const prior = await new sql.Request(transaction).input("job", sql.BigInt, jobId)
      .query<{ prediction_id: string; prediction_json: string }>("SELECT prediction_id,prediction_json FROM eval.predictions WHERE job_id=@job;");
    if (prior.recordset[0] !== undefined) {
      if (canonicalJson(JSON.parse(prior.recordset[0].prediction_json)) !== canonicalJson(prediction)) throw new Error(`B1 prediction drift for ${packet.episode_id}`);
      await transaction.commit();
      return { jobId, predictionId: Number(prior.recordset[0].prediction_id), inserted: false };
    }
    const value = prediction as { incidentClass: string; severity: string; action: string; confidence: number; abstain: boolean };
    const inserted = await new sql.Request(transaction).input("job", sql.BigInt, jobId).input("episode", sql.VarChar(120), packet.episode_id)
      .input("arm", sql.VarChar(80), armId).input("class", sql.VarChar(80), value.incidentClass)
      .input("severity", sql.VarChar(24), value.severity).input("action", sql.VarChar(100), value.action)
      .input("confidence", sql.Decimal(9, 6), value.confidence).input("abstain", sql.Bit, value.abstain)
      .input("outcome", sql.VarChar(32), value.abstain ? "abstention" : "decision")
      .input("json", sql.NVarChar(sql.MAX), canonicalJson(prediction)).query<{ prediction_id: string }>(`
        INSERT eval.predictions(job_id,episode_id,agent_arm_id,prediction_source,predicted_class,predicted_severity,
          predicted_action,confidence,abstained,outcome,complete_case_eligible,prediction_json)
        OUTPUT inserted.prediction_id VALUES(@job,@episode,@arm,'B1',@class,@severity,@action,@confidence,@abstain,@outcome,1,@json);
      `);
    await transaction.commit();
    return { jobId, predictionId: Number(inserted.recordset[0]!.prediction_id), inserted: true };
  } catch (error) {
    await transaction.rollback().catch(() => undefined);
    throw error;
  }
}

function roleArgument(): Array<typeof allowedRoles[number]> {
  const source = argument("--roles") ?? "dev";
  const values = [...new Set(source.split(",").map((value) => value.trim()).filter(Boolean))];
  if (values.length === 0 || !values.every((value) => (allowedRoles as readonly string[]).includes(value))) throw new Error(`Invalid roles: ${source}`);
  return values as Array<typeof allowedRoles[number]>;
}

function safeName(value: string): string { return value.replace(/[^a-z0-9_.-]+/gi, "-").toLowerCase(); }
function safeError(error: unknown): string { return (error as { message?: string }).message ?? String(error); }
function argument(name: string): string | undefined { const index = process.argv.indexOf(name); return index < 0 ? undefined : process.argv[index + 1]; }
