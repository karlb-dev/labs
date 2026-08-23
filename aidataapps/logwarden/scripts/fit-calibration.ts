import { readFile } from "node:fs/promises";
import sql from "mssql";
import { calibratedConfidence, calibrationMetrics, fitPlattCalibration, selectActionThreshold, type CalibrationRow } from "../src/calibration.js";
import { loadStandardCampaign } from "../src/campaign.js";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory } from "../src/run.js";

interface StoredRow {
  prediction_id: string;
  agent_arm_id: string;
  confidence: number;
  end_to_end_success: boolean;
  action_correct: boolean;
  scenario_group_id: string;
}

const startedAtUtc = new Date().toISOString();
const profileKey = requiredArgument("--profile");
const campaign = loadStandardCampaign();
if (!campaign.targetProfiles.includes(profileKey)) throw new Error(`${profileKey} is not a governed target profile`);
const arms = [...campaign.mandatoryInferenceArms, campaign.retrievalArm.armId, ...campaign.derivedArms];
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const receiptPath = `${runDirectory}/metrics/calibration-${safeName(profileKey)}.json`;
const pool = await connect(config.databases.lab, config.databases.controlName, 600_000);

try {
  await assertChronology();
  const rows = await loadRows();
  const artifacts: Array<Record<string, unknown>> = [];
  let inserted = 0;
  for (const armId of arms) {
    const selected = rows.filter((row) => row.agent_arm_id === armId);
    const calibrationRows: CalibrationRow[] = selected.map((row) => ({
      confidence: Number(row.confidence), correct: Boolean(row.end_to_end_success), actionCorrect: Boolean(row.action_correct),
    }));
    const model = fitPlattCalibration(calibrationRows);
    const raw = calibrationRows.map((row) => row.confidence);
    const calibrated = calibrationRows.map((row) => calibratedConfidence(model, row.confidence));
    const rawMetrics = calibrationMetrics(calibrationRows, raw);
    const calibratedMetrics = calibrationMetrics(calibrationRows, calibrated);
    const threshold = selectActionThreshold(calibrationRows, calibrated,
      0.85, 0.5);
    const featureManifest = {
      schemaVersion: 1, outcome: "end_to_end_success", feature: "logit(raw_self_confidence)",
      fitRole: "calibration", missingConfidence: "excluded_and_reported", bins: 10,
    };
    const coefficients = { ...model, threshold, rawMetrics, calibratedMetrics };
    const identity = { schemaVersion: 1, runId: run.runId, profileKey, armId, featureManifest, coefficients,
      trainingPredictionIdsSha256: hashJson(selected.map((row) => Number(row.prediction_id))) };
    const modelSha256 = hashJson(identity);
    const calibrationModelId = `${profileKey}-${armId}-e2e-platt-v1`.slice(0, 100);
    inserted += await persistModel(calibrationModelId, armId, selected.length, featureManifest, coefficients, modelSha256);
    artifacts.push({ calibrationModelId, armId, trainingRows: selected.length, groupCount: new Set(selected.map((row) => row.scenario_group_id)).size,
      modelSha256, model, rawMetrics, calibratedMetrics, threshold });
  }
  const tableBody = `${artifacts.map((row) => canonicalJson(row)).join("\n")}\n`;
  const tablePath = `${runDirectory}/tables/calibration-${safeName(profileKey)}.jsonl`;
  await atomicWrite(tablePath, tableBody, 0o600);
  const receiptBody = {
    schemaVersion: 1, runId: run.runId, profileKey, fitRole: "calibration", arms,
    startedAtUtc, finishedAtUtc: new Date().toISOString(), models: artifacts, insertedThisInvocation: inserted,
    tablePath, tableBytes: Buffer.byteLength(tableBody), tableSha256: sha256(tableBody), testInferenceAuthorized: true,
    disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  await appendExperimentLog(`Fitted and hash-locked ${artifacts.length} calibration-only models for ${profileKey} before test inference; receipt ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({ profileKey, arms, models: artifacts.length, inserted, receiptPath, receiptSha256: receipt.receiptSha256, disposition: "PASS" }, null, 2));
} finally {
  await pool.close();
}

async function assertChronology(): Promise<void> {
  const result = await pool.request().input("run", sql.VarChar(120), run.runId).input("profile", sql.VarChar(80), profileKey).query<{
    status: string; test_predictions: number;
  }>(`
    SELECT campaign.status,
      (SELECT COUNT(*) FROM eval.predictions prediction
       INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=prediction.episode_id
       INNER JOIN control.jobs job ON job.job_id=prediction.job_id AND job.campaign_id=campaign.campaign_id
       WHERE prediction.model_profile_id=@profile AND prediction.control_id IS NULL AND truth.split_role LIKE 'test[_]%') test_predictions
    FROM control.campaigns campaign INNER JOIN control.runs run ON run.campaign_id=campaign.campaign_id WHERE run.run_id=@run;
  `);
  const row = result.recordset[0];
  if (row === undefined || !["frozen", "running"].includes(row.status)) throw new Error("Calibration requires a frozen/running campaign");
  if (Number(row.test_predictions) !== 0) throw new Error(`Calibration leakage guard: ${profileKey} already has test predictions`);
}

async function loadRows(): Promise<StoredRow[]> {
  const result = await pool.request().input("run", sql.VarChar(120), run.runId).input("profile", sql.VarChar(80), profileKey)
    .input("arms", sql.NVarChar(sql.MAX), JSON.stringify(arms)).query<StoredRow>(`
      WITH current_campaign AS (SELECT campaign_id FROM control.runs WHERE run_id=@run),
           selected_arms AS (SELECT CONVERT(varchar(80),value) arm_id FROM OPENJSON(@arms))
      SELECT prediction.prediction_id,prediction.agent_arm_id,prediction.confidence,
        score.end_to_end_success,score.action_correct,truth.scenario_group_id
      FROM current_campaign INNER JOIN control.jobs job ON job.campaign_id=current_campaign.campaign_id
      INNER JOIN eval.predictions prediction ON prediction.job_id=job.job_id
      INNER JOIN selected_arms arm ON arm.arm_id=prediction.agent_arm_id
      INNER JOIN eval.ground_truth_episodes truth ON truth.episode_id=prediction.episode_id AND truth.split_role='calibration'
      INNER JOIN eval.decision_scores score ON score.prediction_id=prediction.prediction_id
      WHERE prediction.model_profile_id=@profile AND prediction.control_id IS NULL AND prediction.confidence IS NOT NULL
      ORDER BY prediction.agent_arm_id,truth.episode_id;
    `);
  for (const arm of arms) {
    const count = result.recordset.filter((row) => row.agent_arm_id === arm).length;
    if (count < 20) throw new Error(`Calibration found only ${count} completed confidence rows for ${profileKey}/${arm}`);
  }
  return result.recordset;
}

async function persistModel(id: string, armId: string, rows: number, featureManifest: unknown, coefficients: unknown, modelSha256: string): Promise<number> {
  const prior = await pool.request().input("id", sql.VarChar(100), id).query<{ model_sha256: string }>(`
    SELECT model_sha256 FROM eval.calibration_models WHERE calibration_model_id=@id;
  `);
  if (prior.recordset[0] !== undefined) {
    if (prior.recordset[0].model_sha256 !== modelSha256) throw new Error(`Calibration drift for ${id}`);
    return 0;
  }
  await pool.request().input("id", sql.VarChar(100), id).input("profile", sql.VarChar(80), profileKey)
    .input("arm", sql.VarChar(80), armId).input("features", sql.NVarChar(sql.MAX), canonicalJson(featureManifest))
    .input("coefficients", sql.NVarChar(sql.MAX), canonicalJson(coefficients)).input("rows", sql.Int, rows)
    .input("hash", sql.Char(64), modelSha256).query(`
      INSERT eval.calibration_models(calibration_model_id,model_profile_id,agent_arm_id,fit_role,method,
        feature_manifest_json,coefficients_json,training_row_count,model_sha256,fitted_at_utc)
      VALUES(@id,@profile,@arm,'calibration','platt_logit_l2_v1',@features,@coefficients,@rows,@hash,SYSUTCDATETIME());
    `);
  return 1;
}

function requiredArgument(name: string): string {
  const value = argument(name);
  if (value === undefined) throw new Error(`${name} is required`);
  return value;
}
function argument(name: string): string | undefined { const index = process.argv.indexOf(name); return index < 0 ? undefined : process.argv[index + 1]; }
function safeName(value: string): string { return value.replace(/[^a-z0-9_.-]+/gi, "-").toLowerCase(); }
