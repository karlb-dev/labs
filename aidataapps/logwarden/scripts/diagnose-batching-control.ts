import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import {
  generatedContentHashes,
  normalizedRequestSha256,
  normalizedResponseSha256,
} from "../src/invariance-diagnostics.js";
import { connect } from "../src/repository.js";
import { appendExperimentLog, atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";

interface TurnRow {
  side: "control" | "primary";
  episode_id: string;
  prediction_id: string;
  turn_ordinal: number;
  request_body: Buffer;
  request_body_sha256: string;
  response_body: Buffer;
  response_body_sha256: string;
}

const profileArgument = valueAfter("--profile");
if (profileArgument === undefined) throw new Error("--profile is required");
const profileKey: string = profileArgument;
const controlId = valueAfter("--control") ?? "batching-sequential-v1";
if (controlId !== "batching-sequential-v1") throw new Error("This diagnostic is restricted to batching-sequential-v1");
const startedAtUtc = new Date().toISOString();
const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const pool = await connect(config.databases.lab, config.databases.controlName, 600_000);

try {
  const turns = await loadTurns();
  const grouped = new Map<string, { control?: TurnRow; primary?: TurnRow }>();
  for (const turn of turns) {
    const key = `${turn.episode_id}:${turn.turn_ordinal}`;
    const pair = grouped.get(key) ?? {};
    pair[turn.side] = turn;
    grouped.set(key, pair);
  }
  const rows = [...grouped.entries()].map(([key, pair]) => {
    const [episodeId, ordinalText] = splitKey(key);
    if (pair.control === undefined || pair.primary === undefined) {
      return {
        schemaVersion: 1, runId: run.runId, profileKey, controlId, episodeId,
        turnOrdinal: Number(ordinalText), pairStatus: "unmatched",
        controlPresent: pair.control !== undefined, primaryPresent: pair.primary !== undefined,
      };
    }
    const controlContent = generatedContentHashes(pair.control.response_body);
    const primaryContent = generatedContentHashes(pair.primary.response_body);
    return {
      schemaVersion: 1, runId: run.runId, profileKey, controlId, episodeId,
      turnOrdinal: Number(ordinalText), pairStatus: "paired",
      controlPredictionId: Number(pair.control.prediction_id),
      primaryPredictionId: Number(pair.primary.prediction_id),
      exactRequestAgreement: pair.control.request_body_sha256 === pair.primary.request_body_sha256,
      provenanceNormalizedRequestAgreement:
        normalizedRequestSha256(pair.control.request_body) === normalizedRequestSha256(pair.primary.request_body),
      rawResponseAgreement: pair.control.response_body_sha256 === pair.primary.response_body_sha256,
      normalizedResponseAgreement:
        normalizedResponseSha256(pair.control.response_body) === normalizedResponseSha256(pair.primary.response_body),
      contentAgreement: controlContent.contentSha256 === primaryContent.contentSha256,
      reasoningAgreement: controlContent.reasoningSha256 === primaryContent.reasoningSha256,
      evidence: {
        controlRequestSha256: pair.control.request_body_sha256,
        primaryRequestSha256: pair.primary.request_body_sha256,
        controlNormalizedRequestSha256: normalizedRequestSha256(pair.control.request_body),
        primaryNormalizedRequestSha256: normalizedRequestSha256(pair.primary.request_body),
        controlResponseSha256: pair.control.response_body_sha256,
        primaryResponseSha256: pair.primary.response_body_sha256,
        controlNormalizedResponseSha256: normalizedResponseSha256(pair.control.response_body),
        primaryNormalizedResponseSha256: normalizedResponseSha256(pair.primary.response_body),
        controlContent, primaryContent,
      },
    };
  }).sort((left, right) => left.episodeId.localeCompare(right.episodeId) || left.turnOrdinal - right.turnOrdinal);
  const paired = rows.filter((row) => row.pairStatus === "paired") as Array<typeof rows[number] & {
    exactRequestAgreement: boolean; provenanceNormalizedRequestAgreement: boolean; rawResponseAgreement: boolean;
    normalizedResponseAgreement: boolean; contentAgreement: boolean; reasoningAgreement: boolean;
  }>;
  const firstTurns = paired.filter((row) => row.turnOrdinal === 0);
  if (firstTurns.length !== 48 || firstTurns.some((row) => !row.exactRequestAgreement)) {
    throw new Error(`Expected 48 exact first-turn pairs, found ${firstTurns.filter((row) => row.exactRequestAgreement).length}/48`);
  }
  const exactInputPairs = paired.filter((row) => row.exactRequestAgreement);
  const exactInputOutputMismatches = exactInputPairs.filter((row) => !row.normalizedResponseAgreement);
  const unmatchedTurnCount = rows.length - paired.length;
  const allInputsExact = unmatchedTurnCount === 0 && paired.every((row) => row.exactRequestAgreement);
  const classification = exactInputOutputMismatches.length > 0 ? "REQUEST_LEVEL_NON_INVARIANT"
    : allInputsExact ? "AGENT_PIPELINE_INVARIANT"
      : "AGENT_INPUT_DRIFT_REQUEST_LEVEL_INVARIANT";
  const counts = {
    episodeCount: new Set(rows.map((row) => row.episodeId)).size,
    pairedTurnCount: paired.length,
    unmatchedTurnCount,
    exactRequestPairs: exactInputPairs.length,
    provenanceNormalizedRequestPairs: paired.filter((row) => row.provenanceNormalizedRequestAgreement).length,
    rawResponsePairs: paired.filter((row) => row.rawResponseAgreement).length,
    normalizedResponsePairs: paired.filter((row) => row.normalizedResponseAgreement).length,
    contentPairs: paired.filter((row) => row.contentAgreement).length,
    reasoningPairs: paired.filter((row) => row.reasoningAgreement).length,
    exactInputNormalizedOutputPairs: exactInputPairs.filter((row) => row.normalizedResponseAgreement).length,
    exactInputOutputMismatches: exactInputOutputMismatches.length,
    firstTurnExactRequests: firstTurns.filter((row) => row.exactRequestAgreement).length,
    firstTurnNormalizedResponses: firstTurns.filter((row) => row.normalizedResponseAgreement).length,
    firstTurnContent: firstTurns.filter((row) => row.contentAgreement).length,
    firstTurnReasoning: firstTurns.filter((row) => row.reasoningAgreement).length,
  };
  const tableBody = `${rows.map((row) => canonicalJson(row)).join("\n")}\n`;
  const tablePath = `${runDirectory}/tables/batching-diagnostic-${safeName(profileKey)}.jsonl`;
  await atomicWrite(tablePath, tableBody, 0o600);
  const receiptBody = {
    schemaVersion: 1, runId: run.runId, profileKey, controlId, startedAtUtc,
    finishedAtUtc: new Date().toISOString(), classification, counts,
    confounds: [
      "vLLM response_body includes per-request response id and creation time outside the generated choices/usage payload",
      "agent tool-result prompts include execution-local callId values",
      "runbook_search results include execution-local retrievalRunId values",
    ],
    interpretation: classification === "AGENT_INPUT_DRIFT_REQUEST_LEVEL_INVARIANT"
      ? "Identical primary/sequential requests produced identical generated choices; the full-agent NON_INVARIANT label cannot be attributed to batching because later prompts contain different execution identities."
      : "See classification and exact-input mismatch counts.",
    tablePath, tableBytes: Buffer.byteLength(tableBody), tableSha256: sha256(tableBody), disposition: "PASS",
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const receiptPath = `${runDirectory}/metrics/batching-diagnostic-${safeName(profileKey)}.json`;
  await atomicWrite(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  await persistEvidence(receipt);
  await appendExperimentLog(`Diagnosed ${profileKey}/${controlId}: ${classification}; exact-input normalized outputs ${counts.exactInputNormalizedOutputPairs}/${counts.exactRequestPairs}, first-turn choices ${counts.firstTurnNormalizedResponses}/48; receipt ${receipt.receiptSha256}.`);
  console.log(JSON.stringify({ profileKey, controlId, classification, counts, receiptPath, receiptSha256: receipt.receiptSha256, disposition: "PASS" }, null, 2));
} finally {
  await pool.close();
}

async function loadTurns(): Promise<TurnRow[]> {
  const result = await pool.request().input("profile", sql.VarChar(80), profileKey).input("control", sql.VarChar(80), controlId).query<TurnRow>(`
    WITH pairs AS (
      SELECT control_prediction.episode_id,control_prediction.prediction_id control_prediction_id,
        primary_prediction.prediction_id primary_prediction_id,control_prediction.agent_run_id control_agent_run_id,
        primary_prediction.agent_run_id primary_agent_run_id
      FROM eval.predictions control_prediction
      INNER JOIN eval.predictions primary_prediction ON primary_prediction.prediction_id=control_prediction.source_prediction_id
      WHERE control_prediction.control_id=@control AND control_prediction.model_profile_id=@profile
    )
    SELECT 'control' side,pair.episode_id,CONVERT(varchar(30),pair.control_prediction_id) prediction_id,
      turn.turn_ordinal,request.request_body,request.request_body_sha256,response.response_body,response.response_body_sha256
    FROM pairs pair INNER JOIN agent.turns turn ON turn.agent_run_id=pair.control_agent_run_id
    INNER JOIN agent.model_requests request ON request.turn_id=turn.turn_id AND request.retry_ordinal=0
    INNER JOIN agent.model_responses response ON response.model_request_id=request.model_request_id
    UNION ALL
    SELECT 'primary' side,pair.episode_id,CONVERT(varchar(30),pair.primary_prediction_id) prediction_id,
      turn.turn_ordinal,request.request_body,request.request_body_sha256,response.response_body,response.response_body_sha256
    FROM pairs pair INNER JOIN agent.turns turn ON turn.agent_run_id=pair.primary_agent_run_id
    INNER JOIN agent.model_requests request ON request.turn_id=turn.turn_id AND request.retry_ordinal=0
    INNER JOIN agent.model_responses response ON response.model_request_id=request.model_request_id;
  `);
  return result.recordset;
}

async function persistEvidence(receipt: Record<string, unknown>): Promise<void> {
  const detail = canonicalJson(receipt);
  const key = `batching-diagnostic-${safeName(profileKey)}`;
  const prior = await pool.request().input("key", sql.VarChar(120), key)
    .query<{ detail_json: string }>("SELECT detail_json FROM control.evidence_events WHERE event_key=@key;");
  if (prior.recordset[0] !== undefined) {
    if (canonicalJson(JSON.parse(prior.recordset[0].detail_json)) !== detail) throw new Error(`Batching diagnostic drift for ${profileKey}`);
    return;
  }
  await pool.request().input("key", sql.VarChar(120), key).input("run", sql.VarChar(120), run.runId)
    .input("detail", sql.NVarChar(sql.MAX), detail).input("hash", sql.Char(64), sha256(detail)).query(`
      INSERT control.evidence_events(event_key,run_id,stage,scientific_tier,disposition,detail_json,detail_sha256)
      VALUES(@key,@run,'negative_control_diagnostic','tier1','PASS',@detail,@hash);
    `);
}

function splitKey(key: string): [string, string] {
  const index = key.lastIndexOf(":");
  return [key.slice(0, index), key.slice(index + 1)];
}

function safeName(value: string): string { return value.replace(/[^a-z0-9_.-]+/gi, "-").toLowerCase(); }
