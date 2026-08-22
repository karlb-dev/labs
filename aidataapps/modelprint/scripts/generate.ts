import { appendFile, readFile, writeFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashFile, hashJson, sha256 } from "../src/hash.js";
import { buildChatRequest, VllmGateway } from "../src/inference.js";
import { loadModelRegistry, resolveModelProfile } from "../src/models.js";
import { maskNames, normalizeText, scanSelfName, scanTemplateResidue } from "../src/prompt-bank.js";
import { appendExperimentLog, resolveRunDirectory, valueAfter } from "../src/run.js";
import type { DecodeConfig } from "../src/types.js";

interface JobRow {
  generation_job_id: number;
  job_key: string;
  prompt_variant_id: string;
  rendered_text: string;
  render_sha256: string;
  prompt_group_id: string;
  split: string;
  decode_config_id: string;
  config_json: string;
  attempts: number;
}
interface AttemptRow {
  jobKey: string; attempt: number; request: string; raw: string | null; status: number | null; latency: number | null;
  errorClass: string | null; errorDetail: string | null; started: string; finished: string;
}
interface SuccessRow {
  jobKey: string; request: string; raw: string; finalText: string; reasoningText: string | null; outputSha: string; normalizedSha: string;
  inputTokens: number | null; outputTokens: number | null; chars: number; words: number; latency: number; finishReason: string | null;
  status: number; attempts: number; truncated: boolean; residue: boolean; selfName: boolean; started: string; finished: string;
}

const profileKey = valueAfter("--profile") ?? process.env.MODEL_PROFILE;
if (!profileKey) throw new Error("Pass --profile");
const limit = Number.parseInt(valueAfter("--limit") ?? "2147483647", 10);
if (!Number.isInteger(limit) || limit < 1) throw new Error("--limit must be a positive integer");
const runDirectory = resolveRunDirectory();
const freezePath = `${runDirectory}/manifests/campaign-freeze.json`;
const freeze = JSON.parse(await readFile(freezePath, "utf8")) as {
  campaignId: number; campaignHash: string; freezeHash: string; expectedJobs: number; campaign: { profiles: string[] };
  governedFileHashes: Record<string, string>;
};
if (!freeze.campaign.profiles.includes(profileKey)) throw new Error(`${profileKey} is not a frozen target profile`);
for (const [path, expected] of Object.entries(freeze.governedFileHashes)) {
  const actual = await hashFile(path);
  if (actual !== expected) throw new Error(`STOP_DATA: governed file drift ${path}: ${expected} -> ${actual}`);
}
const gate = JSON.parse(await readFile(`${runDirectory}/environment/port-gate-${profileKey}.json`, "utf8")) as { status?: string; profile?: string };
if (gate.status !== "PASS" || gate.profile !== profileKey) throw new Error(`STOP_PORT: ${profileKey} has no passing port gate in this run`);

const profile = resolveModelProfile(profileKey, loadModelRegistry());
const config = loadConfig();
const gateway = new VllmGateway(900_000);
await gateway.ready(config.inference.chatBaseUrl);
const concurrency = Math.min(profile.maxNumSeqs, Number.parseInt(valueAfter("--concurrency") ?? "32", 10));
const checkpointSize = Number.parseInt(valueAfter("--checkpoint-size") ?? "200", 10);
if (!Number.isInteger(concurrency) || concurrency < 1 || !Number.isInteger(checkpointSize) || checkpointSize < 1) throw new Error("Invalid concurrency/checkpoint size");

const pool = await new sql.ConnectionPool({ server: config.database.server, port: config.database.port, user: config.database.user,
  password: config.database.password, database: config.database.database, options: { encrypt: false, trustServerCertificate: true },
  requestTimeout: 900_000, pool: { min: 0, max: 20 } }).connect();
const rawPath = `${runDirectory}/raw/generations-${profileKey}.jsonl`;
const failurePath = `${runDirectory}/raw/failures-${profileKey}.jsonl`;
const checkpointPath = `${runDirectory}/checkpoints/generation-${profileKey}.json`;
const startedAt = new Date();

async function runJob(job: JobRow): Promise<{ rawLine: string; failureLine?: string; attempts: AttemptRow[]; success?: SuccessRow }> {
  const decode = JSON.parse(job.config_json) as DecodeConfig;
  const request = buildChatRequest(profile, [{ role: "user", content: job.rendered_text }], decode);
  const attempts: AttemptRow[] = [];
  const overallStarted = new Date();
  for (let retry = 0; retry < 3; retry += 1) {
    const attempt = job.attempts + retry + 1;
    const attemptStarted = new Date();
    try {
      const result = await gateway.completeMessages(config.inference.chatBaseUrl, profile, [{ role: "user", content: job.rendered_text }], decode);
      const finished = new Date();
      attempts.push({ jobKey: job.job_key, attempt, request: JSON.stringify(result.request), raw: JSON.stringify(result.rawResponse), status: result.httpStatus,
        latency: result.latencyMs, errorClass: null, errorDetail: null, started: attemptStarted.toISOString(), finished: finished.toISOString() });
      const normalized = normalizeText(result.finalText);
      const residue = scanTemplateResidue(result.finalText);
      const selfName = scanSelfName(result.finalText);
      const success: SuccessRow = { jobKey: job.job_key, request: JSON.stringify(result.request), raw: JSON.stringify(result.rawResponse), finalText: result.finalText,
        reasoningText: result.reasoningText, outputSha: sha256(result.finalText), normalizedSha: sha256(normalized), inputTokens: result.inputTokens,
        outputTokens: result.outputTokens, chars: result.finalText.length, words: result.finalText.trim() ? result.finalText.trim().split(/\s+/).length : 0,
        latency: result.latencyMs, finishReason: result.finishReason, status: result.httpStatus, attempts: attempt, truncated: result.finishReason === "length",
        residue: residue.templateResidueFound, selfName: selfName.selfNameFound, started: overallStarted.toISOString(), finished: finished.toISOString() };
      const record = { schemaVersion: 1, campaignId: freeze.campaignId, campaignHash: freeze.campaignHash, freezeHash: freeze.freezeHash, profile: profileKey,
        generationJobId: job.generation_job_id, jobKey: job.job_key, promptVariantId: job.prompt_variant_id, promptGroupId: job.prompt_group_id,
        split: job.split, decodeConfigId: job.decode_config_id, request: result.request, rawResponse: result.rawResponse, finalText: result.finalText,
        reasoningText: result.reasoningText, inputTokens: result.inputTokens, outputTokens: result.outputTokens, finishReason: result.finishReason,
        httpStatus: result.httpStatus, latencyMs: result.latencyMs, outputSha256: success.outputSha, normalizedOutputSha256: success.normalizedSha,
        truncated: success.truncated, templateResidue: residue, selfName, startedAt: overallStarted.toISOString(), finishedAt: finished.toISOString() };
      return { rawLine: JSON.stringify(record), attempts, success };
    } catch (caught) {
      const finished = new Date();
      const errorClass = caught instanceof Error ? caught.name : "UnknownError";
      const errorDetail = caught instanceof Error ? caught.message : String(caught);
      attempts.push({ jobKey: job.job_key, attempt, request: JSON.stringify(request), raw: null, status: null, latency: finished.getTime() - attemptStarted.getTime(),
        errorClass, errorDetail: errorDetail.slice(0, 8000), started: attemptStarted.toISOString(), finished: finished.toISOString() });
      if (retry < 2) await new Promise((resolve) => setTimeout(resolve, 500 * 2 ** retry));
    }
  }
  const final = attempts.at(-1)!;
  const record = { schemaVersion: 1, campaignId: freeze.campaignId, campaignHash: freeze.campaignHash, freezeHash: freeze.freezeHash, profile: profileKey,
    generationJobId: job.generation_job_id, jobKey: job.job_key, promptVariantId: job.prompt_variant_id, promptGroupId: job.prompt_group_id,
    decodeConfigId: job.decode_config_id, request, errorClass: final.errorClass, errorDetail: final.errorDetail, attempts: attempts.length,
    startedAt: overallStarted.toISOString(), finishedAt: final.finished };
  return { rawLine: JSON.stringify(record), failureLine: JSON.stringify(record), attempts };
}

async function concurrentMap<T, R>(items: T[], workers: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results = Array<R>(items.length); let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(workers, items.length) }, async () => {
    while (true) { const index = cursor++; if (index >= items.length) return; results[index] = await fn(items[index]!); }
  }));
  return results;
}

async function persist(outcomes: Awaited<ReturnType<typeof runJob>>[]) {
  const attempts = outcomes.flatMap((row) => row.attempts);
  const successes = outcomes.flatMap((row) => row.success ? [row.success] : []);
  const failedKeys = outcomes.filter((row) => !row.success).map((row) => ({ jobKey: JSON.parse(row.rawLine).jobKey as string }));
  const executeJson = (rows: unknown[], query: string) => pool.request().input("rows", sql.NVarChar(sql.MAX), JSON.stringify(rows)).query(query);
  if (attempts.length) await executeJson(attempts, `
    INSERT dbo.generation_attempts(generation_job_id,attempt_number,request_json,raw_response_json,http_status,latency_ms,error_class,error_detail,started_at,finished_at)
    SELECT j.generation_job_id,s.attempt,s.request,s.raw,s.status,s.latency,s.errorClass,s.errorDetail,s.started,s.finished
    FROM OPENJSON(@rows) WITH (jobKey char(64) '$.jobKey', attempt int '$.attempt', request nvarchar(max) '$.request', raw nvarchar(max) '$.raw', status int '$.status', latency float '$.latency', errorClass varchar(100) '$.errorClass', errorDetail nvarchar(max) '$.errorDetail', started datetime2(3) '$.started', finished datetime2(3) '$.finished') s
    JOIN dbo.generation_jobs j ON j.job_key=s.jobKey
    WHERE NOT EXISTS (SELECT 1 FROM dbo.generation_attempts a WHERE a.generation_job_id=j.generation_job_id AND a.attempt_number=s.attempt);`);
  if (successes.length) {
    await executeJson(successes, `
      INSERT dbo.generations(generation_job_id,campaign_id,model_profile_id,prompt_variant_id,decode_config_id,split,sample_index,request_json,raw_response_json,final_text,reasoning_text,output_sha256,normalized_output_sha256,input_token_count,output_token_count,reference_token_count,reasoning_token_count,output_char_count,output_word_count,length_band,latency_ms,finish_reason,http_status,attempt_count,truncated,template_residue_found,self_name_found,started_at,finished_at,error_class,error_detail)
      SELECT j.generation_job_id,j.campaign_id,j.model_profile_id,j.prompt_variant_id,j.decode_config_id,g.split,j.sample_index,s.request,s.raw,s.finalText,s.reasoningText,s.outputSha,s.normalizedSha,s.inputTokens,s.outputTokens,NULL,NULL,s.chars,s.words,NULL,s.latency,s.finishReason,s.status,s.attempts,s.truncated,s.residue,s.selfName,s.started,s.finished,NULL,NULL
      FROM OPENJSON(@rows) WITH (jobKey char(64) '$.jobKey', request nvarchar(max) '$.request', raw nvarchar(max) '$.raw', finalText nvarchar(max) '$.finalText', reasoningText nvarchar(max) '$.reasoningText', outputSha char(64) '$.outputSha', normalizedSha char(64) '$.normalizedSha', inputTokens int '$.inputTokens', outputTokens int '$.outputTokens', chars int '$.chars', words int '$.words', latency float '$.latency', finishReason varchar(80) '$.finishReason', status int '$.status', attempts int '$.attempts', truncated bit '$.truncated', residue bit '$.residue', selfName bit '$.selfName', started datetime2(3) '$.started', finished datetime2(3) '$.finished') s
      JOIN dbo.generation_jobs j ON j.job_key=s.jobKey JOIN dbo.prompt_variants v ON v.prompt_variant_id=j.prompt_variant_id JOIN dbo.prompt_groups g ON g.prompt_group_id=v.prompt_group_id
      WHERE NOT EXISTS (SELECT 1 FROM dbo.generations x WHERE x.generation_job_id=j.generation_job_id);
      UPDATE j SET status='complete',completed_at=SYSUTCDATETIME() FROM dbo.generation_jobs j JOIN OPENJSON(@rows) WITH (jobKey char(64) '$.jobKey') s ON s.jobKey=j.job_key;`);
    const artifactRows = successes.flatMap((row) => {
      const normalized = normalizeText(row.finalText); const masked = maskNames(normalized);
      return [
        { jobKey: row.jobKey, view: "raw-final-v1", hash: sha256(normalized), text: normalized },
        { jobKey: row.jobKey, view: "name-masked-v1", hash: sha256(masked), text: masked },
      ];
    });
    await executeJson(artifactRows, `
      INSERT dbo.text_artifacts(text_view_id,normalized_sha256,artifact_text)
      SELECT s.view,s.hash,MIN(s.text) FROM OPENJSON(@rows) WITH (view varchar(80) '$.view', hash char(64) '$.hash', text nvarchar(max) '$.text') s
      WHERE NOT EXISTS (SELECT 1 FROM dbo.text_artifacts t WHERE t.text_view_id=s.view AND t.normalized_sha256=s.hash)
      GROUP BY s.view,s.hash;
      INSERT dbo.generation_text_artifacts(generation_id,text_artifact_id)
      SELECT DISTINCT g.generation_id,t.text_artifact_id FROM OPENJSON(@rows) WITH (jobKey char(64) '$.jobKey', view varchar(80) '$.view', hash char(64) '$.hash') s
      JOIN dbo.generation_jobs j ON j.job_key=s.jobKey JOIN dbo.generations g ON g.generation_job_id=j.generation_job_id
      JOIN dbo.text_artifacts t ON t.text_view_id=s.view AND t.normalized_sha256=s.hash
      WHERE NOT EXISTS (SELECT 1 FROM dbo.generation_text_artifacts m WHERE m.generation_id=g.generation_id AND m.text_artifact_id=t.text_artifact_id);`);
  }
  if (failedKeys.length) await executeJson(failedKeys, `UPDATE j SET status='failed' FROM dbo.generation_jobs j JOIN OPENJSON(@rows) WITH (jobKey char(64) '$.jobKey') s ON s.jobKey=j.job_key;`);
}

try {
  await pool.request().input("campaign", sql.BigInt, freeze.campaignId).query("UPDATE dbo.campaigns SET status='running' WHERE campaign_id=@campaign AND status='frozen';");
  const jobs = await pool.request().input("campaign", sql.BigInt, freeze.campaignId).input("profile", sql.VarChar(80), profileKey).input("limit", sql.Int, limit).query<JobRow>(`
    SELECT TOP (@limit) j.generation_job_id,j.job_key,j.prompt_variant_id,v.rendered_text,v.render_sha256,v.prompt_group_id,g.split,j.decode_config_id,d.config_json,
      (SELECT COUNT(*) FROM dbo.generation_attempts a WHERE a.generation_job_id=j.generation_job_id) AS attempts
    FROM dbo.generation_jobs j JOIN dbo.prompt_variants v ON v.prompt_variant_id=j.prompt_variant_id JOIN dbo.prompt_groups g ON g.prompt_group_id=v.prompt_group_id
    JOIN dbo.decode_configs d ON d.decode_config_id=j.decode_config_id
    WHERE j.campaign_id=@campaign AND j.model_profile_id=@profile AND j.status IN ('pending','failed')
    ORDER BY j.generation_job_id;`);
  let completedThisInvocation = 0; let failedThisInvocation = 0;
  for (let offset = 0; offset < jobs.recordset.length; offset += checkpointSize) {
    const chunk = jobs.recordset.slice(offset, offset + checkpointSize);
    const outcomes = await concurrentMap(chunk, concurrency, runJob);
    await appendFile(rawPath, `${outcomes.map((row) => row.rawLine).join("\n")}\n`);
    const failures = outcomes.flatMap((row) => row.failureLine ? [row.failureLine] : []);
    if (failures.length) await appendFile(failurePath, `${failures.join("\n")}\n`);
    await persist(outcomes);
    completedThisInvocation += outcomes.filter((row) => row.success).length; failedThisInvocation += failures.length;
    const status = await pool.request().input("campaign", sql.BigInt, freeze.campaignId).input("profile", sql.VarChar(80), profileKey).query<{ status: string; count: number }>(`
      SELECT status,COUNT(*) AS count FROM dbo.generation_jobs WHERE campaign_id=@campaign AND model_profile_id=@profile GROUP BY status;`);
    const checkpoint = { schemaVersion: 1, profile: profileKey, campaignId: freeze.campaignId, campaignHash: freeze.campaignHash,
      updatedAt: new Date().toISOString(), invocation: { selected: jobs.recordset.length, completed: completedThisInvocation, failed: failedThisInvocation },
      database: Object.fromEntries(status.recordset.map((row) => [row.status, row.count])), nextOffset: offset + chunk.length };
    await writeFile(checkpointPath, `${JSON.stringify({ ...checkpoint, checkpointHash: hashJson(checkpoint) }, null, 2)}\n`);
    console.log(JSON.stringify(checkpoint));
  }
  await pool.request().query(`UPDATE t SET multi_source_exact_duplicate=1 FROM dbo.text_artifacts t WHERE EXISTS (
    SELECT 1 FROM dbo.generation_text_artifacts m JOIN dbo.generations g ON g.generation_id=m.generation_id
    WHERE m.text_artifact_id=t.text_artifact_id GROUP BY m.text_artifact_id HAVING COUNT(DISTINCT g.model_profile_id)>1);`);
  const completeness = await pool.request().input("campaign", sql.BigInt, freeze.campaignId).input("profile", sql.VarChar(80), profileKey).query<{
    decode_config_id: string; status: string; count: number;
  }>(`SELECT decode_config_id,status,COUNT(*) AS count FROM dbo.generation_jobs WHERE campaign_id=@campaign AND model_profile_id=@profile GROUP BY decode_config_id,status ORDER BY decode_config_id,status;`);
  const summary = { schemaVersion: 1, profile: profileKey, campaignId: freeze.campaignId, campaignHash: freeze.campaignHash, startedAt: startedAt.toISOString(),
    finishedAt: new Date().toISOString(), selectedThisInvocation: jobs.recordset.length, completedThisInvocation, failedThisInvocation,
    cells: completeness.recordset, rawPath, rawSha256: await hashFile(rawPath) };
  await writeFile(`${runDirectory}/manifests/generation-${profileKey}.json`, `${JSON.stringify({ ...summary, manifestHash: hashJson(summary) }, null, 2)}\n`);
  await appendExperimentLog(`MP-4 generation invocation completed for ${profileKey}; selected=${jobs.recordset.length}; completed=${completedThisInvocation}; failed=${failedThisInvocation}.`);
  console.log(JSON.stringify(summary, null, 2));
  if (failedThisInvocation) process.exitCode = 2;
} finally { await pool.close(); }
