import { appendFile, readFile, writeFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashFile, hashJson, sha256 } from "../src/hash.js";
import { buildChatRequest, VllmGateway } from "../src/inference.js";
import { resolveModelProfile } from "../src/models.js";
import { maskNames, normalizeText, scanSelfName, scanTemplateResidue } from "../src/prompt-bank.js";
import { appendExperimentLog, resolveRunDirectory, valueAfter } from "../src/run.js";
import type { DecodeConfig } from "../src/types.js";

interface Job {
  generation_job_id: number; job_key: string; prompt_variant_id: string; rendered_text: string; prompt_group_id: string;
  split: string; decode_config_id: string; config_json: string; attempts: number;
}
interface Attempt {
  jobKey: string; attempt: number; request: string; raw: string | null; status: number | null; latency: number;
  errorClass: string | null; errorDetail: string | null; started: string; finished: string;
}
interface Success {
  jobKey: string; request: string; raw: string; finalText: string; reasoningText: string | null; outputSha: string; normalizedSha: string;
  inputTokens: number | null; outputTokens: number | null; chars: number; words: number; latency: number; finishReason: string | null;
  status: number; attempts: number; truncated: boolean; residue: boolean; selfName: boolean; started: string; finished: string;
}

const profileKey = valueAfter("--profile") ?? process.env.MODEL_PROFILE;
if (!profileKey) throw new Error("Pass --profile");
const runDirectory = resolveRunDirectory();
const primary = JSON.parse(await readFile(`${runDirectory}/manifests/campaign-freeze.json`, "utf8")) as {
  campaignId: number; campaignHash: string; governedFileHashes: Record<string, string>;
};
const suite = JSON.parse(await readFile(`${runDirectory}/manifests/robustness-freeze.json`, "utf8")) as {
  campaignId: number; campaignHash: string; freezeHash: string; primaryCampaignHash: string; profiles: string[]; expectedJobs: number;
};
if (suite.primaryCampaignHash !== primary.campaignHash || !suite.profiles.includes(profileKey)) throw new Error("STOP_DATA: robustness freeze does not match the primary campaign");
for (const [path, expected] of Object.entries(primary.governedFileHashes)) {
  if (await hashFile(path) !== expected) throw new Error(`STOP_DATA: governed file drift ${path}`);
}
const gate = JSON.parse(await readFile(`${runDirectory}/environment/port-gate-${profileKey}.json`, "utf8")) as { status?: string; profile?: string };
if (gate.status !== "PASS" || gate.profile !== profileKey) throw new Error(`STOP_PORT: ${profileKey} has no passing port gate`);
const profile = resolveModelProfile(profileKey); const config = loadConfig(); const gateway = new VllmGateway(900_000);
await gateway.ready(config.inference.chatBaseUrl);
const concurrency = Math.min(profile.maxNumSeqs, Number.parseInt(valueAfter("--concurrency") ?? "32", 10));
const checkpointSize = Number.parseInt(valueAfter("--checkpoint-size") ?? "100", 10);
const limit = Number.parseInt(valueAfter("--limit") ?? "2147483647", 10);
if (![concurrency, checkpointSize, limit].every((value) => Number.isInteger(value) && value > 0)) throw new Error("Invalid numeric argument");
const pool = await new sql.ConnectionPool({ server: config.database.server, port: config.database.port, user: config.database.user,
  password: config.database.password, database: config.database.database, options: { encrypt: false, trustServerCertificate: true },
  requestTimeout: 900_000, pool: { min: 0, max: 20 } }).connect();
const rawPath = `${runDirectory}/raw/generations-robustness-${profileKey}.jsonl`;
const checkpointPath = `${runDirectory}/checkpoints/generation-robustness-${profileKey}.json`;

async function concurrentMap<T, R>(items: T[], workers: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const output = Array<R>(items.length); let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(items.length, workers) }, async () => {
    while (true) { const index = cursor++; if (index >= items.length) return; output[index] = await fn(items[index]!); }
  }));
  return output;
}

async function runJob(job: Job): Promise<{ job: Job; attempts: Attempt[]; success?: Success; error?: string }> {
  const decode = JSON.parse(job.config_json) as DecodeConfig; const request = buildChatRequest(profile, [{ role: "user", content: job.rendered_text }], decode);
  const attempts: Attempt[] = []; const overallStarted = new Date();
  for (let retry = 0; retry < 3; retry += 1) {
    const attempt = job.attempts + retry + 1; const started = new Date();
    try {
      const result = await gateway.completeMessages(config.inference.chatBaseUrl, profile, [{ role: "user", content: job.rendered_text }], decode);
      const finished = new Date(); const normalized = normalizeText(result.finalText); const residue = scanTemplateResidue(result.finalText); const selfName = scanSelfName(result.finalText);
      attempts.push({ jobKey: job.job_key, attempt, request: JSON.stringify(result.request), raw: JSON.stringify(result.rawResponse), status: result.httpStatus,
        latency: result.latencyMs, errorClass: null, errorDetail: null, started: started.toISOString(), finished: finished.toISOString() });
      return { job, attempts, success: { jobKey: job.job_key, request: JSON.stringify(result.request), raw: JSON.stringify(result.rawResponse), finalText: result.finalText,
        reasoningText: result.reasoningText, outputSha: sha256(result.finalText), normalizedSha: sha256(normalized), inputTokens: result.inputTokens, outputTokens: result.outputTokens,
        chars: result.finalText.length, words: result.finalText.trim() ? result.finalText.trim().split(/\s+/).length : 0, latency: result.latencyMs,
        finishReason: result.finishReason, status: result.httpStatus, attempts: attempt, truncated: result.finishReason === "length",
        residue: residue.templateResidueFound, selfName: selfName.selfNameFound, started: overallStarted.toISOString(), finished: finished.toISOString() } };
    } catch (caught) {
      const finished = new Date(); const detail = caught instanceof Error ? caught.message : String(caught);
      attempts.push({ jobKey: job.job_key, attempt, request: JSON.stringify(request), raw: null, status: null, latency: finished.getTime() - started.getTime(),
        errorClass: caught instanceof Error ? caught.name : "UnknownError", errorDetail: detail.slice(0, 8000), started: started.toISOString(), finished: finished.toISOString() });
      if (retry < 2) await new Promise((resolve) => setTimeout(resolve, 500 * 2 ** retry));
    }
  }
  return { job, attempts, error: attempts.at(-1)!.errorDetail ?? "unknown error" };
}

async function persist(outcomes: Awaited<ReturnType<typeof runJob>>[]) {
  const attempts = outcomes.flatMap((row) => row.attempts); const successes = outcomes.flatMap((row) => row.success ? [row.success] : []);
  const execute = (rows: unknown[], query: string) => pool.request().input("rows", sql.NVarChar(sql.MAX), JSON.stringify(rows)).query(query);
  await execute(attempts, `INSERT dbo.generation_attempts(generation_job_id,attempt_number,request_json,raw_response_json,http_status,latency_ms,error_class,error_detail,started_at,finished_at)
    SELECT j.generation_job_id,s.attempt,s.request,s.raw,s.status,s.latency,s.errorClass,s.errorDetail,s.started,s.finished FROM OPENJSON(@rows) WITH
    (jobKey char(64) '$.jobKey',attempt int '$.attempt',request nvarchar(max) '$.request',raw nvarchar(max) '$.raw',status int '$.status',latency float '$.latency',errorClass varchar(100) '$.errorClass',errorDetail nvarchar(max) '$.errorDetail',started datetime2(3) '$.started',finished datetime2(3) '$.finished') s
    JOIN dbo.generation_jobs j ON j.job_key=s.jobKey WHERE NOT EXISTS(SELECT 1 FROM dbo.generation_attempts a WHERE a.generation_job_id=j.generation_job_id AND a.attempt_number=s.attempt);`);
  if (successes.length) {
    await execute(successes, `INSERT dbo.generations(generation_job_id,campaign_id,model_profile_id,prompt_variant_id,decode_config_id,split,sample_index,request_json,raw_response_json,final_text,reasoning_text,output_sha256,normalized_output_sha256,input_token_count,output_token_count,reference_token_count,reasoning_token_count,output_char_count,output_word_count,length_band,latency_ms,finish_reason,http_status,attempt_count,truncated,template_residue_found,self_name_found,started_at,finished_at,error_class,error_detail)
      SELECT j.generation_job_id,j.campaign_id,j.model_profile_id,j.prompt_variant_id,j.decode_config_id,g.split,j.sample_index,s.request,s.raw,s.finalText,s.reasoningText,s.outputSha,s.normalizedSha,s.inputTokens,s.outputTokens,NULL,NULL,s.chars,s.words,NULL,s.latency,s.finishReason,s.status,s.attempts,s.truncated,s.residue,s.selfName,s.started,s.finished,NULL,NULL
      FROM OPENJSON(@rows) WITH (jobKey char(64) '$.jobKey',request nvarchar(max) '$.request',raw nvarchar(max) '$.raw',finalText nvarchar(max) '$.finalText',reasoningText nvarchar(max) '$.reasoningText',outputSha char(64) '$.outputSha',normalizedSha char(64) '$.normalizedSha',inputTokens int '$.inputTokens',outputTokens int '$.outputTokens',chars int '$.chars',words int '$.words',latency float '$.latency',finishReason varchar(80) '$.finishReason',status int '$.status',attempts int '$.attempts',truncated bit '$.truncated',residue bit '$.residue',selfName bit '$.selfName',started datetime2(3) '$.started',finished datetime2(3) '$.finished') s
      JOIN dbo.generation_jobs j ON j.job_key=s.jobKey JOIN dbo.prompt_variants v ON v.prompt_variant_id=j.prompt_variant_id JOIN dbo.prompt_groups g ON g.prompt_group_id=v.prompt_group_id
      WHERE NOT EXISTS(SELECT 1 FROM dbo.generations x WHERE x.generation_job_id=j.generation_job_id);
      UPDATE j SET status='complete',completed_at=SYSUTCDATETIME() FROM dbo.generation_jobs j JOIN OPENJSON(@rows) WITH (jobKey char(64) '$.jobKey') s ON s.jobKey=j.job_key;`);
    const artifacts = successes.flatMap((row) => {
      const normalized = normalizeText(row.finalText); return [
        { jobKey: row.jobKey, viewId: "raw-final-v1", hash: sha256(normalized), text: normalized },
        { jobKey: row.jobKey, viewId: "name-masked-v1", hash: sha256(maskNames(normalized)), text: maskNames(normalized) },
      ];
    });
    await execute(artifacts, `INSERT dbo.text_artifacts(text_view_id,normalized_sha256,artifact_text)
      SELECT s.viewId,s.hash,MIN(s.text) FROM OPENJSON(@rows) WITH (viewId varchar(80) '$.viewId',hash char(64) '$.hash',text nvarchar(max) '$.text') s
      WHERE NOT EXISTS(SELECT 1 FROM dbo.text_artifacts t WHERE t.text_view_id=s.viewId AND t.normalized_sha256=s.hash) GROUP BY s.viewId,s.hash;
      INSERT dbo.generation_text_artifacts(generation_id,text_artifact_id)
      SELECT DISTINCT g.generation_id,t.text_artifact_id FROM OPENJSON(@rows) WITH (jobKey char(64) '$.jobKey',viewId varchar(80) '$.viewId',hash char(64) '$.hash') s
      JOIN dbo.generation_jobs j ON j.job_key=s.jobKey JOIN dbo.generations g ON g.generation_job_id=j.generation_job_id
      JOIN dbo.text_artifacts t ON t.text_view_id=s.viewId AND t.normalized_sha256=s.hash
      WHERE NOT EXISTS(SELECT 1 FROM dbo.generation_text_artifacts m WHERE m.generation_id=g.generation_id AND m.text_artifact_id=t.text_artifact_id);`);
  }
  const failures = outcomes.filter((row) => row.error).map((row) => ({ jobKey: row.job.job_key }));
  if (failures.length) await execute(failures, `UPDATE j SET status='failed' FROM dbo.generation_jobs j JOIN OPENJSON(@rows) WITH (jobKey char(64) '$.jobKey') s ON s.jobKey=j.job_key;`);
}

try {
  await pool.request().input("campaign", sql.BigInt, suite.campaignId).query("UPDATE dbo.campaigns SET status='running' WHERE campaign_id=@campaign AND status='frozen';");
  const jobs = await pool.request().input("campaign", sql.BigInt, suite.campaignId).input("profile", sql.VarChar(80), profileKey).input("limit", sql.Int, limit).query<Job>(`
    SELECT TOP (@limit) j.generation_job_id,j.job_key,j.prompt_variant_id,v.rendered_text,v.prompt_group_id,g.split,j.decode_config_id,d.config_json,
      (SELECT COUNT(*) FROM dbo.generation_attempts a WHERE a.generation_job_id=j.generation_job_id) attempts
    FROM dbo.generation_jobs j JOIN dbo.prompt_variants v ON v.prompt_variant_id=j.prompt_variant_id JOIN dbo.prompt_groups g ON g.prompt_group_id=v.prompt_group_id
    JOIN dbo.decode_configs d ON d.decode_config_id=j.decode_config_id WHERE j.campaign_id=@campaign AND j.model_profile_id=@profile AND j.status IN('pending','failed') ORDER BY j.generation_job_id;`);
  let done = 0; let failures = 0;
  for (let offset = 0; offset < jobs.recordset.length; offset += checkpointSize) {
    const chunk = jobs.recordset.slice(offset, offset + checkpointSize); const outcomes = await concurrentMap(chunk, concurrency, runJob);
    await appendFile(rawPath, `${outcomes.map((row) => JSON.stringify({ schemaVersion: 1, campaignId: suite.campaignId, campaignHash: suite.campaignHash,
      profile: profileKey, jobKey: row.job.job_key, promptVariantId: row.job.prompt_variant_id, promptGroupId: row.job.prompt_group_id,
      success: row.success ?? null, error: row.error ?? null, attempts: row.attempts })).join("\n")}\n`);
    await persist(outcomes); done += outcomes.filter((row) => row.success).length; failures += outcomes.filter((row) => row.error).length;
    const checkpoint = { schemaVersion: 1, profile: profileKey, campaignId: suite.campaignId, selected: jobs.recordset.length, done, failures,
      offset: offset + chunk.length, updatedAt: new Date().toISOString() };
    await writeFile(checkpointPath, `${JSON.stringify({ ...checkpoint, checkpointHash: hashJson(checkpoint) }, null, 2)}\n`); console.log(JSON.stringify(checkpoint));
  }
  const summary = { schemaVersion: 1, profile: profileKey, campaignId: suite.campaignId, campaignHash: suite.campaignHash,
    selected: jobs.recordset.length, done, failures, rawPath, rawSha256: jobs.recordset.length ? await hashFile(rawPath) : null, finishedAt: new Date().toISOString() };
  await writeFile(`${runDirectory}/manifests/generation-robustness-${profileKey}.json`, `${JSON.stringify({ ...summary, manifestHash: hashJson(summary) }, null, 2)}\n`);
  await appendExperimentLog(`MP-4 robustness generation completed for ${profileKey}; selected=${jobs.recordset.length}; completed=${done}; failed=${failures}.`);
  console.log(JSON.stringify(summary, null, 2)); if (failures) process.exitCode = 2;
} finally { await pool.close(); }
