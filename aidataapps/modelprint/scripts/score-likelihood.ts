import { appendFile, readFile, writeFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashFile, hashJson } from "../src/hash.js";
import { resolveModelProfile } from "../src/models.js";
import { resolveRunDirectory, valueAfter } from "../src/run.js";
import { sliceAssistantLogprobs, type TokenLogprobCandidate } from "../src/likelihood.js";

interface ScoreJob { generation_id: number; prompt: string; final_text: string; source_model: string; decode_key: string; }
interface ChatScoreResponse { prompt_token_ids?: number[]; prompt_logprobs?: Array<Record<string, TokenLogprobCandidate> | null>; }
interface CompletionScoreResponse { choices?: Array<{ logprobs?: { token_logprobs?: Array<number | null> } }> }

const scorerKey = valueAfter("--scorer") ?? process.env.MODEL_PROFILE;
if (!scorerKey) throw new Error("Pass --scorer");
const scorer = resolveModelProfile(scorerKey);
const runDirectory = resolveRunDirectory();
const freeze = JSON.parse(await readFile(`${runDirectory}/manifests/campaign-freeze.json`, "utf8")) as { campaignId: number; campaignHash: string };
const config = loadConfig(); const rootUrl = config.inference.chatBaseUrl.replace(/\/v1$/, "");
const concurrency = Number.parseInt(valueAfter("--concurrency") ?? "32", 10); const checkpointSize = Number.parseInt(valueAfter("--checkpoint-size") ?? "200", 10);
const includeHv = process.argv.includes("--include-hv"); const limit = Number.parseInt(valueAfter("--limit") ?? "2147483647", 10);
const includeRobustness = process.argv.includes("--include-robustness");
const robustness = includeRobustness
  ? await readFile(`${runDirectory}/manifests/robustness-freeze.json`, "utf8").then((value) => JSON.parse(value) as { campaignId: number }).catch(() => null)
  : null;
if (includeRobustness && !robustness) throw new Error("--include-robustness requires manifests/robustness-freeze.json");
const campaignIds = [Number(freeze.campaignId), ...(robustness ? [Number(robustness.campaignId)] : [])];
if (!campaignIds.every(Number.isSafeInteger)) throw new Error("Invalid campaign ID in likelihood manifests");
const campaignSql = campaignIds.join(",");
const rawPath = `${runDirectory}/raw/likelihood-${scorerKey}.jsonl`;
const pool = await new sql.ConnectionPool({ server: config.database.server, port: config.database.port, user: config.database.user,
  password: config.database.password, database: config.database.database, options: { encrypt: false, trustServerCertificate: true }, requestTimeout: 900_000 }).connect();

async function post<T>(path: string, body: unknown): Promise<T> {
  let detail = "";
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch(`${rootUrl}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: AbortSignal.timeout(900_000) });
      detail = await response.text(); if (!response.ok) throw new Error(`HTTP ${response.status}: ${detail.slice(0, 1000)}`);
      return JSON.parse(detail) as T;
    } catch (error) { detail = error instanceof Error ? error.message : String(error); if (attempt < 3) await new Promise((resolve) => setTimeout(resolve, 500 * attempt)); }
  }
  throw new Error(detail);
}

async function score(job: ScoreJob, prompted: boolean) {
  if (!job.final_text) return { prompted, ll: 0, tokens: 0, chars: 0, bitsPerChar: Number.POSITIVE_INFINITY, detail: { empty: true } };
  if (prompted) {
    const response = await post<ChatScoreResponse>("/v1/chat/completions", { model: scorerKey,
      messages: [{ role: "user", content: job.prompt }, { role: "assistant", content: job.final_text }], temperature: 0, top_p: 1, top_k: 0, min_p: 0,
      repetition_penalty: 1, presence_penalty: 0, frequency_penalty: 0, seed: 0, max_tokens: 1, n: 1, stop: [], prompt_logprobs: 1,
      return_token_ids: true, chat_template_kwargs: scorer.chatTemplateKwargs });
    const tokens = response.prompt_token_ids ?? []; const logs = response.prompt_logprobs ?? [];
    const span = sliceAssistantLogprobs(tokens, logs, job.final_text);
    if (!span) throw new Error(`No decoded assistant span in prompt logprobs for generation ${job.generation_id}`);
    const values = span.values;
    const ll = values.reduce((a, b) => a + b, 0); return { prompted, ll, tokens: values.length, chars: job.final_text.length,
      bitsPerChar: -ll / Math.LN2 / Math.max(job.final_text.length, 1), detail: { prefixTokens: span.firstTokenIndex, fullTokens: tokens.length,
        assistantCharStart: span.assistantCharStart, assistantCharEnd: span.assistantCharEnd,
        slicing: span.alignment === "exact" ? "decoded-character-overlap-v1" : "decoded-character-overlap-v2-byte-fallback" } };
  }
  const response = await post<CompletionScoreResponse>("/v1/completions", { model: scorerKey, prompt: job.final_text, temperature: 0, top_p: 1, top_k: 0,
    min_p: 0, repetition_penalty: 1, presence_penalty: 0, frequency_penalty: 0, seed: 0, max_tokens: 0, echo: true, logprobs: 1, stop: [] });
  const values = response.choices?.[0]?.logprobs?.token_logprobs?.filter((value): value is number => Number.isFinite(value)) ?? [];
  if (!values.length) throw new Error(`No unprompted logprobs for generation ${job.generation_id}`);
  const ll = values.reduce((a, b) => a + b, 0); return { prompted, ll, tokens: values.length, chars: job.final_text.length,
    bitsPerChar: -ll / Math.LN2 / Math.max(job.final_text.length, 1), detail: {} };
}

async function mapConcurrent<T, R>(items: T[], count: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const output = Array<R>(items.length); let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(count, items.length) }, async () => { while (true) { const index = cursor++; if (index >= items.length) return; output[index] = await fn(items[index]!); } }));
  return output;
}

try {
  const models = await fetch(`${rootUrl}/v1/models`).then((response) => response.json()) as { data?: Array<{ id?: string }> };
  if (!models.data?.some((row) => row.id === scorerKey)) throw new Error(`STOP_PORT: ${scorerKey} is not resident`);
  const query = await pool.request().input("scorer", sql.VarChar(80), scorerKey)
    .input("includeHv", sql.Bit, includeHv).input("limit", sql.Int, limit).query<ScoreJob>(`
      SELECT TOP (@limit) g.generation_id,v.rendered_text AS prompt,g.final_text,g.model_profile_id AS source_model,JSON_VALUE(d.config_json,'$.key') AS decode_key
      FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id
      WHERE g.campaign_id IN (${campaignSql}) AND (@includeHv=1 OR JSON_VALUE(d.config_json,'$.key')<>'hv') AND
        ((NOT EXISTS(SELECT 1 FROM dbo.likelihood_scores l WHERE l.generation_id=g.generation_id AND l.scoring_model_profile_id=@scorer AND l.prompted=1)) OR
         (NOT EXISTS(SELECT 1 FROM dbo.likelihood_scores l WHERE l.generation_id=g.generation_id AND l.scoring_model_profile_id=@scorer AND l.prompted=0)))
      ORDER BY g.generation_id;`);
  let done = 0; let failures = 0;
  for (let offset = 0; offset < query.recordset.length; offset += checkpointSize) {
    const jobs = query.recordset.slice(offset, offset + checkpointSize);
    const outcomes = await mapConcurrent(jobs, concurrency, async (job) => {
      const channels = await Promise.allSettled([score(job, true), score(job, false)]);
      const scores = channels.flatMap((value) => value.status === "fulfilled" ? [value.value] : []);
      const errors = channels.flatMap((value) => value.status === "rejected" ? [value.reason instanceof Error ? value.reason.message : String(value.reason)] : []);
      return { job, scores, error: errors.length ? errors.join("; ") : null };
    });
    await appendFile(rawPath, `${outcomes.map((row) => JSON.stringify({ schemaVersion: 1, scorer: scorerKey, ...row, scoredAt: new Date().toISOString() })).join("\n")}\n`);
    const scores = outcomes.flatMap((row) => row.scores.map((score) => ({ generationId: row.job.generation_id, scorer: scorerKey, prompted: score.prompted,
      ll: score.ll, tokens: score.tokens, chars: score.chars, bitsPerChar: score.bitsPerChar, runId: runDirectory.split("/").at(-1) }))).filter((row) => Number.isFinite(row.bitsPerChar));
    if (scores.length) await pool.request().input("rows", sql.NVarChar(sql.MAX), JSON.stringify(scores)).query(`
      INSERT dbo.likelihood_scores(generation_id,scoring_model_profile_id,prompted,ll_sum_nats,tokens_m,chars,bits_per_char,bits_per_ref_token,scoring_run_id)
      SELECT s.generationId,s.scorer,s.prompted,s.ll,s.tokens,s.chars,s.bitsPerChar,NULL,s.runId FROM OPENJSON(@rows) WITH
        (generationId bigint '$.generationId',scorer varchar(80) '$.scorer',prompted bit '$.prompted',ll float '$.ll',tokens int '$.tokens',chars int '$.chars',bitsPerChar float '$.bitsPerChar',runId varchar(120) '$.runId') s
      WHERE NOT EXISTS(SELECT 1 FROM dbo.likelihood_scores l WHERE l.generation_id=s.generationId AND l.scoring_model_profile_id=s.scorer AND l.prompted=s.prompted);`);
    done += outcomes.filter((row) => !row.error).length; failures += outcomes.filter((row) => row.error).length;
    const checkpoint = { scorer: scorerKey, selected: query.recordset.length, done, failures, offset: offset + jobs.length, updatedAt: new Date().toISOString() };
    await writeFile(`${runDirectory}/checkpoints/likelihood-${scorerKey}.json`, `${JSON.stringify({ ...checkpoint, hash: hashJson(checkpoint) }, null, 2)}\n`);
    console.log(JSON.stringify(checkpoint));
  }
  const manifest = { schemaVersion: 1, scorer: scorerKey, campaignIds, selected: query.recordset.length, done, failures,
    includeHv, includeRobustness, rawPath, rawSha256: query.recordset.length ? await hashFile(rawPath) : null, finishedAt: new Date().toISOString() };
  await writeFile(`${runDirectory}/manifests/likelihood-${scorerKey}.json`, `${JSON.stringify({ ...manifest, manifestHash: hashJson(manifest) }, null, 2)}\n`);
  console.log(JSON.stringify(manifest, null, 2)); if (failures) process.exitCode = 2;
} finally { await pool.close(); }
