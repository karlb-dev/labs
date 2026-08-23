import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { parseArgs } from "node:util";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { resolveRunDirectory } from "../src/run.js";
import { recomposeForParse } from "../src/recompose.js";
import { extractCandidate } from "../src/extract.js";
import { scoreCandidate, type EvalContract } from "../src/scoring.js";

// GT-7 replay runner: raw-text transport over the package's frozen chat
// messages, one model profile per invocation, deterministic reference
// decode. Every request is fully terminal (raw response -> extraction ->
// score persisted in one transaction), so the per-request status row in
// completion.requests is the resume cursor: re-running the same command
// skips done rows. --max-minutes gives a clean pause point for the laptop
// shutdown contract.
//
// Usage:
//   npm run quality:run -- --profile gemma-4-e4b-mlx [--arm M1-packaged]
//     [--limit 25] [--max-minutes 20] [--campaign mac-quality]
//
// Decode (frozen for the mac reference cell, recorded per request):
//   temperature 0, top_p 1, no stop sequences (gold statements end with ';'
//   and OpenAI-style stop strings are excluded from output, which would
//   destroy normalized_exact — model EOS + class max_tokens bound decode),
//   max_tokens by completion class: cursor_fragment 192, intent_query 384.

const SCRIPTDOM_DLL = "dotnet/GhostType.ScriptDom/bin/Release/net8.0/GhostType.ScriptDom.dll";
const RECORDS_PATH = "data/ghosttype_dataset_v2/records/ghosttype_sql_completions_v2.jsonl";
const MAX_TOKENS: Record<string, number> = { cursor_fragment: 192, intent_query: 384 };

const { values: args } = parseArgs({
  options: {
    profile: { type: "string" },
    arm: { type: "string", default: "M1-packaged" },
    campaign: { type: "string", default: "mac-quality" },
    limit: { type: "string" },
    "max-minutes": { type: "string" },
  },
});
if (!args.profile) throw new Error("--profile is required (a key of config/models.mac.json profiles)");
const profileId = args.profile;
const arm = args.arm!;
const limit = args.limit ? Number(args.limit) : Infinity;
const deadlineMs = args["max-minutes"] ? Date.now() + Number(args["max-minutes"]) * 60_000 : Infinity;

const registry = JSON.parse(readFileSync("config/models.mac.json", "utf8"));
const profile = registry.profiles[profileId];
if (!profile) throw new Error(`unknown profile ${profileId}`);
const serverKind = profile.baseUrl.includes("8021") ? "mlx_vlm" : "mlx_lm";

// Frozen chat messages come from the vendored package (hash-verified at
// GT-1) so the transport replays exactly what the package shipped.
const messagesByCase = new Map<string, Array<{ role: string; content: string }>>();
for (const line of readFileSync(RECORDS_PATH, "utf8").split("\n")) {
  if (line.trim() === "") continue;
  const record = JSON.parse(line);
  messagesByCase.set(record.id, record.messages);
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const runId = runDirectory.split("/").at(-1)!;

const child = spawn("dotnet", [SCRIPTDOM_DLL], { stdio: ["pipe", "pipe", "inherit"] });
const readers: Array<(value: string) => void> = [];
createInterface({ input: child.stdout }).on("line", (line) => readers.shift()?.(line));
const scriptDom = (payload: Record<string, unknown>): Promise<Record<string, any>> => {
  child.stdin.write(`${JSON.stringify(payload)}\n`);
  return new Promise((resolve) => readers.push((line) => resolve(JSON.parse(line))));
};

const pool = await new sql.ConnectionPool({ ...config.databases.admin, database: config.databases.controlName }).connect();

interface CaseRow {
  case_id: string; completion_class: string; expect_empty: boolean;
  doc_prefix: string; doc_suffix: string; eval_json: string; oracle_json: string;
}
const cases = (await pool.request().query<CaseRow>(
  `SELECT case_id, completion_class, expect_empty, doc_prefix, doc_suffix, eval_json, oracle_json
   FROM dataset.cases ORDER BY case_id`)).recordset;
const goldRows = (await pool.request().query<{ case_id: string; insertion: string }>(
  "SELECT case_id, insertion FROM dataset.gold_candidates ORDER BY case_id, ordinal")).recordset;
const goldByCase = new Map<string, string[]>();
for (const row of goldRows) goldByCase.set(row.case_id, [...(goldByCase.get(row.case_id) ?? []), row.insertion]);

// Seed pending request rows (idempotent) and re-queue prior failures.
const seed = new sql.Transaction(pool);
await seed.begin();
for (const row of cases) {
  const requestId = `${runId}:${row.case_id}:${profileId}:${arm}`.slice(0, 120);
  const decodeConfig = {
    transport: "raw-text", temperature: 0, top_p: 1, stop: null,
    // Reasoning-channel profiles need headroom beyond the content class cap
    // or they truncate mid-thought with empty content (GT-7 dev evidence).
    max_tokens: (MAX_TOKENS[row.completion_class] ?? 384) + (profile.reasoningAllowanceTokens ?? 0),
    servedModelId: profile.servedModelId, baseUrl: profile.baseUrl,
    reasoningPolicy: profile.reasoningPolicy ?? null, arm, campaign: args.campaign,
    decodeLabel: profile.decodeLabel ?? "deterministic-reference",
  };
  const promptSha = createHash("sha256")
    .update(JSON.stringify(messagesByCase.get(row.case_id) ?? []), "utf8").digest("hex");
  await new sql.Request(seed)
    .input("id", sql.VarChar(120), requestId).input("run", sql.VarChar(120), runId)
    .input("campaign", sql.VarChar(80), args.campaign!).input("case", sql.VarChar(80), row.case_id)
    .input("profile", sql.VarChar(80), profileId).input("arm", sql.VarChar(40), arm)
    .input("decode", sql.NVarChar(sql.MAX), JSON.stringify(decodeConfig))
    .input("sha", sql.Char(64), promptSha)
    .query(`IF NOT EXISTS (SELECT 1 FROM completion.requests WHERE request_id=@id)
        INSERT completion.requests(request_id,run_id,campaign,case_id,profile_id,arm,decode_config_json,prompt_sha256)
        VALUES(@id,@run,@campaign,@case,@profile,@arm,@decode,@sha);
      ELSE IF EXISTS (SELECT 1 FROM completion.requests WHERE request_id=@id AND status='failed')
        UPDATE completion.requests SET status='pending' WHERE request_id=@id;`);
}
await seed.commit();

const pending = (await pool.request()
  .input("run", sql.VarChar(120), runId).input("profile", sql.VarChar(80), profileId).input("arm", sql.VarChar(40), arm)
  .query<{ request_id: string; case_id: string; decode_config_json: string }>(
    `SELECT request_id, case_id, decode_config_json FROM completion.requests
     WHERE run_id=@run AND profile_id=@profile AND arm=@arm AND status='pending' ORDER BY case_id`)).recordset;
const doneAlready = cases.length - pending.length;
console.log(`[quality:run] profile=${profileId} arm=${arm} pending=${pending.length} done=${doneAlready}`);

const caseById = new Map(cases.map((row) => [row.case_id, row]));
let processed = 0, succeeded = 0, failed = 0;
const outcomes: Record<string, number> = {};

for (const job of pending) {
  if (processed >= limit) { console.log("[quality:run] --limit reached"); break; }
  if (Date.now() > deadlineMs) { console.log("[quality:run] --max-minutes reached; safe to stop/resume"); break; }
  const row = caseById.get(job.case_id)!;
  const decodeConfig = JSON.parse(job.decode_config_json);
  let messages = messagesByCase.get(job.case_id);
  if (!messages) { failed += 1; continue; }
  if (profile.reasoningPolicy?.promptPrefixEveryUserTurn) {
    const prefix = profile.reasoningPolicy.promptPrefixEveryUserTurn;
    messages = messages.map((message) =>
      message.role === "user" ? { ...message, content: `${prefix}\n${message.content}` } : message);
  }

  const startedAt = Date.now();
  let response: Response | null = null;
  let body = "";
  try {
    response = await fetch(`${profile.baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: profile.servedModelId, messages,
        temperature: 0, top_p: 1, max_tokens: decodeConfig.max_tokens,
        stream: false, logprobs: true,
        ...(profile.reasoningPolicy?.chatTemplateKwargs
          ? { chat_template_kwargs: profile.reasoningPolicy.chatTemplateKwargs } : {}),
      }),
      signal: AbortSignal.timeout(600_000),
    });
    body = await response.text();
  } catch (error) {
    body = String(error);
  }
  const modelMs = Date.now() - startedAt;

  if (!response || !response.ok) {
    failed += 1;
    await pool.request().input("id", sql.VarChar(120), job.request_id)
      .query("UPDATE completion.requests SET status='failed' WHERE request_id=@id");
    console.error(`[quality:run] ${job.case_id}: HTTP ${response?.status ?? "ERR"} ${body.slice(0, 160)}`);
    if (failed >= 5 && succeeded === 0) { console.error("[quality:run] server looks down; stopping"); break; }
    continue;
  }

  const extractStart = Date.now();
  const parsedBody = JSON.parse(body);
  const message = parsedBody.choices?.[0]?.message ?? {};
  const reasoningField = profile.reasoningPolicy?.serverField;
  const reasoningChars = reasoningField && typeof message[reasoningField] === "string" ? message[reasoningField].length : null;
  const rawText: string = typeof message.content === "string" ? message.content : "";
  const extraction = extractCandidate(rawText, row.doc_suffix);

  const flags = JSON.parse(row.oracle_json) as { parse_eligible?: boolean };
  let parseIntroduced: number | null = null;
  if (flags.parse_eligible === true && extraction.extracted !== "" && extraction.formatFailure === null) {
    const joiner = recomposeForParse(row.doc_prefix, extraction.extracted, row.doc_suffix).joiner;
    const delta = await scriptDom({ op: "insertDelta", prefix: row.doc_prefix + joiner, candidate: extraction.extracted, suffix: row.doc_suffix });
    if (delta.ok) parseIntroduced = delta.structuralIntroducedCount;
  }
  const contract = JSON.parse(row.eval_json) as EvalContract;
  const score = scoreCandidate({
    candidate: extraction.formatFailure ? rawText : extraction.extracted,
    contract, acceptedInsertions: goldByCase.get(row.case_id) ?? [],
    docPrefix: row.doc_prefix, docSuffix: row.doc_suffix, parseIntroduced,
  });
  // Truncation is not abstention: an empty content channel after a
  // finish_reason=length decode means the budget died in reasoning.
  const truncated = parsedBody.choices?.[0]?.finish_reason === "length" && extraction.extracted === "";
  const outcome = truncated ? "fail"
    : extraction.formatFailure !== null && !contract.expect_empty ? "format_fail" : score.outcome;
  const failureStage = truncated ? "truncation" : extraction.formatFailure !== null ? "extraction" : null;
  const extractMs = Date.now() - extractStart;

  const usage = parsedBody.usage ?? {};
  const persistStart = Date.now();
  const writer = new sql.Transaction(pool);
  await writer.begin();
  try {
    await new sql.Request(writer)
      .input("id", sql.VarChar(120), job.request_id).input("status", sql.Int, response.status)
      .input("body", sql.NVarChar(sql.MAX), body)
      .input("sha", sql.Char(64), createHash("sha256").update(body, "utf8").digest("hex"))
      .input("latency", sql.Float, modelMs)
      .input("pt", sql.Int, usage.prompt_tokens ?? null).input("ct", sql.Int, usage.completion_tokens ?? null)
      .input("rc", sql.Int, reasoningChars)
      .query(`IF EXISTS (SELECT 1 FROM completion.raw_model_responses WHERE request_id=@id)
                DELETE FROM completion.raw_model_responses WHERE request_id=@id;
              INSERT completion.raw_model_responses(request_id,http_status,raw_body,raw_sha256,latency_ms,prompt_tokens,completion_tokens,reasoning_chars)
              VALUES(@id,@status,@body,@sha,@latency,@pt,@ct,@rc)`);
    await new sql.Request(writer)
      .input("cid", sql.VarChar(140), `${job.request_id}:0`).input("rid", sql.VarChar(120), job.request_id)
      .input("raw", sql.NVarChar(sql.MAX), rawText).input("ext", sql.NVarChar(sql.MAX), extraction.extracted)
      .input("ev", sql.VarChar(40), extraction.extractionVersion)
      .input("trim", sql.Int, extraction.suffixOverlapTrimmed)
      .input("ff", sql.VarChar(40), extraction.formatFailure)
      .input("abst", sql.Bit, extraction.isAbstention ? 1 : 0)
      .query(`IF EXISTS (SELECT 1 FROM completion.candidates WHERE candidate_id=@cid)
                DELETE FROM completion.candidates WHERE candidate_id=@cid;
              INSERT completion.candidates(candidate_id,request_id,ordinal,raw_text,extracted_text,extraction_version,suffix_overlap_trimmed,format_failure,is_abstention)
              VALUES(@cid,@rid,0,@raw,@ext,@ev,@trim,@ff,@abst)`);
    await new sql.Request(writer)
      .input("rid", sql.VarChar(120), job.request_id)
      .input("decision", sql.VarChar(20), extraction.isAbstention ? "abstain" : extraction.formatFailure ? "fail" : "offer")
      .input("cid", sql.VarChar(140), `${job.request_id}:0`)
      .query(`IF EXISTS (SELECT 1 FROM completion.final_decisions WHERE request_id=@rid)
                DELETE FROM completion.final_decisions WHERE request_id=@rid;
              INSERT completion.final_decisions(request_id,decision,selected_candidate_id) VALUES(@rid,@decision,@cid)`);
    await new sql.Request(writer)
      .input("run", sql.VarChar(120), runId).input("case", sql.VarChar(80), row.case_id)
      .input("profile", sql.VarChar(80), profileId).input("arm", sql.VarChar(40), arm)
      .input("outcome", sql.VarChar(30), outcome).input("fs", sql.VarChar(30), failureStage)
      .input("nx", sql.Bit, score.normalizedExact)
      .input("cp", sql.Bit, score.constraintPass).input("gp", sql.Bit, score.groundingPass)
      .input("ep", sql.Bit, score.emptyPolicyPass).input("nm", sql.Bit, score.noMarkdownPass)
      .input("ii", sql.Bit, score.insertionIntegrityPass).input("pp", sql.Bit, score.parsePass)
      .input("lat", sql.Float, modelMs).input("ct", sql.Int, usage.completion_tokens ?? null)
      .input("detail", sql.NVarChar(sql.MAX), JSON.stringify({
        extraction: { ...extraction, extracted: undefined }, reasoningChars,
        candidatePreview: extraction.extracted.slice(0, 200),
      }))
      .query(`DELETE FROM eval.row_scores WHERE run_id=@run AND case_id=@case AND profile_id=@profile AND arm=@arm;
              INSERT eval.row_scores(run_id,case_id,profile_id,arm,eligible,outcome,failure_stage,normalized_exact,constraint_pass,grounding_pass,empty_policy_pass,no_markdown_pass,insertion_integrity_pass,parse_pass,latency_ms,completion_tokens,detail_json)
              VALUES(@run,@case,@profile,@arm,1,@outcome,@fs,@nx,@cp,@gp,@ep,@nm,@ii,@pp,@lat,@ct,@detail)`);
    await new sql.Request(writer)
      .input("rid", sql.VarChar(120), job.request_id).input("profile", sql.VarChar(80), profileId)
      .input("model", sql.Float, modelMs).input("extract", sql.Float, extractMs)
      .input("persist", sql.Float, Date.now() - persistStart)
      .input("total", sql.Float, Date.now() - startedAt)
      .input("pt", sql.Int, usage.prompt_tokens ?? null).input("ct", sql.Int, usage.completion_tokens ?? null)
      .input("tps", sql.Float, usage.completion_tokens ? usage.completion_tokens / (modelMs / 1000) : null)
      .input("kind", sql.VarChar(30), serverKind)
      .query(`IF EXISTS (SELECT 1 FROM telemetry.model_requests WHERE request_id=@rid)
                DELETE FROM telemetry.model_requests WHERE request_id=@rid;
              INSERT telemetry.model_requests(request_id,profile_id,model_ms,extract_ms,persist_ms,total_ms,prompt_tokens,completion_tokens,tokens_per_second,server_kind)
              VALUES(@rid,@profile,@model,@extract,@persist,@total,@pt,@ct,@tps,@kind)`);
    await new sql.Request(writer)
      .input("id", sql.VarChar(120), job.request_id)
      .query("UPDATE completion.requests SET status='done' WHERE request_id=@id");
    await writer.commit();
  } catch (error) {
    await writer.rollback();
    throw error;
  }

  processed += 1; succeeded += 1;
  outcomes[outcome] = (outcomes[outcome] ?? 0) + 1;
  if (processed % 10 === 0 || processed === 1) {
    console.log(`[quality:run] ${processed}/${pending.length} last=${row.case_id} outcome=${outcome} ${Math.round(modelMs)}ms`);
  }
}

child.stdin.end();
const remaining = pending.length - processed;
console.log(JSON.stringify({ profileId, arm, processed, succeeded, failed, remaining, outcomes }, null, 2));
await pool.close();
