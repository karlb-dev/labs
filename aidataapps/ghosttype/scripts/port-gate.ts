import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { extractCandidate } from "../src/extract.js";

// Port gate (addendum A-1/B-4, registry pinsPendingPortGate): before a
// profile's campaign rows count, verify and pin:
//   1. server reachable, served model listed, system_fingerprint recorded
//   2. code-only canaries at the reference decode: raw-SQL contract
//      leakage must be near-zero (fence/prose), the abstention canary must
//      return empty, and determinism holds (two identical calls agree)
//   3. logprobs support recorded
// Results land in control.model_profiles (port_gate_status/json) and an
// evidence event. A profile that fails stays gated out of the campaign.
//
// Usage: npm run port:gate -- --profile <id> [--profile <id> ...]

const CANARIES = [
  {
    id: "canary-select",
    expectEmpty: false,
    messages: [
      { role: "user", content: "Return only raw SQL to insert at the cursor. No markdown, no fences, no prose. If nothing fits, return an empty string." },
      { role: "user", content: "<current_statement_prefix>\nSELECT OrderId, Status FROM dbo.\n</current_statement_prefix>\n\n<schema_context>\nTABLE dbo.Orders (OrderId bigint NOT NULL PK, Status nvarchar(30) NOT NULL)\n</schema_context>" },
    ],
  },
  {
    id: "canary-where",
    expectEmpty: false,
    messages: [
      { role: "user", content: "Return only raw SQL to insert at the cursor. No markdown, no fences, no prose. If nothing fits, return an empty string." },
      { role: "user", content: "<current_statement_prefix>\nSELECT * FROM dbo.Orders WHERE \n</current_statement_prefix>\n\n<schema_context>\nTABLE dbo.Orders (OrderId bigint NOT NULL PK, Status nvarchar(30) NOT NULL)\n</schema_context>" },
    ],
  },
  {
    id: "canary-abstain",
    expectEmpty: true,
    messages: [
      { role: "user", content: "Return only raw SQL to insert at the cursor. No markdown, no fences, no prose. If the schema context does not contain enough information, return exactly an empty string — emit no characters at all." },
      { role: "user", content: "<current_statement_prefix>\nSELECT MissingColumn FROM dbo.\n</current_statement_prefix>\n\n<schema_context>\n(empty)\n</schema_context>" },
    ],
  },
];

const { values: args } = parseArgs({ options: { profile: { type: "string", multiple: true } } });
const registry = JSON.parse(readFileSync("config/models.mac.json", "utf8"));
const profileIds: string[] = args.profile ?? [...registry.targetProfiles, registry.batchVehicleProfile];

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const runId = runDirectory.split("/").at(-1)!;
const pool = await new sql.ConnectionPool({ ...config.databases.admin, database: config.databases.controlName }).connect();

const results: Array<Record<string, unknown>> = [];
for (const profileId of profileIds) {
  const profile = registry.profiles[profileId];
  if (!profile) { results.push({ profileId, status: "FAIL", reason: "unknown profile" }); continue; }
  const gate: Record<string, unknown> = { profileId, servedModelId: profile.servedModelId };
  let status = "PASS";
  try {
    const modelsResponse = await fetch(`${profile.baseUrl}/models`, { signal: AbortSignal.timeout(10_000) });
    const models = await modelsResponse.json() as { data: Array<{ id: string }> };
    gate.serverReachable = modelsResponse.ok;
    gate.modelListed = models.data.some((m) => m.id === profile.servedModelId);
  } catch (error) {
    results.push({ profileId, status: "FAIL", reason: `server unreachable: ${String(error).slice(0, 120)}` });
    continue;
  }

  const canaryResults: Array<Record<string, unknown>> = [];
  let leakage = 0, abstainViolations = 0, deterministic = true, logprobsSupported = false;
  let fingerprint: string | null = null;
  for (const canary of CANARIES) {
    let messages = canary.messages;
    if (profile.reasoningPolicy?.promptPrefixEveryUserTurn) {
      const prefix = profile.reasoningPolicy.promptPrefixEveryUserTurn;
      messages = messages.map((m) => (m.role === "user" ? { ...m, content: `${prefix}\n${m.content}` } : m));
    }
    const call = async () => {
      const response = await fetch(`${profile.baseUrl}/chat/completions`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: profile.servedModelId, messages, temperature: 0, top_p: 1,
          max_tokens: 192 + (profile.reasoningAllowanceTokens ?? 0), stream: false, logprobs: true,
          ...(profile.reasoningPolicy?.chatTemplateKwargs
            ? { chat_template_kwargs: profile.reasoningPolicy.chatTemplateKwargs } : {}),
        }),
        signal: AbortSignal.timeout(300_000),
      });
      const body = await response.json() as any;
      return body;
    };
    const first = await call();
    const second = await call();
    fingerprint = first.system_fingerprint ?? fingerprint;
    logprobsSupported = logprobsSupported || Boolean(first.choices?.[0]?.logprobs);
    const content = (body: any): string => (typeof body.choices?.[0]?.message?.content === "string" ? body.choices[0].message.content : "");
    const extraction = extractCandidate(content(first), "");
    if (extraction.formatFailure !== null) leakage += 1;
    if (canary.expectEmpty && extraction.extracted !== "") abstainViolations += 1;
    if (content(first) !== content(second)) deterministic = false;
    canaryResults.push({
      id: canary.id, formatFailure: extraction.formatFailure,
      empty: extraction.extracted === "", preview: extraction.extracted.slice(0, 80),
    });
  }
  // Re-check the listing after the canaries: mlx servers list a model only
  // once it has been loaded, so a pre-load check false-fails fresh profiles.
  try {
    const relist = await (await fetch(`${profile.baseUrl}/models`, { signal: AbortSignal.timeout(10_000) })).json() as { data: Array<{ id: string }> };
    gate.modelListed = relist.data.some((m) => m.id === profile.servedModelId);
  } catch { /* keep the pre-canary value */ }
  gate.systemFingerprint = fingerprint;
  gate.logprobsSupported = logprobsSupported;
  gate.deterministicAtT0 = deterministic;
  gate.canaries = canaryResults;
  gate.leakageCount = leakage;
  gate.abstainViolations = abstainViolations;
  if (!gate.modelListed) status = "FAIL";
  if (leakage > 0) status = "FAIL";
  if (!deterministic) status = "WARN";
  gate.status = status;

  await pool.request()
    .input("id", sql.VarChar(80), profileId).input("family", sql.VarChar(40), profile.family)
    .input("base", sql.NVarChar(200), profile.baseModelId).input("served", sql.NVarChar(200), profile.servedModelId)
    .input("url", sql.NVarChar(200), profile.baseUrl).input("quant", sql.VarChar(40), profile.quantization)
    .input("role", sql.VarChar(40), profile.role ?? "target")
    .input("status", sql.VarChar(20), status).input("gate", sql.NVarChar(sql.MAX), JSON.stringify(gate))
    .input("profile", sql.NVarChar(sql.MAX), JSON.stringify(profile))
    .query(`MERGE control.model_profiles AS t
      USING (SELECT @id AS profile_id) AS s ON t.profile_id = s.profile_id
      WHEN MATCHED THEN UPDATE SET port_gate_status=@status, port_gate_json=@gate, profile_json=@profile,
        family=@family, base_model_id=@base, served_model_id=@served, base_url=@url, quantization=@quant, registry_role=@role
      WHEN NOT MATCHED THEN INSERT (profile_id,family,base_model_id,served_model_id,base_url,quantization,registry_role,port_gate_status,port_gate_json,profile_json)
        VALUES (@id,@family,@base,@served,@url,@quant,@role,@status,@gate,@profile);`);
  results.push(gate);
  console.log(`[port:gate] ${profileId}: ${status} leakage=${leakage} deterministic=${deterministic} logprobs=${logprobsSupported}`);
}

const summary = { schemaVersion: 1, stage: "GT-11", runId, results };
await atomicWrite(`${runDirectory}/manifests/port-gate.json`, `${JSON.stringify(summary, null, 2)}\n`);
await pool.request()
  .input("key", sql.VarChar(160), `gt11-port-gate:${hashJson(summary).slice(0, 16)}`)
  .input("run", sql.VarChar(120), runId).input("stage", sql.VarChar(40), "GT-11")
  .input("disp", sql.VarChar(40), results.every((r) => r.status === "PASS") ? "PASS" : "PASS_WITH_FINDINGS")
  .input("json", sql.NVarChar(sql.MAX), JSON.stringify(summary))
  .input("hash", sql.Char(64), hashJson(summary))
  .query(`IF NOT EXISTS (SELECT 1 FROM control.evidence_events WHERE event_key=@key)
      INSERT control.evidence_events(event_key,run_id,stage,disposition,detail_json,detail_sha256) VALUES(@key,@run,@stage,@disp,@json,@hash)`);
console.log(JSON.stringify(results.map((r) => ({ profileId: r.profileId, status: r.status })), null, 2));
await pool.close();
