import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson, sha256 } from "../src/hash.js";
import { VllmGateway } from "../src/inference.js";
import { loadModelRegistry, resolveModelProfile } from "../src/models.js";
import { scanSelfName, scanTemplateResidue } from "../src/prompt-bank.js";
import { appendExperimentLog, resolveRunDirectory, valueAfter } from "../src/run.js";
import { decodeCell, type ChatMessage } from "../src/types.js";

const profileKey = valueAfter("--profile") ?? process.env.MODEL_PROFILE;
if (!profileKey) throw new Error("Pass --profile");
const profile = resolveModelProfile(profileKey, loadModelRegistry());
const runDirectory = resolveRunDirectory();
const config = loadConfig();
const gateway = new VllmGateway(900_000);
const startedAt = new Date().toISOString();
const project = process.env.COMPOSE_PROJECT_NAME ?? "aidataapps-modelprint";
const containerName = process.env.CHAT_CONTAINER_NAME ?? `${project}-chat-${profileKey.replace(/[^a-z0-9_.-]+/gi, "-")}`;

async function waitReady() {
  const deadline = Date.now() + 3_600_000;
  let error: unknown;
  while (Date.now() < deadline) {
    try { await gateway.ready(config.inference.chatBaseUrl); return; } catch (caught) { error = caught; }
    await new Promise((resolve) => setTimeout(resolve, 5_000));
  }
  throw error ?? new Error("Timed out waiting for vLLM");
}

const canaryTexts = [
  "Explain why a metal bicycle frame may feel colder than a wooden bench at the same room temperature. Give a concise physical explanation.",
  "A service has 99.9% monthly availability. State one practical implication and one caveat in four short labeled sections.",
  "Write a TypeScript function that returns the median of a finite numeric array, then show one example.",
  "Using only this context, answer the question. Context: Aster bikes use a 45 N·m crank-bolt torque. Question: What torque should be used?",
];
const decode = decodeCell("det", 160);
const artifact: Record<string, unknown> = { schemaVersion: 1, profile: profileKey, profileHash: hashJson(profile), startedAt, status: "running" };
let disposition = "STOP_PORT";
let failure: unknown;
try {
  await waitReady();
  const modelsResponse = await fetch(`${config.inference.chatBaseUrl}/models`, { signal: AbortSignal.timeout(30_000) });
  const models = await modelsResponse.json() as { data?: Array<{ id?: string }> };
  if (!models.data?.some((row) => row.id === profileKey)) throw new Error(`Served model list does not contain ${profileKey}`);

  const container = JSON.parse(execFileSync("docker", ["inspect", containerName], { encoding: "utf8" }))[0] as {
    Image?: string; Config?: { Image?: string; Labels?: Record<string, string> }; State?: { Status?: string; StartedAt?: string };
  };
  if (container.Config?.Labels?.["ai.labs.model-profile"] !== profileKey) throw new Error("Container profile label mismatch");
  const expectedImageId = (JSON.parse(await readFile("data/manifests/model-registry-snapshot.json", "utf8")) as { profiles: Record<string, { imageId: string }> }).profiles[profileKey]?.imageId;
  if (!expectedImageId || container.Image !== expectedImageId) throw new Error(`Image digest mismatch: ${container.Image} != ${expectedImageId}`);

  const messages = canaryTexts.map((content): ChatMessage[] => [{ role: "user", content }]);
  const samples = [];
  for (const item of messages) {
    const result = await gateway.completeMessages(config.inference.chatBaseUrl, profile, item, decode);
    const tokenized = await gateway.tokenize(config.inference.chatBaseUrl, profile.key, item);
    const residue = scanTemplateResidue(result.finalText);
    const naming = scanSelfName(result.finalText);
    if (!result.finalText) throw new Error("Canary returned an empty final answer");
    if (residue.templateResidueFound) throw new Error(`Template residue in canary: ${residue.residueKinds.join(",")}`);
    if (naming.selfNameFound) throw new Error(`Self-name leakage in canary: ${naming.names.join(",")}`);
    samples.push({ promptSha256: sha256(item[0]!.content), renderTokenSha256: hashJson(tokenized), request: result.request,
      rawResponse: result.rawResponse, finalText: result.finalText, reasoningText: result.reasoningText, finishReason: result.finishReason,
      latencyMs: result.latencyMs, residue, naming });
  }

  const repeatMessage = messages[0]!;
  const sequential = [
    await gateway.completeMessages(config.inference.chatBaseUrl, profile, repeatMessage, decode),
    await gateway.completeMessages(config.inference.chatBaseUrl, profile, repeatMessage, decode),
  ];
  const batched = await Promise.all(Array.from({ length: 4 }, () => gateway.completeMessages(config.inference.chatBaseUrl, profile, repeatMessage, decode)));
  const repeatHashes = [...sequential, ...batched].map((row) => sha256(row.finalText));
  if (new Set(repeatHashes).size !== 1) throw new Error("Deterministic sequential/batched canary outputs differ");

  const logs = execFileSync("docker", ["logs", "--tail", "500", containerName], { encoding: "utf8", maxBuffer: 8 * 1024 * 1024 });
  const samplingLines = logs.split(/\n/).filter((line) => /SamplingParams\(/.test(line));
  const effective = samplingLines.at(-1) ?? null;
  Object.assign(artifact, { effectiveSamplingParamsLine: effective, effectiveSamplingParamsEvidence: effective ? "server-log" : "missing" });
  if (effective) {
    for (const expected of ["n=1", "presence_penalty=0", "frequency_penalty=0", "repetition_penalty=1", "temperature=0", "top_p=1", "top_k=0", "min_p=0", "seed=0", "max_tokens=160"]) {
      if (!effective.includes(expected)) throw new Error(`Effective SamplingParams does not contain ${expected}`);
    }
  } else throw new Error("The server did not emit effective SamplingParams; --enable-log-requests is required");
  const engineArgs = profile.campaignArgs;
  if (!engineArgs.includes("--generation-config") || !engineArgs.includes("vllm")) throw new Error("Server is not frozen to --generation-config vllm");

  const gpu = execFileSync("nvidia-smi", ["--query-gpu=name,driver_version,memory.total,memory.used,memory.free", "--format=csv,noheader"], { encoding: "utf8" }).trim();
  Object.assign(artifact, { status: "PASS", finishedAt: new Date().toISOString(), servedModels: models.data, container, engineArgs,
    deterministicRepeatHashes: repeatHashes, samples, gpu });
  disposition = "PASS";
} catch (caught) {
  failure = caught;
  Object.assign(artifact, { status: "STOP_PORT", finishedAt: new Date().toISOString(), error: caught instanceof Error ? caught.message : String(caught) });
} finally {
  const finalArtifact = { ...artifact, artifactHash: hashJson(artifact) };
  await writeFile(`${runDirectory}/environment/port-gate-${profileKey}.json`, `${JSON.stringify(finalArtifact, null, 2)}\n`);
  const pool = await new sql.ConnectionPool({ server: config.database.server, port: config.database.port, user: config.database.user,
    password: config.database.password, database: config.database.database, options: { encrypt: false, trustServerCertificate: true } }).connect();
  try {
    const campaign = await pool.request().query<{ campaign_id: number }>("SELECT TOP (1) campaign_id FROM dbo.campaigns WHERE status IN ('frozen','running') ORDER BY campaign_id DESC;");
    await pool.request().input("campaign", sql.BigInt, campaign.recordset[0]?.campaign_id ?? null).input("run", sql.VarChar(120), runDirectory.split("/").at(-1)!)
      .input("kind", sql.VarChar(80), `port-gate:${profileKey}`).input("disposition", sql.VarChar(80), disposition)
      .input("detail", sql.NVarChar(sql.MAX), JSON.stringify(finalArtifact)).query(`INSERT dbo.evidence_events(campaign_id,run_id,stage,event_kind,disposition,detail_json)
        VALUES(@campaign,@run,'MP-3',@kind,@disposition,@detail);`);
  } finally { await pool.close(); }
  await appendExperimentLog(`MP-3 port gate ${disposition} for ${profileKey}; artifact=${relativePath(`${runDirectory}/environment/port-gate-${profileKey}.json`)}.`);
}
if (failure) throw failure;
console.log(JSON.stringify({ profile: profileKey, disposition, artifact: `${runDirectory}/environment/port-gate-${profileKey}.json` }, null, 2));

function relativePath(path: string): string { return path.replace(`${process.cwd()}/`, "").replace(`${process.cwd()}`, "."); }
