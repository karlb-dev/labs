import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashFile, hashJson, sha256 } from "../src/hash.js";
import { appendExperimentLog, resolveRunDirectory } from "../src/run.js";
import { decodeCell, type PromptVariant } from "../src/types.js";

const runDirectory = resolveRunDirectory();
const primary = JSON.parse(await readFile(`${runDirectory}/manifests/campaign-freeze.json`, "utf8")) as {
  campaignId: number; campaignHash: string; freezeHash: string; bankHash: string;
  governedFileHashes: Record<string, string>; campaign: { profiles: string[] };
};
for (const [path, expected] of Object.entries(primary.governedFileHashes)) {
  const actual = await hashFile(path);
  if (actual !== expected) throw new Error(`STOP_DATA: governed file drift ${path}: ${expected} -> ${actual}`);
}
const variants = (await readFile("data/prompt-bank/variants.jsonl", "utf8")).split(/\n/).filter(Boolean)
  .map((line) => JSON.parse(line) as PromptVariant)
  .filter((row) => row.evaluationOnly)
  .sort((left, right) => left.promptVariantId.localeCompare(right.promptVariantId));
const counts = Object.fromEntries([...new Set(variants.map((row) => row.carrierId))].sort()
  .map((carrier) => [carrier, variants.filter((row) => row.carrierId === carrier).length]));
if (variants.length !== 501 || counts["persona-v1"] !== 256 || counts["rag-grounded-v1"] !== 60 || counts["direct-v1"] !== 185) {
  throw new Error(`STOP_DATA: expected frozen 501-row robustness bank, received ${JSON.stringify(counts)}`);
}
const decodeRows = [...new Set(variants.map((row) => row.maxTokens))].sort((a, b) => a - b).map((maxTokens) => {
  const config = decodeCell("det", maxTokens); const hash = hashJson(config);
  return { id: `det-${maxTokens}-${hash.slice(0, 8)}`, hash, config };
});
const manifestBase = {
  schemaVersion: 1,
  kind: "evaluation-only-robustness-v1",
  primaryCampaignId: primary.campaignId,
  primaryCampaignHash: primary.campaignHash,
  primaryFreezeHash: primary.freezeHash,
  bankHash: primary.bankHash,
  profiles: primary.campaign.profiles,
  variantIds: variants.map((row) => row.promptVariantId),
  variantCount: variants.length,
  decodeCell: "det",
  carrierCounts: counts,
  preregisteredInPrimaryFreeze: true,
};
const campaignHash = hashJson(manifestBase); const freezeHash = hashJson({ ...manifestBase, campaignHash });
const config = loadConfig().database;
const pool = await new sql.ConnectionPool({ server: config.server, port: config.port, user: config.user, password: config.password,
  database: config.database, options: { encrypt: false, trustServerCertificate: true }, requestTimeout: 900_000 }).connect();
const executeRows = (rows: unknown[], query: string) => pool.request().input("rows", sql.NVarChar(sql.MAX), JSON.stringify(rows)).query(query);
try {
  await executeRows(decodeRows.map((row) => ({ ...row, config: JSON.stringify(row.config) })), `
    INSERT dbo.decode_configs(decode_config_id,config_hash,config_json)
    SELECT s.id,s.hash,s.config FROM OPENJSON(@rows) WITH (id varchar(40) '$.id',hash char(64) '$.hash',config nvarchar(max) '$.config') s
    WHERE NOT EXISTS(SELECT 1 FROM dbo.decode_configs d WHERE d.decode_config_id=s.id);`);
  const existing = await pool.request().input("hash", sql.Char(64), campaignHash).query<{ campaign_id: number }>("SELECT campaign_id FROM dbo.campaigns WHERE campaign_hash=@hash;");
  let campaignId = existing.recordset[0]?.campaign_id;
  if (!campaignId) {
    const main = await pool.request().input("id", sql.BigInt, primary.campaignId).query<{ governing_spec_hash: string; governing_addendum_hash: string }>(
      "SELECT governing_spec_hash,governing_addendum_hash FROM dbo.campaigns WHERE campaign_id=@id;");
    const hashes = main.recordset[0]; if (!hashes) throw new Error("Primary campaign is absent from SQL");
    const inserted = await pool.request().input("hash", sql.Char(64), campaignHash).input("spec", sql.Char(64), hashes.governing_spec_hash)
      .input("addendum", sql.Char(64), hashes.governing_addendum_hash).input("config", sql.NVarChar(sql.MAX), JSON.stringify(manifestBase))
      .query<{ campaign_id: number }>(`INSERT dbo.campaigns(campaign_name,tier,campaign_hash,governing_spec_hash,governing_addendum_hash,status,config_json)
        OUTPUT inserted.campaign_id VALUES('robustness-v1','dev',@hash,@spec,@addendum,'frozen',@config);`);
    campaignId = inserted.recordset[0]!.campaign_id;
  }
  const manifest = { ...manifestBase, campaignHash, freezeHash, campaignId, expectedJobs: variants.length * primary.campaign.profiles.length };
  await pool.request().input("campaign", sql.BigInt, campaignId).input("freeze", sql.Char(64), freezeHash)
    .input("manifest", sql.NVarChar(sql.MAX), JSON.stringify(manifest)).input("commit", sql.Char(40), execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim())
    .query(`IF NOT EXISTS(SELECT 1 FROM dbo.campaign_freezes WHERE freeze_hash=@freeze)
      INSERT dbo.campaign_freezes(campaign_id,freeze_hash,manifests_json,git_commit) VALUES(@campaign,@freeze,@manifest,@commit);`);
  const decodeByBudget = new Map(decodeRows.map((row) => [row.config.max_tokens, row.id]));
  const jobs = primary.campaign.profiles.flatMap((profile) => variants.map((variant) => ({
    jobKey: sha256(`${campaignHash}:${profile}:${variant.promptVariantId}:${decodeByBudget.get(variant.maxTokens)}:0`),
    campaignId, profile, variant: variant.promptVariantId, decode: decodeByBudget.get(variant.maxTokens), sample: 0,
  })));
  for (let offset = 0; offset < jobs.length; offset += 1000) await executeRows(jobs.slice(offset, offset + 1000), `
    INSERT dbo.generation_jobs(job_key,campaign_id,model_profile_id,prompt_variant_id,decode_config_id,sample_index)
    SELECT s.jobKey,s.campaignId,s.profile,s.variant,s.decode,s.sample FROM OPENJSON(@rows) WITH
      (jobKey char(64) '$.jobKey',campaignId bigint '$.campaignId',profile varchar(80) '$.profile',variant varchar(80) '$.variant',decode varchar(40) '$.decode',sample int '$.sample') s
    WHERE NOT EXISTS(SELECT 1 FROM dbo.generation_jobs j WHERE j.job_key=s.jobKey);`);
  const count = await pool.request().input("campaign", sql.BigInt, campaignId).query<{ count: number }>("SELECT COUNT(*) count FROM dbo.generation_jobs WHERE campaign_id=@campaign;");
  if (count.recordset[0]?.count !== manifest.expectedJobs) throw new Error(`STOP_DATA: robustness jobs ${count.recordset[0]?.count}/${manifest.expectedJobs}`);
  await writeFile(`${runDirectory}/manifests/robustness-freeze.json`, `${JSON.stringify({ ...manifest, frozenAt: new Date().toISOString() }, null, 2)}\n`);
  await appendExperimentLog(`MP-2 robustness companion freeze linked to primary campaign ${primary.campaignId}; campaign=${campaignHash}; jobs=${manifest.expectedJobs}; carrier counts=${JSON.stringify(counts)}.`);
  console.log(JSON.stringify(manifest, null, 2));
} finally { await pool.close(); }
