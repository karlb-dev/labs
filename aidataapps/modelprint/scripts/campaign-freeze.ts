import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, relative } from "node:path";
import { parse } from "csv-parse/sync";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashFile, hashJson, sha256 } from "../src/hash.js";
import { STYLE_SCHEMA, STYLE_SCHEMA_HASH } from "../src/style.js";
import { decodeCell, type PromptGroup, type PromptVariant } from "../src/types.js";
import { LAB_ROOT, appendExperimentLog, ensureRunLayout, resolveRunDirectory, valueAfter } from "../src/run.js";

type JsonObject = Record<string, unknown>;

const runDirectory = resolveRunDirectory();
await ensureRunLayout(runDirectory);
const configPath = valueAfter("--config") ?? "config/campaigns/full.json";
const absoluteConfigPath = configPath.startsWith("/") ? configPath : `${LAB_ROOT}/${configPath}`;
const campaign = JSON.parse(await readFile(absoluteConfigPath, "utf8")) as {
  schemaVersion: number; name: string; profiles: string[]; decodeCells: Array<"det" | "nat-0" | "nat-1" | "hv">;
  renderedPromptLimit: number; groupLimit: number; seed: number;
};
if (!new Set(["smoke", "dev", "standard", "full"]).has(campaign.name)) throw new Error(`Invalid campaign tier ${campaign.name}`);

const readJson = async <T>(path: string): Promise<T> => JSON.parse(await readFile(`${LAB_ROOT}/${path}`, "utf8")) as T;
const readJsonl = async <T>(path: string): Promise<T[]> => (await readFile(`${LAB_ROOT}/${path}`, "utf8")).split(/\n/).filter(Boolean).map((line) => JSON.parse(line) as T);
const csvRows = async (path: string): Promise<Record<string, string>[]> => parse(await readFile(`${LAB_ROOT}/${path}`, "utf8"), { columns: true, skip_empty_lines: true }) as Record<string, string>[];

const tier = await readJson<{ groupIds: string[]; variantIds: string[]; hash: string }>(`data/manifests/tier-${campaign.name}.json`);
const bank = await readJson<{ groupCount: number; variantCount: number; bankHash: string; tiers: Record<string, { groupIds: string[]; variantIds: string[]; hash: string }> }>("data/manifests/prompt-bank.json");
const sources = await readJson<{ sources: JsonObject[]; bankHash: string }>("data/manifests/prompt-sources.json");
const registry = await readJson<{ snapshotHash: string; sourceCommit: string; profiles: Record<string, JsonObject>; embeddings: Record<string, JsonObject> }>("data/manifests/model-registry-snapshot.json");
const modelConfig = await readJson<{ targetProfiles: string[] }>("config/models.json");
const thresholds = await readJson<JsonObject>("config/thresholds.json");
const evaluationPlan = await readJson<JsonObject>("config/evaluation-plan.json");
const carriers = await readJson<JsonObject>("config/carriers.json");
const groups = await readJsonl<PromptGroup>("data/prompt-bank/groups.jsonl");
const variants = await readJsonl<PromptVariant>("data/prompt-bank/variants.jsonl");
const capabilityPath = `${runDirectory}/environment/sql-server-capabilities.json`;
if (!existsSync(capabilityPath)) throw new Error("STOP_CAPABILITY: run npm run doctor before freezing");
const capability = JSON.parse(await readFile(capabilityPath, "utf8")) as { snapshotHash?: string; syntaxSelection?: string; queryPlanEvidence?: { supported?: boolean; detail?: { indexOperatorPresent?: boolean } } };

// Hard pre-freeze integrity gates.
for (const path of ["data/audits/prompt_group_overlap.csv", "data/audits/canonical_text_duplicates.csv", "data/audits/prompt_name_leakage.csv"]) {
  const rows = await csvRows(path);
  if (rows.length) throw new Error(`STOP_DATA: ${path} has ${rows.length} failing rows`);
}
const groupMap = new Map(groups.map((row) => [row.promptGroupId, row]));
const variantMap = new Map(variants.map((row) => [row.promptVariantId, row]));
const nearPairs = await csvRows("data/audits/near_duplicate_pairs.csv");
if (nearPairs.some((row) => groupMap.get(String(row.left))?.split !== groupMap.get(String(row.right))?.split)) throw new Error("STOP_DATA: a near-duplicate pair crosses splits");
const relationSplits = new Map<string, Set<string>>();
for (const group of groups.filter((row) => row.stratum === "controlled-relation")) {
  const values = relationSplits.get(group.family) ?? new Set<string>(); values.add(group.split); relationSplits.set(group.family, values);
}
if ([...relationSplits.values()].some((values) => values.size !== 1)) throw new Error("STOP_DATA: a relation template family crosses splits");
for (const [small, large] of [["smoke", "dev"], ["dev", "standard"], ["standard", "full"]] as const) {
  const left = new Set(bank.tiers[small]!.groupIds); const right = new Set(bank.tiers[large]!.groupIds);
  if ([...left].some((id) => !right.has(id))) throw new Error(`STOP_DATA: ${small} is not nested in ${large}`);
}
if (tier.groupIds.length !== campaign.groupLimit || tier.variantIds.length !== campaign.renderedPromptLimit) throw new Error("STOP_DATA: campaign limits disagree with the tier manifest");
if (tier.variantIds.some((id) => !variantMap.has(id))) throw new Error("STOP_DATA: tier references an unknown prompt variant");
if (JSON.stringify(campaign.profiles) !== JSON.stringify(modelConfig.targetProfiles)) throw new Error("STOP_DATA: campaign target order differs from the frozen registry");
if (campaign.profiles.some((id) => !registry.profiles[id])) throw new Error("STOP_DATA: target profile is absent from the registry snapshot");
if (!capability.snapshotHash || !["exact", "ann_legacy", "ann_v3"].includes(capability.syntaxSelection ?? "")) throw new Error("STOP_CAPABILITY: SQL capability snapshot has no valid search disposition");
if (capability.syntaxSelection !== "exact" && (!capability.queryPlanEvidence?.supported || !capability.queryPlanEvidence.detail?.indexOperatorPresent)) throw new Error("STOP_CAPABILITY: ANN mode lacks query-plan evidence");

const governedFiles = [
  "docs/SPEC.md", "docs/SPEC_ADDENDUM.md", "config/models.json", "config/carriers.json", "config/thresholds.json",
  "config/evaluation-plan.json", relative(LAB_ROOT, absoluteConfigPath), "data/manifests/model-registry-snapshot.json",
  "data/manifests/prompt-sources.json", "data/manifests/prompt-bank.json", `data/manifests/tier-${campaign.name}.json`,
  "data/audits/SUMMARY.json", "requirements.lock", "package-lock.json",
  "src/types.ts", "src/inference.ts", "src/prompt-bank.ts", "src/style.ts", "src/attribution.ts",
  "scripts/campaign-freeze.ts", "scripts/port-gate.ts", "scripts/generate.ts",
];
const fileHashes = Object.fromEntries(await Promise.all(governedFiles.map(async (path) => [path, await hashFile(`${LAB_ROOT}/${path}`)])));
const capabilityFileHash = await hashFile(capabilityPath);
const scientificManifest = {
  schemaVersion: 1,
  campaign,
  campaignConfigPath: relative(LAB_ROOT, absoluteConfigPath),
  tierManifest: tier,
  bankHash: bank.bankHash,
  sourceBankHash: sources.bankHash,
  modelRegistrySnapshotHash: registry.snapshotHash,
  sqlCapabilitySnapshotHash: capability.snapshotHash,
  sqlCapabilityFileHash: capabilityFileHash,
  styleSchema: STYLE_SCHEMA,
  styleSchemaHash: STYLE_SCHEMA_HASH,
  thresholds,
  evaluationPlan,
  governedFileHashes: fileHashes,
};
const campaignHash = hashJson(scientificManifest);
const freeze = { ...scientificManifest, campaignHash };
const freezeHash = hashJson(freeze);

const selectedVariants = tier.variantIds.map((id) => variantMap.get(id)!);
const tokenBudgets = [...new Set(selectedVariants.map((row) => row.maxTokens))].sort((a, b) => a - b);
const decodeConfigs = campaign.decodeCells.flatMap((cell) => tokenBudgets.map((maxTokens) => {
  const config = decodeCell(cell, maxTokens);
  const hash = hashJson(config);
  return { id: `${cell}-${maxTokens}-${hash.slice(0, 8)}`, hash, config };
}));
const decodeMap = new Map(decodeConfigs.map((row) => [`${row.config.key}:${row.config.max_tokens}`, row]));

const dbConfig = loadConfig().database;
const pool = await new sql.ConnectionPool({
  server: dbConfig.server, port: dbConfig.port, user: dbConfig.user, password: dbConfig.password, database: dbConfig.database,
  options: { encrypt: false, trustServerCertificate: true }, requestTimeout: 900_000, pool: { min: 0, max: 10 },
}).connect();
const batches = <T>(rows: T[], size = 400): T[][] => Array.from({ length: Math.ceil(rows.length / size) }, (_, index) => rows.slice(index * size, (index + 1) * size));
const executeJson = async (rows: unknown[], query: string) => pool.request().input("rows", sql.NVarChar(sql.MAX), JSON.stringify(rows)).query(query);
try {
  const sourceRows = sources.sources.map((source) => ({
    id: source.id,
    kind: source.kind,
    uri: source.path ?? source.source ?? JSON.stringify(source.inputs ?? source.repository ?? source.id),
    revision: source.revision ?? registry.sourceCommit,
    sha: source.sha256 ?? source.raw_sha256,
    license: source.license ?? null,
    manifest: JSON.stringify(source),
  }));
  await executeJson(sourceRows, `
    IF EXISTS (SELECT 1 FROM OPENJSON(@rows) WITH (id varchar(80) '$.id', sha char(64) '$.sha') s JOIN dbo.prompt_sources t ON t.prompt_source_id=s.id WHERE t.source_sha256<>s.sha) THROW 51000, 'STOP_DATA: prompt source drift in SQL', 1;
    INSERT dbo.prompt_sources(prompt_source_id,source_kind,source_uri,source_revision,source_sha256,license_id,manifest_json)
    SELECT s.id,s.kind,s.uri,s.revision,s.sha,s.license,s.manifest FROM OPENJSON(@rows) WITH
      (id varchar(80) '$.id', kind varchar(32) '$.kind', uri nvarchar(500) '$.uri', revision varchar(80) '$.revision', sha char(64) '$.sha', license varchar(80) '$.license', manifest nvarchar(max) '$.manifest') s
    WHERE NOT EXISTS (SELECT 1 FROM dbo.prompt_sources t WHERE t.prompt_source_id=s.id);`);

  const profileRows = Object.entries(registry.profiles).map(([id, profile]) => ({ id, family: profile.family, repository: profile.modelId, revision: profile.revision,
    image: profile.vllmImage, imageDigest: String(profile.vllmImage).split("@sha256:")[1], chatTemplateSha: profile.chatTemplateSha256 ?? null,
    profileHash: profile.profileHash, profileJson: JSON.stringify(profile), target: campaign.profiles.includes(id) }));
  await executeJson(profileRows, `
    IF EXISTS (SELECT 1 FROM OPENJSON(@rows) WITH (id varchar(80) '$.id', profileHash char(64) '$.profileHash') s JOIN dbo.model_profiles t ON t.model_profile_id=s.id WHERE t.profile_hash<>s.profileHash) THROW 51000, 'STOP_DATA: model profile drift in SQL', 1;
    INSERT dbo.model_profiles(model_profile_id,family,model_repository,model_revision,image_reference,image_digest,chat_template_sha256,profile_hash,profile_json,is_target)
    SELECT s.id,s.family,s.repository,s.revision,s.image,s.imageDigest,s.chatTemplateSha,s.profileHash,s.profileJson,s.target FROM OPENJSON(@rows) WITH
      (id varchar(80) '$.id', family varchar(40) '$.family', repository nvarchar(240) '$.repository', revision char(40) '$.revision', image nvarchar(300) '$.image', imageDigest char(64) '$.imageDigest', chatTemplateSha char(64) '$.chatTemplateSha', profileHash char(64) '$.profileHash', profileJson nvarchar(max) '$.profileJson', target bit '$.target') s
    WHERE NOT EXISTS (SELECT 1 FROM dbo.model_profiles t WHERE t.model_profile_id=s.id);`);

  const embeddingRows = Object.entries(registry.embeddings).map(([id, profile]) => ({ id, repository: profile.modelId, revision: profile.revision,
    dimensions: profile.dimensions, profileHash: profile.profileHash, profileJson: JSON.stringify(profile) }));
  await executeJson(embeddingRows, `
    IF EXISTS (SELECT 1 FROM OPENJSON(@rows) WITH (id varchar(80) '$.id', profileHash char(64) '$.profileHash') s JOIN dbo.embedding_profiles t ON t.embedding_profile_id=s.id WHERE t.profile_hash<>s.profileHash) THROW 51000, 'STOP_DATA: embedding profile drift in SQL', 1;
    INSERT dbo.embedding_profiles(embedding_profile_id,model_repository,model_revision,dimensions,profile_hash,profile_json)
    SELECT s.id,s.repository,s.revision,s.dimensions,s.profileHash,s.profileJson FROM OPENJSON(@rows) WITH
      (id varchar(80) '$.id', repository nvarchar(240) '$.repository', revision char(40) '$.revision', dimensions int '$.dimensions', profileHash char(64) '$.profileHash', profileJson nvarchar(max) '$.profileJson') s
    WHERE NOT EXISTS (SELECT 1 FROM dbo.embedding_profiles t WHERE t.embedding_profile_id=s.id);`);

  for (const chunk of batches(groups)) await executeJson(chunk.map((row) => ({ id: row.promptGroupId, source: row.sourceId, sourceRow: row.sourceRowId, family: row.family, domain: row.domain,
    stratum: row.stratum, split: row.split, text: row.canonicalText, sha: sha256(row.canonicalText), metadata: JSON.stringify(row.metadata) })), `
    IF EXISTS (SELECT 1 FROM OPENJSON(@rows) WITH (id varchar(120) '$.id', sha char(64) '$.sha') s JOIN dbo.prompt_groups t ON t.prompt_group_id=s.id WHERE t.canonical_sha256<>s.sha) THROW 51000, 'STOP_DATA: prompt group drift in SQL', 1;
    INSERT dbo.prompt_groups(prompt_group_id,prompt_source_id,source_row_id,family,domain,stratum,split,canonical_text,canonical_sha256,metadata_json)
    SELECT s.id,s.source,s.sourceRow,s.family,s.domain,s.stratum,s.split,s.text,s.sha,s.metadata FROM OPENJSON(@rows) WITH
      (id varchar(120) '$.id', source varchar(80) '$.source', sourceRow varchar(160) '$.sourceRow', family varchar(100) '$.family', domain varchar(100) '$.domain', stratum varchar(80) '$.stratum', split varchar(40) '$.split', text nvarchar(max) '$.text', sha char(64) '$.sha', metadata nvarchar(max) '$.metadata') s
    WHERE NOT EXISTS (SELECT 1 FROM dbo.prompt_groups t WHERE t.prompt_group_id=s.id);`);

  for (const chunk of batches(variants)) await executeJson(chunk.map((row) => ({ id: row.promptVariantId, groupId: row.promptGroupId, carrier: row.carrierId, text: row.renderedText,
    sha: row.renderSha256, maxTokens: row.maxTokens, evaluationOnly: row.evaluationOnly, tiers: JSON.stringify(row.tierMembership), metadata: JSON.stringify(row.metadata) })), `
    IF EXISTS (SELECT 1 FROM OPENJSON(@rows) WITH (id varchar(80) '$.id', sha char(64) '$.sha') s JOIN dbo.prompt_variants t ON t.prompt_variant_id=s.id WHERE t.render_sha256<>s.sha) THROW 51000, 'STOP_DATA: prompt variant drift in SQL', 1;
    INSERT dbo.prompt_variants(prompt_variant_id,prompt_group_id,carrier_id,rendered_text,render_sha256,max_tokens,evaluation_only,tier_membership_json,metadata_json)
    SELECT s.id,s.groupId,s.carrier,s.text,s.sha,s.maxTokens,s.evaluationOnly,s.tiers,s.metadata FROM OPENJSON(@rows) WITH
      (id varchar(80) '$.id', groupId varchar(120) '$.groupId', carrier varchar(80) '$.carrier', text nvarchar(max) '$.text', sha char(64) '$.sha', maxTokens int '$.maxTokens', evaluationOnly bit '$.evaluationOnly', tiers nvarchar(200) '$.tiers', metadata nvarchar(max) '$.metadata') s
    WHERE NOT EXISTS (SELECT 1 FROM dbo.prompt_variants t WHERE t.prompt_variant_id=s.id);`);

  await executeJson(decodeConfigs.map((row) => ({ id: row.id, hash: row.hash, config: JSON.stringify(row.config) })), `
    IF EXISTS (SELECT 1 FROM OPENJSON(@rows) WITH (id varchar(40) '$.id', hash char(64) '$.hash') s JOIN dbo.decode_configs t ON t.decode_config_id=s.id WHERE t.config_hash<>s.hash) THROW 51000, 'STOP_DATA: decode config drift in SQL', 1;
    INSERT dbo.decode_configs(decode_config_id,config_hash,config_json)
    SELECT s.id,s.hash,s.config FROM OPENJSON(@rows) WITH (id varchar(40) '$.id', hash char(64) '$.hash', config nvarchar(max) '$.config') s
    WHERE NOT EXISTS (SELECT 1 FROM dbo.decode_configs t WHERE t.decode_config_id=s.id);`);

  const existing = await pool.request().input("hash", sql.Char(64), campaignHash).query<{ campaign_id: number }>("SELECT campaign_id FROM dbo.campaigns WHERE campaign_hash=@hash;");
  let campaignId = existing.recordset[0]?.campaign_id;
  if (!campaignId) {
    const inserted = await pool.request().input("name", sql.VarChar(100), campaign.name).input("tier", sql.VarChar(20), campaign.name)
      .input("hash", sql.Char(64), campaignHash).input("spec", sql.Char(64), fileHashes["docs/SPEC.md"]!).input("addendum", sql.Char(64), fileHashes["docs/SPEC_ADDENDUM.md"]!)
      .input("config", sql.NVarChar(sql.MAX), JSON.stringify(freeze)).query<{ campaign_id: number }>(`
        INSERT dbo.campaigns(campaign_name,tier,campaign_hash,governing_spec_hash,governing_addendum_hash,status,config_json)
        OUTPUT inserted.campaign_id VALUES(@name,@tier,@hash,@spec,@addendum,'frozen',@config);`);
    campaignId = inserted.recordset[0]!.campaign_id;
    await pool.request().input("name", sql.VarChar(100), campaign.name).input("campaignId", sql.BigInt, campaignId)
      .query("UPDATE dbo.campaigns SET status='stopped' WHERE campaign_name=@name AND campaign_id<>@campaignId AND status IN ('building','frozen','running');");
  }
  await pool.request().input("campaignId", sql.BigInt, campaignId).input("freezeHash", sql.Char(64), freezeHash)
    .input("manifest", sql.NVarChar(sql.MAX), JSON.stringify(freeze)).input("gitCommit", sql.Char(40), execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", cwd: LAB_ROOT }).trim())
    .query(`IF NOT EXISTS (SELECT 1 FROM dbo.campaign_freezes WHERE freeze_hash=@freezeHash)
      INSERT dbo.campaign_freezes(campaign_id,freeze_hash,manifests_json,git_commit) VALUES(@campaignId,@freezeHash,@manifest,@gitCommit);`);

  let expectedJobs = 0;
  for (const profile of campaign.profiles) {
    const jobs = selectedVariants.flatMap((variant) => campaign.decodeCells.map((cell) => {
      const decode = decodeMap.get(`${cell}:${variant.maxTokens}`)!;
      return { jobKey: sha256(`${campaignHash}:${profile}:${variant.promptVariantId}:${decode.id}:0`), campaignId, profile, variant: variant.promptVariantId, decode: decode.id, sample: 0 };
    }));
    expectedJobs += jobs.length;
    for (const chunk of batches(jobs, 1000)) await executeJson(chunk, `
      INSERT dbo.generation_jobs(job_key,campaign_id,model_profile_id,prompt_variant_id,decode_config_id,sample_index)
      SELECT s.jobKey,s.campaignId,s.profile,s.variant,s.decode,s.sample FROM OPENJSON(@rows) WITH
        (jobKey char(64) '$.jobKey', campaignId bigint '$.campaignId', profile varchar(80) '$.profile', variant varchar(80) '$.variant', decode varchar(40) '$.decode', sample int '$.sample') s
      WHERE NOT EXISTS (SELECT 1 FROM dbo.generation_jobs t WHERE t.job_key=s.jobKey);`);
  }
  const count = await pool.request().input("id", sql.BigInt, campaignId).query<{ count: number }>("SELECT COUNT(*) AS count FROM dbo.generation_jobs WHERE campaign_id=@id;");
  if (count.recordset[0]?.count !== expectedJobs) throw new Error(`STOP_DATA: expected ${expectedJobs} jobs, found ${count.recordset[0]?.count}`);

  const run = JSON.parse(readFileSync(`${runDirectory}/run.json`, "utf8")) as { runId: string };
  const manifestOutput = { ...freeze, freezeHash, campaignId, expectedJobs, frozenAt: new Date().toISOString() };
  await writeFile(`${runDirectory}/manifests/campaign-freeze.json`, `${JSON.stringify(manifestOutput, null, 2)}\n`);
  for (const path of ["config/models.json", "config/carriers.json", "config/thresholds.json", "config/evaluation-plan.json", "data/manifests/model-registry-snapshot.json", "data/manifests/prompt-sources.json", "data/manifests/prompt-bank.json", `data/manifests/tier-${campaign.name}.json`]) {
    await copyFile(`${LAB_ROOT}/${path}`, `${runDirectory}/manifests/${basename(path)}`);
  }
  await writeFile(`${LAB_ROOT}/MODELPRINT_FREEZE_RECORD.md`, `# ModelPrint Freeze Record\n\n` +
    `Scientific inputs for campaign \`${campaign.name}\` were frozen before any target-model load.\n\n` +
    `- Campaign hash: \`${campaignHash}\`\n- Freeze hash: \`${freezeHash}\`\n- Prompt-bank hash: \`${bank.bankHash}\`\n` +
    `- Model-registry snapshot: \`${registry.snapshotHash}\`\n- SQL capability snapshot: \`${capability.snapshotHash}\` (\`${capability.syntaxSelection}\`)\n` +
    `- Tier: ${tier.groupIds.length} groups / ${tier.variantIds.length} rendered prompts\n- Target profiles: ${campaign.profiles.join(", ")}\n` +
    `- Decode cells: ${campaign.decodeCells.join(", ")}\n- Frozen jobs: ${expectedJobs.toLocaleString("en-US")}\n- SQL campaign ID: ${campaignId}\n` +
    `- Run: \`${run.runId}\`\n\nThe complete machine-readable record is \`${relative(LAB_ROOT, runDirectory)}/manifests/campaign-freeze.json\`. ` +
    `The run directory is intentionally excluded from Git and retained as an immutable execution artifact.\n`);
  await appendExperimentLog(`MP-2 campaign freeze completed for ${run.runId}; campaign=${campaignHash}; freeze=${freezeHash}; jobs=${expectedJobs}.`);
  console.log(JSON.stringify({ campaignId, campaignHash, freezeHash, expectedJobs, runDirectory }, null, 2));
} finally {
  await pool.close();
}
