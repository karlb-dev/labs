import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { scenarioCatalogSchema, scenarioIdentity } from "../src/scenarios.js";

interface RunManifest { runId: string }

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const catalogPath = argument("--catalog") ?? "config/scenarios/smoke.json";
const scheduleName = argument("--schedule") ?? "smoke-v1";
const role = argument("--role") ?? "dev";
const repeats = Number(argument("--repeat-count") ?? "1");
const rateProfile = argument("--rate-profile") ?? "smoke-fast";
const catalogBytes = await readFile(catalogPath, "utf8");
const catalog = scenarioCatalogSchema.parse(JSON.parse(catalogBytes));
const pool = await connect(config.databases.lab, config.databases.controlName);
try {
  for (const scenario of catalog.scenarios) {
    const identity = scenarioIdentity(catalog.catalogId, scenario);
    const existing = await pool.request()
      .input("id", sql.VarChar(100), scenario.id)
      .query<{ scenario_sha256: string }>("SELECT scenario_sha256 FROM workload.scenario_definitions WHERE scenario_id=@id;");
    const recordedHash = existing.recordset[0]?.scenario_sha256;
    if (recordedHash !== undefined && recordedHash !== identity.sha256)
      throw new Error(`Scenario drift for ${scenario.id}: database=${recordedHash} catalog=${identity.sha256}`);
    await pool.request()
      .input("id", sql.VarChar(100), scenario.id)
      .input("group", sql.VarChar(100), scenario.groupId)
      .input("family", sql.VarChar(80), scenario.family)
      .input("regime", sql.Char(1), scenario.regime)
      .input("description", sql.NVarChar(1000), scenario.description)
      .input("injector", sql.NVarChar(128), `driver:${scenario.injector}`)
      .input("class", sql.VarChar(80), scenario.expectedClass)
      .input("severity", sql.VarChar(24), scenario.expectedSeverity)
      .input("abstain", sql.Bit, scenario.shouldAbstain)
      .input("config", sql.NVarChar(sql.MAX), canonicalJson(identity.config))
      .input("truth", sql.NVarChar(sql.MAX), canonicalJson(identity.groundTruth))
      .input("hash", sql.Char(64), identity.sha256)
      .query(`
        IF NOT EXISTS (SELECT 1 FROM workload.scenario_definitions WHERE scenario_id=@id)
          INSERT workload.scenario_definitions
          (scenario_id, scenario_group_id, family, regime, scenario_version, description,
           injector_procedure, driver_id, max_runtime_seconds, safety_class,
           cleanup_procedure, expected_class, expected_severity, should_abstain,
           is_multi_event, is_context_dependent, config_json, ground_truth_json,
           scenario_sha256)
          VALUES(@id,@group,@family,@regime,1,@description,@injector,'typescript-v1',30,
            'benign-disposable','lab.usp_reset_seed',@class,@severity,@abstain,0,0,
            @config,@truth,@hash);
      `);

    const variantId = `${scenario.id}-dev-v1`;
    const variant = {
      schemaVersion: 1,
      scenarioId: scenario.id,
      variantGroupId: scenario.groupId,
      splitRole: role,
      parameters: { injector: scenario.injector, messageVariant: 1 },
      messageViewPolicy: "raw-minimal-v1",
    };
    const variantHash = hashJson(variant);
    const priorVariant = await pool.request()
      .input("id", sql.VarChar(120), variantId)
      .query<{ variant_sha256: string }>("SELECT variant_sha256 FROM workload.scenario_variants WHERE scenario_variant_id=@id;");
    if (priorVariant.recordset[0]?.variant_sha256 !== undefined && priorVariant.recordset[0]!.variant_sha256 !== variantHash)
      throw new Error(`Scenario variant drift for ${variantId}`);
    await pool.request()
      .input("id", sql.VarChar(120), variantId)
      .input("scenario", sql.VarChar(100), scenario.id)
      .input("group", sql.VarChar(120), scenario.groupId)
      .input("role", sql.VarChar(40), role)
      .input("parameters", sql.NVarChar(sql.MAX), canonicalJson(variant.parameters))
      .input("policy", sql.VarChar(40), variant.messageViewPolicy)
      .input("hash", sql.Char(64), variantHash)
      .query(`
        IF NOT EXISTS (SELECT 1 FROM workload.scenario_variants WHERE scenario_variant_id=@id)
          INSERT workload.scenario_variants
            (scenario_variant_id,scenario_id,variant_group_id,split_role,parameter_json,message_view_policy,variant_sha256)
          VALUES(@id,@scenario,@group,@role,@parameters,@policy,@hash);
      `);
    for (const expected of scenario.expectedEvidence) {
      const rule = { schemaVersion: 1, scenarioId: scenario.id, variantId, ...expected };
      await pool.request()
        .input("variant", sql.VarChar(120), variantId)
        .input("source", sql.VarChar(40), expected.source)
        .input("event", sql.VarChar(120), expected.event)
        .input("error", sql.Int, expected.errorNumber ?? null)
        .input("minimum", sql.Int, expected.minimumCount)
        .input("rule", sql.NVarChar(sql.MAX), canonicalJson(rule))
        .query(`
          IF NOT EXISTS
          (
            SELECT 1 FROM workload.expected_evidence
            WHERE scenario_variant_id=@variant AND source_kind=@source
              AND ISNULL(event_name,'')=ISNULL(@event,'')
              AND ISNULL(error_number,-1)=ISNULL(@error,-1)
          )
            INSERT workload.expected_evidence
              (scenario_variant_id,source_kind,event_name,error_number,minimum_count,match_rule_json,required)
            VALUES(@variant,@source,@event,@error,@minimum,@rule,1);
        `);
    }
  }

  const manifest = catalog.scenarios.map((scenario) => ({ id: scenario.id, sha256: scenarioIdentity(catalog.catalogId, scenario).sha256 }));
  const manifestHash = hashJson(manifest);
  const campaign = await pool.request()
    .input("manifest", sql.Char(64), manifestHash)
    .query<{ campaign_id: number }>(`
      UPDATE control.campaigns SET scenario_manifest_hash=@manifest
      WHERE status='building' AND (scenario_manifest_hash IS NULL OR scenario_manifest_hash=@manifest);
      SELECT TOP (1) campaign_id FROM control.campaigns
      WHERE status='building' AND scenario_manifest_hash=@manifest ORDER BY campaign_id;
    `);
  const campaignId = campaign.recordset[0]?.campaign_id;
  if (campaignId === undefined) throw new Error("No mutable campaign accepted the scenario manifest");
  const schedule = await pool.request()
    .input("campaign_id", sql.BigInt, campaignId)
    .input("schedule_name", sql.VarChar(80), scheduleName)
    .input("seed", sql.BigInt, 0)
    .input("scenario_role_filter", sql.VarChar(40), role)
    .input("repeat_count", sql.Int, repeats)
    .input("rate_profile", sql.VarChar(40), rateProfile)
    .execute("workload.usp_build_schedule");
  const scheduleSets = schedule.recordsets as unknown as Array<Array<Record<string, unknown>>>;
  const scheduleRow = scheduleSets[0]?.[0];
  const scheduleItems = scheduleSets[1]?.length ?? 0;
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    catalogId: catalog.catalogId,
    catalogPath,
    catalogSha256: sha256(catalogBytes),
    manifestHash,
    scenarioCount: catalog.scenarios.length,
    schedule: scheduleRow,
    scheduleItems,
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(`${runDirectory}/capture/scenario-build.json`, `${JSON.stringify(receipt, null, 2)}\n`);
  console.log(JSON.stringify(receipt, null, 2));
} finally {
  await pool.close();
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
