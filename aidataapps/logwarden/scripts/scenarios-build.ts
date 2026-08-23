import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { scenarioCatalogManifest, scenarioCatalogSchema, scenarioIdentity, scenarioVariantIdentity, scenarioVariants, validateScenarioCatalog } from "../src/scenarios.js";

interface RunManifest { runId: string }

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const catalogPath = argument("--catalog") ?? "config/scenarios/smoke.json";
const scheduleName = argument("--schedule") ?? "smoke-v1";
const role = argument("--role") ?? "dev";
const scheduleRoleArgument = argument("--schedule-role") ?? role;
const scheduleRole = scheduleRoleArgument === "all" ? null : scheduleRoleArgument;
const replaceBuildingManifest = process.argv.includes("--replace-building-manifest");
const replaceBuildingCatalog = process.argv.includes("--replace-building-catalog");
const developmentOnly = process.argv.includes("--development-only");
const repeats = Number(argument("--repeat-count") ?? "1");
const seed = Number(argument("--seed") ?? "0");
const rateProfile = argument("--rate-profile") ?? "smoke-fast";
if (!Number.isSafeInteger(seed)) throw new Error("--seed must be a safe integer");
const catalogBytes = await readFile(catalogPath, "utf8");
const catalog = scenarioCatalogSchema.parse(JSON.parse(catalogBytes));
validateScenarioCatalog(catalog);
const pool = await connect(config.databases.lab, config.databases.controlName);
let replacedCatalog: Record<string, unknown> | null = null;
try {
  if (replaceBuildingCatalog) {
    const removed = await pool.request()
      .input("run", sql.VarChar(120), run.runId)
      .input("catalog", sql.VarChar(80), catalog.catalogId)
      .query<Record<string, number>>(`
        DECLARE @campaign_id bigint;
        SELECT @campaign_id=campaign.campaign_id
        FROM control.campaigns AS campaign
        INNER JOIN control.runs AS run ON run.campaign_id=campaign.campaign_id
        WHERE run.run_id=@run AND campaign.status='building';
        IF @campaign_id IS NULL THROW 51610, 'Catalog replacement requires the run building campaign', 1;
        IF EXISTS (SELECT 1 FROM control.campaign_freezes WHERE campaign_id=@campaign_id)
          THROW 51611, 'Catalog replacement is forbidden after any campaign freeze', 1;
        IF EXISTS
        (
          SELECT 1 FROM workload.injection_executions AS execution
          INNER JOIN workload.schedule_items AS item ON item.schedule_item_id=execution.schedule_item_id
          INNER JOIN workload.scenario_variants AS variant ON variant.scenario_variant_id=item.scenario_variant_id
          INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
          WHERE JSON_VALUE(scenario.config_json,'$.catalogId')=@catalog
        ) THROW 51612, 'Catalog replacement is forbidden after an injection execution exists', 1;
        IF EXISTS
        (
          SELECT 1 FROM ingest.incident_packets AS packet
          INNER JOIN workload.scenario_variants AS variant ON variant.scenario_variant_id=packet.scenario_variant_id
          INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
          WHERE JSON_VALUE(scenario.config_json,'$.catalogId')=@catalog
        ) THROW 51613, 'Catalog replacement is forbidden after a packet exists', 1;
        DECLARE @schedule_count int,@scenario_count int,@variant_count int,@evidence_count int;
        SELECT @schedule_count=COUNT(DISTINCT schedule.schedule_id)
        FROM workload.schedules AS schedule
        INNER JOIN workload.schedule_items AS item ON item.schedule_id=schedule.schedule_id
        INNER JOIN workload.scenario_variants AS variant ON variant.scenario_variant_id=item.scenario_variant_id
        INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
        WHERE JSON_VALUE(scenario.config_json,'$.catalogId')=@catalog;
        SELECT @scenario_count=COUNT(*) FROM workload.scenario_definitions
          WHERE JSON_VALUE(config_json,'$.catalogId')=@catalog;
        SELECT @variant_count=COUNT(*) FROM workload.scenario_variants AS variant
          INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
          WHERE JSON_VALUE(scenario.config_json,'$.catalogId')=@catalog;
        SELECT @evidence_count=COUNT(*) FROM workload.expected_evidence AS evidence
          INNER JOIN workload.scenario_variants AS variant ON variant.scenario_variant_id=evidence.scenario_variant_id
          INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
          WHERE JSON_VALUE(scenario.config_json,'$.catalogId')=@catalog;
        BEGIN TRANSACTION;
        DELETE item FROM workload.schedule_items AS item
        INNER JOIN workload.scenario_variants AS variant ON variant.scenario_variant_id=item.scenario_variant_id
        INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
        WHERE JSON_VALUE(scenario.config_json,'$.catalogId')=@catalog;
        DELETE schedule FROM workload.schedules AS schedule
        WHERE NOT EXISTS (SELECT 1 FROM workload.schedule_items AS item WHERE item.schedule_id=schedule.schedule_id)
          AND JSON_VALUE(schedule.config_json,'$.catalogId')=@catalog;
        DELETE evidence FROM workload.expected_evidence AS evidence
        INNER JOIN workload.scenario_variants AS variant ON variant.scenario_variant_id=evidence.scenario_variant_id
        INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
        WHERE JSON_VALUE(scenario.config_json,'$.catalogId')=@catalog;
        DELETE variant FROM workload.scenario_variants AS variant
        INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
        WHERE JSON_VALUE(scenario.config_json,'$.catalogId')=@catalog;
        DELETE FROM workload.scenario_definitions WHERE JSON_VALUE(config_json,'$.catalogId')=@catalog;
        COMMIT;
        SELECT @schedule_count AS removed_schedules,@scenario_count AS removed_scenarios,
          @variant_count AS removed_variants,@evidence_count AS removed_evidence_rules;
      `);
    replacedCatalog = removed.recordset[0] ?? null;
  }
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
      .input("runtime", sql.Int, scenario.maxRuntimeSeconds ?? 30)
      .input("multi", sql.Bit, scenario.isMultiEvent ?? false)
      .input("context", sql.Bit, scenario.isContextDependent ?? false)
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
          VALUES(@id,@group,@family,@regime,1,@description,@injector,'typescript-v2',@runtime,
            'benign-disposable','lab.usp_reset_seed',@class,@severity,@abstain,@multi,@context,
            @config,@truth,@hash);
      `);

    for (const variant of scenarioVariants(scenario, role as "dev" | "calibration" | "test_id" | "test_variant_holdout" | "test_unknown" | "test_live_parity" | "test_storm")) {
      const variantIdentity = scenarioVariantIdentity(catalog.catalogId, scenario, variant);
      const variantHash = hashJson(variantIdentity);
      const priorVariant = await pool.request()
        .input("id", sql.VarChar(120), variant.id)
        .query<{ variant_sha256: string }>("SELECT variant_sha256 FROM workload.scenario_variants WHERE scenario_variant_id=@id;");
      if (priorVariant.recordset[0]?.variant_sha256 !== undefined && priorVariant.recordset[0]!.variant_sha256 !== variantHash)
        throw new Error(`Scenario variant drift for ${variant.id}`);
      await pool.request()
        .input("id", sql.VarChar(120), variant.id)
        .input("scenario", sql.VarChar(100), scenario.id)
        .input("group", sql.VarChar(120), scenario.groupId)
        .input("role", sql.VarChar(40), variant.splitRole)
        .input("parameters", sql.NVarChar(sql.MAX), canonicalJson(variant.parameters))
        .input("policy", sql.VarChar(40), variant.messageViewPolicy)
        .input("rate", sql.VarChar(80), variant.rateContextId)
        .input("hash", sql.Char(64), variantHash)
        .query(`
          IF NOT EXISTS (SELECT 1 FROM workload.scenario_variants WHERE scenario_variant_id=@id)
            INSERT workload.scenario_variants
              (scenario_variant_id,scenario_id,variant_group_id,split_role,parameter_json,message_view_policy,rate_context_id,variant_sha256)
            VALUES(@id,@scenario,@group,@role,@parameters,@policy,@rate,@hash);
        `);
      for (const expected of scenario.expectedEvidence) {
        const rule = { schemaVersion: 1, catalogId: catalog.catalogId, scenarioId: scenario.id, variantId: variant.id, ...expected };
        const ruleJson = canonicalJson(rule);
        const priorRule = await pool.request()
          .input("variant", sql.VarChar(120), variant.id)
          .input("source", sql.VarChar(40), expected.source)
          .input("event", sql.VarChar(120), expected.event)
          .input("error", sql.Int, expected.errorNumber ?? null)
          .query<{ match_rule_json: string; minimum_count: number; maximum_count: number | null; required: boolean }>(`
            SELECT match_rule_json,minimum_count,maximum_count,required
            FROM workload.expected_evidence
            WHERE scenario_variant_id=@variant AND source_kind=@source
              AND ISNULL(event_name,'')=ISNULL(@event,'') AND ISNULL(error_number,-1)=ISNULL(@error,-1);
          `);
        if (priorRule.recordset[0] !== undefined &&
            (priorRule.recordset[0].match_rule_json !== ruleJson || Number(priorRule.recordset[0].minimum_count) !== expected.minimumCount ||
             (priorRule.recordset[0].maximum_count === null ? undefined : Number(priorRule.recordset[0].maximum_count)) !== expected.maximumCount ||
             priorRule.recordset[0].required !== (expected.required ?? true)))
          throw new Error(`Expected-evidence drift for ${variant.id}/${expected.source}/${expected.event}`);
      await pool.request()
        .input("variant", sql.VarChar(120), variant.id)
        .input("source", sql.VarChar(40), expected.source)
        .input("event", sql.VarChar(120), expected.event)
        .input("error", sql.Int, expected.errorNumber ?? null)
        .input("minimum", sql.Int, expected.minimumCount)
        .input("maximum", sql.Int, expected.maximumCount ?? null)
        .input("rule", sql.NVarChar(sql.MAX), ruleJson)
        .input("required", sql.Bit, expected.required ?? true)
        .query(`
          IF NOT EXISTS
          (
            SELECT 1 FROM workload.expected_evidence
            WHERE scenario_variant_id=@variant AND source_kind=@source
              AND ISNULL(event_name,'')=ISNULL(@event,'')
              AND ISNULL(error_number,-1)=ISNULL(@error,-1)
          )
            INSERT workload.expected_evidence
              (scenario_variant_id,source_kind,event_name,error_number,minimum_count,maximum_count,match_rule_json,required)
            VALUES(@variant,@source,@event,@error,@minimum,@maximum,@rule,@required);
        `);
      }
    }
  }

  const manifest = scenarioCatalogManifest(catalog, role as "dev" | "calibration" | "test_id" | "test_variant_holdout" | "test_unknown" | "test_live_parity" | "test_storm");
  const manifestHash = hashJson(manifest);
  const campaign = await pool.request()
    .input("run", sql.VarChar(120), run.runId)
    .input("manifest", sql.Char(64), manifestHash)
    .input("replace", sql.Bit, replaceBuildingManifest)
    .input("development", sql.Bit, developmentOnly)
    .query<{ campaign_id: number; prior_manifest_hash: string | null }>(`
      DECLARE @campaign_id bigint,@prior char(64);
      SELECT @campaign_id=campaign.campaign_id,@prior=campaign.scenario_manifest_hash
      FROM control.campaigns AS campaign
      INNER JOIN control.runs AS run ON run.campaign_id=campaign.campaign_id
      WHERE run.run_id=@run AND campaign.status='building';
      IF @campaign_id IS NULL THROW 51600, 'No mutable campaign belongs to this run', 1;
      IF @development=0 AND @prior IS NOT NULL AND @prior<>@manifest AND @replace=0
        THROW 51601, 'Building campaign already has a different scenario manifest; explicit replacement is required', 1;
      IF @development=0 AND @prior<>@manifest AND EXISTS (SELECT 1 FROM control.campaign_freezes WHERE campaign_id=@campaign_id)
        THROW 51602, 'A frozen campaign manifest cannot be replaced', 1;
      IF @development=0 UPDATE control.campaigns SET scenario_manifest_hash=@manifest WHERE campaign_id=@campaign_id;
      SELECT @campaign_id AS campaign_id,@prior AS prior_manifest_hash;
    `);
  const campaignId = campaign.recordset[0]?.campaign_id;
  if (campaignId === undefined) throw new Error("No mutable campaign accepted the scenario manifest");
  const schedule = await pool.request()
    .input("campaign_id", sql.BigInt, campaignId)
    .input("schedule_name", sql.VarChar(80), scheduleName)
    .input("seed", sql.BigInt, seed)
    .input("scenario_role_filter", sql.VarChar(40), scheduleRole)
    .input("catalog_id", sql.VarChar(80), catalog.catalogId)
    .input("repeat_count", sql.Int, repeats)
    .input("rate_profile", sql.VarChar(40), rateProfile)
    .execute("workload.usp_build_schedule");
  const scheduleSets = schedule.recordsets as unknown as Array<Array<Record<string, unknown>>>;
  const scheduleRow = scheduleSets[0]?.[0];
  const scheduleItemRows = scheduleSets[1] ?? [];
  const scheduleItems = scheduleItemRows.length;
  const maxPlannedOffsetMs = scheduleItemRows.reduce((maximum, item) => Math.max(maximum, Number(item.planned_offset_ms ?? 0)), 0);
  const scheduleSummary = scheduleRow === undefined ? null : {
    scheduleId: scheduleRow.schedule_id,
    campaignId: scheduleRow.campaign_id,
    scheduleName: scheduleRow.schedule_name,
    seed: scheduleRow.seed,
    rateProfile: scheduleRow.rate_profile,
    scheduleHash: scheduleRow.schedule_hash,
    status: scheduleRow.status,
    configSha256: sha256(String(scheduleRow.config_json)),
    maxPlannedOffsetMs,
  };
  const receiptBody = {
    schemaVersion: 1,
    runId: run.runId,
    catalogId: catalog.catalogId,
    catalogPath,
    catalogSha256: sha256(catalogBytes),
    manifestHash,
    priorManifestHash: campaign.recordset[0]?.prior_manifest_hash ?? null,
    replacedBuildingManifest: replaceBuildingManifest,
    replacedCatalog,
    developmentOnly,
    attachedManifestToCampaign: !developmentOnly,
    scenarioCount: catalog.scenarios.length,
    variantCount: manifest.reduce((total, scenario) => total + scenario.variants.length, 0),
    scheduleRole,
    schedule: scheduleSummary,
    scheduleItems,
  };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  const receiptName = developmentOnly ? `scenario-build-${safeName(scheduleName)}.json` : "scenario-build.json";
  await atomicWrite(`${runDirectory}/capture/${receiptName}`, `${JSON.stringify(receipt, null, 2)}\n`);
  console.log(JSON.stringify(receipt, null, 2));
} finally {
  await pool.close();
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}

function safeName(value: string): string {
  if (!/^[a-z0-9-]{1,80}$/i.test(value)) throw new Error(`Unsafe schedule name: ${value}`);
  return value.toLowerCase();
}
