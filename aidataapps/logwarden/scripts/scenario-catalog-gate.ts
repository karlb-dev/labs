import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson, sha256 } from "../src/hash.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";
import { scenarioCatalogManifest, scenarioCatalogSchema, scenarioVariantIdentity, scenarioVariants, validateScenarioCatalog } from "../src/scenarios.js";

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as { runId: string };
const catalogPath = argument("--catalog") ?? "config/scenarios/standard-v1.json";
const scheduleName = argument("--schedule") ?? "standard-v1";
const catalogBytes = await readFile(catalogPath, "utf8");
const catalog = scenarioCatalogSchema.parse(JSON.parse(catalogBytes));
validateScenarioCatalog(catalog);
const manifest = scenarioCatalogManifest(catalog, "dev");
const manifestHash = hashJson(manifest);
const expectedScenarios = new Map(manifest.map((scenario) => [scenario.id, scenario.sha256]));
const expectedVariants = new Map(catalog.scenarios.flatMap((scenario) => scenarioVariants(scenario, "dev")
  .map((variant) => [variant.id, hashJson(scenarioVariantIdentity(catalog.catalogId, scenario, variant))] as const)));

const pool = await connect(config.databases.lab, config.databases.controlName, 30_000);
let evidence: Record<string, unknown>;
try {
  const scenarioRows = await pool.request().input("catalog", sql.VarChar(80), catalog.catalogId)
    .query<{ scenario_id: string; scenario_sha256: string }>(`
      SELECT scenario_id,scenario_sha256 FROM workload.scenario_definitions
      WHERE JSON_VALUE(config_json,'$.catalogId')=@catalog ORDER BY scenario_id;
    `);
  const variantRows = await pool.request().input("catalog", sql.VarChar(80), catalog.catalogId)
    .query<{ scenario_variant_id: string; variant_sha256: string }>(`
      SELECT variant.scenario_variant_id,variant.variant_sha256
      FROM workload.scenario_variants AS variant
      INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
      WHERE JSON_VALUE(scenario.config_json,'$.catalogId')=@catalog ORDER BY variant.scenario_variant_id;
    `);
  const scenarioDrift = scenarioRows.recordset.filter((row) => expectedScenarios.get(row.scenario_id) !== row.scenario_sha256);
  const variantDrift = variantRows.recordset.filter((row) => expectedVariants.get(row.scenario_variant_id) !== row.variant_sha256);
  if (scenarioRows.recordset.length !== expectedScenarios.size || variantRows.recordset.length !== expectedVariants.size || scenarioDrift.length > 0 || variantDrift.length > 0)
    throw new Error(`Catalog hash reconciliation failed: ${JSON.stringify({ scenarioRows: scenarioRows.recordset.length, variantRows: variantRows.recordset.length, scenarioDrift, variantDrift })}`);

  const summary = await pool.request()
    .input("catalog", sql.VarChar(80), catalog.catalogId)
    .input("schedule", sql.VarChar(80), scheduleName)
    .input("run", sql.VarChar(120), run.runId)
    .query<Record<string, number | string | boolean | null>>(`
      DECLARE @schedule_id bigint=(SELECT schedule_id FROM workload.schedules WHERE schedule_name=@schedule);
      SELECT
        (SELECT COUNT(*) FROM workload.scenario_definitions WHERE JSON_VALUE(config_json,'$.catalogId')=@catalog) AS scenario_count,
        (SELECT COUNT(*) FROM workload.scenario_variants AS v INNER JOIN workload.scenario_definitions AS s ON s.scenario_id=v.scenario_id WHERE JSON_VALUE(s.config_json,'$.catalogId')=@catalog) AS variant_count,
        (SELECT COUNT(DISTINCT family) FROM workload.scenario_definitions WHERE JSON_VALUE(config_json,'$.catalogId')=@catalog) AS family_count,
        (SELECT COUNT(DISTINCT regime) FROM workload.scenario_definitions WHERE JSON_VALUE(config_json,'$.catalogId')=@catalog) AS regime_count,
        (SELECT COUNT(*) FROM workload.schedule_items WHERE schedule_id=@schedule_id) AS schedule_items,
        (SELECT COUNT(*) FROM (SELECT v.variant_group_id FROM workload.scenario_variants AS v INNER JOIN workload.scenario_definitions AS s ON s.scenario_id=v.scenario_id WHERE JSON_VALUE(s.config_json,'$.catalogId')=@catalog GROUP BY v.variant_group_id HAVING COUNT(DISTINCT v.split_role)>1) AS violations) AS cross_role_groups,
        (SELECT COUNT(*) FROM workload.scenario_definitions WHERE JSON_VALUE(config_json,'$.catalogId')=@catalog AND
          (JSON_VALUE(ground_truth_json,'$.incidentClass')<>expected_class OR JSON_VALUE(ground_truth_json,'$.severity')<>expected_severity OR TRY_CONVERT(bit,JSON_VALUE(ground_truth_json,'$.shouldAbstain'))<>should_abstain)) AS truth_alignment_violations,
        (SELECT COUNT(*) FROM workload.scenario_definitions WHERE JSON_VALUE(config_json,'$.catalogId')=@catalog AND
          (SELECT COUNT(*) FROM OPENJSON(ground_truth_json,'$.requiredTools'))>4) AS tool_budget_violations,
        (SELECT COUNT(*) FROM workload.expected_evidence AS e INNER JOIN workload.scenario_variants AS v ON v.scenario_variant_id=e.scenario_variant_id INNER JOIN workload.scenario_definitions AS s ON s.scenario_id=v.scenario_id WHERE JSON_VALUE(s.config_json,'$.catalogId')=@catalog) AS evidence_rules,
        (SELECT COUNT(*) FROM workload.scenario_definitions AS s CROSS APPLY OPENJSON(s.ground_truth_json,'$.acceptableRunbooks') AS expected LEFT JOIN kb.runbooks AS book ON book.runbook_id=expected.value AND book.corpus_id='primary-v1' WHERE JSON_VALUE(s.config_json,'$.catalogId')=@catalog AND book.runbook_id IS NULL) AS missing_runbooks,
        (SELECT COUNT(*) FROM workload.scenario_definitions WHERE JSON_VALUE(config_json,'$.catalogId')=@catalog AND JSON_QUERY(ground_truth_json,'$.acceptableRunbooks')=N'[]') AS no_answer_templates,
        (SELECT MAX(planned_offset_ms) FROM workload.schedule_items WHERE schedule_id=@schedule_id) AS max_planned_offset_ms,
        (SELECT schedule_hash FROM workload.schedules WHERE schedule_id=@schedule_id) AS schedule_hash,
        (SELECT JSON_VALUE(config_json,'$.catalogId') FROM workload.schedules WHERE schedule_id=@schedule_id) AS schedule_catalog_id,
        (SELECT COUNT(*) FROM OPENJSON((SELECT config_json FROM workload.schedules WHERE schedule_id=@schedule_id),'$.selectedVariants')) AS selected_variant_count,
        (SELECT campaign.scenario_manifest_hash FROM control.campaigns AS campaign INNER JOIN control.runs AS run ON run.campaign_id=campaign.campaign_id WHERE run.run_id=@run) AS campaign_manifest_hash;
    `);
  const summaryRow = summary.recordset[0]!;
  const roleRows = await pool.request().input("schedule", sql.VarChar(80), scheduleName)
    .query<{ split_role: string; episode_count: number }>(`
      SELECT variant.split_role,COUNT(*) AS episode_count
      FROM workload.schedule_items AS item
      INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=item.schedule_id
      INNER JOIN workload.scenario_variants AS variant ON variant.scenario_variant_id=item.scenario_variant_id
      WHERE schedule.schedule_name=@schedule GROUP BY variant.split_role ORDER BY variant.split_role;
    `);
  const familyRows = await pool.request().input("catalog", sql.VarChar(80), catalog.catalogId)
    .query<{ family: string; test_episodes: number }>(`
      SELECT scenario.family,COUNT(*) AS test_episodes
      FROM workload.scenario_variants AS variant
      INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
      WHERE JSON_VALUE(scenario.config_json,'$.catalogId')=@catalog
        AND variant.split_role IN ('test_id','test_variant_holdout')
      GROUP BY scenario.family ORDER BY scenario.family;
    `);
  const gapRows = await pool.request().input("schedule", sql.VarChar(80), scheduleName)
    .query<{ minimum_margin_ms: string | number; group_count: number; invalid_group_sizes: number }>(`
      WITH grouped AS
      (
        SELECT variant.variant_group_id,MIN(item.planned_offset_ms) AS first_offset,
          MAX(item.planned_offset_ms) AS last_offset,COUNT(*) AS episode_count,
          MAX(COALESCE(TRY_CONVERT(int,JSON_VALUE(scenario.config_json,'$.packetWindow.afterSeconds')),90)) AS after_seconds,
          MAX(COALESCE(TRY_CONVERT(int,JSON_VALUE(scenario.config_json,'$.packetWindow.beforeSeconds')),30)) AS before_seconds
        FROM workload.schedule_items AS item
        INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=item.schedule_id
        INNER JOIN workload.scenario_variants AS variant ON variant.scenario_variant_id=item.scenario_variant_id
        INNER JOIN workload.scenario_definitions AS scenario ON scenario.scenario_id=variant.scenario_id
        WHERE schedule.schedule_name=@schedule GROUP BY variant.variant_group_id
      ), ordered AS
      (
        SELECT *,LAG(last_offset) OVER(ORDER BY first_offset) AS prior_last_offset,
          LAG(after_seconds) OVER(ORDER BY first_offset) AS prior_after_seconds
        FROM grouped
      )
      SELECT MIN(first_offset-prior_last_offset-(prior_after_seconds+before_seconds)*1000) AS minimum_margin_ms,
        COUNT(*) AS group_count,SUM(CASE WHEN episode_count<>10 THEN 1 ELSE 0 END) AS invalid_group_sizes
      FROM ordered;
    `);
  const expectedRoles = { calibration: 60, dev: 60, test_id: 300, test_unknown: 60, test_variant_holdout: 120 };
  const actualRoles = Object.fromEntries(roleRows.recordset.map((row) => [row.split_role, Number(row.episode_count)]));
  const gapRow = gapRows.recordset[0]!;
  const failures = {
    scenarioCount: Number(summaryRow.scenario_count) !== 60,
    variantCount: Number(summaryRow.variant_count) !== 600,
    familyCount: Number(summaryRow.family_count) !== 10,
    regimeCount: Number(summaryRow.regime_count) !== 5,
    scheduleItems: Number(summaryRow.schedule_items) !== 600,
    crossRoleGroups: Number(summaryRow.cross_role_groups) !== 0,
    truthAlignment: Number(summaryRow.truth_alignment_violations) !== 0,
    toolBudget: Number(summaryRow.tool_budget_violations) !== 0,
    missingRunbooks: Number(summaryRow.missing_runbooks) !== 0,
    roleAllocation: JSON.stringify(actualRoles) !== JSON.stringify(expectedRoles),
    familyMinimum: familyRows.recordset.length !== 10 || familyRows.recordset.some((row) => Number(row.test_episodes) < 40),
    gapPolicy: Number(gapRow.group_count) !== 60 || Number(gapRow.invalid_group_sizes) !== 0 || Number(gapRow.minimum_margin_ms) < 0,
    scheduleCatalog: summaryRow.schedule_catalog_id !== catalog.catalogId || Number(summaryRow.selected_variant_count) !== 600,
    campaignManifest: summaryRow.campaign_manifest_hash !== manifestHash,
  };
  if (Object.values(failures).some(Boolean)) throw new Error(`Scenario catalog gate failed: ${JSON.stringify({ failures, summaryRow, actualRoles, gapRow })}`);
  evidence = {
    ...summaryRow,
    roles: actualRoles,
    testEpisodesByFamily: Object.fromEntries(familyRows.recordset.map((row) => [row.family, Number(row.test_episodes)])),
    gapPolicy: { groupCount: Number(gapRow.group_count), episodesPerGroup: 10, minimumMarginMs: Number(gapRow.minimum_margin_ms), intentionalWithinGroupOverlap: true },
  };
} finally {
  await pool.close();
}

const receiptBody = {
  schemaVersion: 1,
  runId: run.runId,
  catalogId: catalog.catalogId,
  catalogPath,
  catalogFileSha256: sha256(catalogBytes),
  manifestHash,
  scheduleName,
  evidence,
  injectedEpisodes: 0,
  frozen: false,
  disposition: "PASS",
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/capture/catalog-gate.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({ catalogId: catalog.catalogId, templates: 60, variants: 600, injectedEpisodes: 0, disposition: "PASS", receiptSha256: receipt.receiptSha256 }, null, 2));

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
