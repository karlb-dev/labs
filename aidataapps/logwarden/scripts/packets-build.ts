import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { canonicalJson, hashJson } from "../src/hash.js";
import { buildIncidentPacket, packetLeakageFindings, type PacketSourceEvent } from "../src/packets.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

interface RunManifest { runId: string }
interface EpisodeRow {
  injection_execution_id: string;
  episode_id: string;
  injector_request_json: string;
  scenario_variant_id: string;
  split_role: string;
  scenario_id: string;
  scenario_group_id: string;
  family: string;
  regime: string;
  expected_class: string;
  expected_severity: string;
  should_abstain: boolean;
  ground_truth_json: string;
  campaign_id: string;
}
interface EventRow {
  canonical_event_id: string;
  source_kind: string;
  source_event_name: string;
  occurred_at_utc: Date;
  error_number: number | null;
  severity: number | null;
  state: number | null;
  message_raw: string | null;
  client_app_name: string | null;
  event_fingerprint: string | null;
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const scheduleName = argument("--schedule") ?? "smoke-v1";
const pool = await connect(config.databases.lab, config.databases.controlName);
const episodes = await pool.request()
  .input("run", sql.VarChar(120), run.runId)
  .input("schedule", sql.VarChar(80), scheduleName)
  .query<EpisodeRow>(`
    SELECT x.injection_execution_id,x.episode_id,x.injector_request_json,
           v.scenario_variant_id,v.split_role,s.scenario_id,s.scenario_group_id,
           s.family,s.regime,s.expected_class,s.expected_severity,s.should_abstain,
           s.ground_truth_json,schedule.campaign_id
    FROM workload.injection_executions AS x
    INNER JOIN workload.schedule_items AS item ON item.schedule_item_id=x.schedule_item_id
    INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=item.schedule_id
    INNER JOIN workload.scenario_variants AS v ON v.scenario_variant_id=item.scenario_variant_id
    INNER JOIN workload.scenario_definitions AS s ON s.scenario_id=v.scenario_id
    WHERE x.run_id=@run AND schedule.schedule_name=@schedule
      AND x.verified=1 AND x.cleanup_verified=1
    ORDER BY item.ordinal;
  `);
const results: Array<Record<string, unknown>> = [];
try {
  for (const episode of episodes.recordset) {
    const request = JSON.parse(episode.injector_request_json) as { correlationToken: string };
    const eventRows = await pool.request()
      .input("injection", sql.BigInt, episode.injection_execution_id)
      .query<EventRow>(`
        SELECT c.canonical_event_id,c.source_kind,c.source_event_name,c.occurred_at_utc,
               c.error_number,c.severity,c.state,c.message_raw,c.client_app_name,
               f.event_fingerprint
        FROM ingest.injection_event_links AS link
        INNER JOIN ingest.canonical_events AS c ON c.canonical_event_id=link.canonical_event_id
        LEFT JOIN ingest.event_fingerprints AS f ON f.canonical_event_id=c.canonical_event_id
        WHERE link.injection_execution_id=@injection
        ORDER BY CASE link.match_role WHEN 'anchor' THEN 0 ELSE 1 END,c.occurred_at_utc,c.canonical_event_id;
      `);
    if (eventRows.recordset.length === 0) throw new Error(`Verified episode has no linked events: ${episode.episode_id}`);
    const sourceEvents: PacketSourceEvent[] = eventRows.recordset.map((event) => ({
      canonicalEventId: String(event.canonical_event_id),
      sourceKind: event.source_kind,
      eventName: event.source_event_name,
      occurredAtUtc: event.occurred_at_utc.toISOString(),
      errorNumber: event.error_number,
      severity: event.severity,
      state: event.state,
      message: event.message_raw,
      clientAppName: event.client_app_name,
      fingerprint: event.event_fingerprint,
    }));
    const availableTools = ["get_recent_incident_counts", "runbook_search"];
    const packet = buildIncidentPacket({
      episodeId: episode.episode_id,
      correlationToken: request.correlationToken,
      sourceEvents,
      availableTools,
    });
    const packetJson = canonicalJson(packet);
    const packetSha256 = hashJson(packet);
    const leakage = packetLeakageFindings(packetJson, [
      request.correlationToken,
      request.correlationToken.slice(0, 12),
      episode.scenario_id,
      episode.scenario_group_id,
    ]);
    if (leakage.length > 0) throw new Error(`Packet leakage for ${episode.episode_id}: ${leakage.join(",")}`);
    const sourceManifest = {
      schemaVersion: 1,
      injectionExecutionId: episode.injection_execution_id,
      canonicalEventIds: sourceEvents.map((event) => event.canonicalEventId),
      sourceSetSha256: packet.hashes.sourceSetSha256,
    };
    const toolManifest = { schemaVersion: 1, registryVersion: "tools-v1-building", availableTools };
    const priorPacket = await pool.request()
      .input("episode", sql.VarChar(120), episode.episode_id)
      .query<{ packet_sha256: string }>("SELECT packet_sha256 FROM ingest.incident_packets WHERE episode_id=@episode;");
    if (priorPacket.recordset[0]?.packet_sha256 !== undefined && priorPacket.recordset[0]!.packet_sha256 !== packetSha256)
      throw new Error(`Packet drift for ${episode.episode_id}`);
    await pool.request()
      .input("episode", sql.VarChar(120), episode.episode_id)
      .input("campaign", sql.BigInt, episode.campaign_id)
      .input("variant", sql.VarChar(120), episode.scenario_variant_id)
      .input("role", sql.VarChar(40), episode.split_role)
      .input("json", sql.NVarChar(sql.MAX), packetJson)
      .input("hash", sql.Char(64), packetSha256)
      .input("source", sql.NVarChar(sql.MAX), canonicalJson(sourceManifest))
      .input("tools", sql.NVarChar(sql.MAX), canonicalJson(toolManifest))
      .query(`
        IF NOT EXISTS (SELECT 1 FROM ingest.incident_packets WHERE episode_id=@episode)
          INSERT ingest.incident_packets
            (episode_id,campaign_id,scenario_variant_id,split_role,packet_version,
             packet_json,packet_sha256,source_manifest_json,frozen_tool_manifest_json,
             frozen_at_utc,is_valid)
          VALUES(@episode,@campaign,@variant,@role,'incident-packet-v1',@json,@hash,@source,@tools,SYSUTCDATETIME(),1);
      `);

    const truth = JSON.parse(episode.ground_truth_json) as {
      preferredAction: string; correctActions: string[]; acceptableRunbooks: string[];
      requiredTools: string[]; optionalTools: string[]; forbiddenTools: string[];
    };
    const protectedTruth = { schemaVersion: 1, episodeId: episode.episode_id, scenarioId: episode.scenario_id, ...truth };
    const truthSha256 = hashJson(protectedTruth);
    await pool.request()
      .input("episode", sql.VarChar(120), episode.episode_id)
      .input("variant", sql.VarChar(120), episode.scenario_variant_id)
      .input("group", sql.VarChar(100), episode.scenario_group_id)
      .input("family", sql.VarChar(80), episode.family)
      .input("regime", sql.Char(1), episode.regime)
      .input("role", sql.VarChar(40), episode.split_role)
      .input("class", sql.VarChar(80), episode.expected_class)
      .input("severity", sql.VarChar(24), episode.expected_severity)
      .input("action", sql.VarChar(100), truth.preferredAction)
      .input("actions", sql.NVarChar(sql.MAX), canonicalJson(truth.correctActions))
      .input("abstain", sql.Bit, episode.should_abstain)
      .input("runbooks", sql.NVarChar(sql.MAX), canonicalJson(truth.acceptableRunbooks))
      .input("truth", sql.NVarChar(sql.MAX), canonicalJson(protectedTruth))
      .input("hash", sql.Char(64), truthSha256)
      .query(`
        IF NOT EXISTS (SELECT 1 FROM eval.ground_truth_episodes WHERE episode_id=@episode)
          INSERT eval.ground_truth_episodes
            (episode_id,scenario_variant_id,scenario_group_id,family,regime,split_role,
             expected_class,expected_severity,expected_action,acceptable_actions_json,
             should_abstain,expected_runbooks_json,protected_truth_json,truth_sha256,frozen_at_utc)
          VALUES(@episode,@variant,@group,@family,@regime,@role,@class,@severity,@action,
            @actions,@abstain,@runbooks,@truth,@hash,SYSUTCDATETIME());
      `);
    for (const [expectation, tools] of [["required", truth.requiredTools], ["optional", truth.optionalTools], ["forbidden", truth.forbiddenTools]] as const) {
      for (const tool of tools) await pool.request()
        .input("episode", sql.VarChar(120), episode.episode_id)
        .input("tool", sql.VarChar(80), tool)
        .input("expectation", sql.VarChar(24), expectation)
        .query(`
          IF NOT EXISTS (SELECT 1 FROM eval.tool_expectations WHERE episode_id=@episode AND tool_name=@tool)
            INSERT eval.tool_expectations(episode_id,tool_name,expectation,acceptable_args_json,reason)
            VALUES(@episode,@tool,@expectation,N'{}','frozen scenario ground truth');
        `);
    }

    const correlationKey = `episode:${episode.episode_id}`;
    const incident = await pool.request()
      .input("key", sql.VarChar(160), correlationKey)
      .input("first", sql.DateTime2(7), eventRows.recordset[0]!.occurred_at_utc)
      .input("last", sql.DateTime2(7), eventRows.recordset.at(-1)!.occurred_at_utc)
      .input("count", sql.Int, sourceEvents.length)
      .query<{ incident_id: string }>(`
        IF NOT EXISTS (SELECT 1 FROM ops.incidents WHERE correlation_key=@key)
          INSERT ops.incidents(correlation_key,source_mode,first_event_at_utc,last_event_at_utc,
            current_class,current_severity,status,event_count,decision_count)
          VALUES(@key,'replay',@first,@last,'unclassified','unknown','open',@count,0);
        SELECT incident_id FROM ops.incidents WHERE correlation_key=@key;
      `);
    const incidentId = incident.recordset[0]!.incident_id;
    for (const event of sourceEvents) await pool.request()
      .input("incident", sql.BigInt, incidentId)
      .input("event", sql.BigInt, event.canonicalEventId)
      .query(`
        IF NOT EXISTS (SELECT 1 FROM ops.incident_events WHERE incident_id=@incident AND canonical_event_id=@event)
          INSERT ops.incident_events(incident_id,canonical_event_id) VALUES(@incident,@event);
      `);
    results.push({ episodeId: episode.episode_id, packetSha256, truthSha256, linkedEvents: sourceEvents.length, leakageFindings: leakage.length });
  }
} finally {
  await pool.close();
}

const receiptBody = { schemaVersion: 1, runId: run.runId, scheduleName, packetCount: results.length, packets: results };
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/packets/build.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({ runId: run.runId, scheduleName, packetCount: results.length, receiptSha256: receipt.receiptSha256 }, null, 2));

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
