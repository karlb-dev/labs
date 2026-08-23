import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { packetLeakageFindings } from "../src/packets.js";
import { connect } from "../src/repository.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

interface RunManifest { runId: string }

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const run = JSON.parse(await readFile(`${runDirectory}/run.json`, "utf8")) as RunManifest;
const scheduleName = argument("--schedule") ?? "smoke-v1";
const pool = await connect(config.databases.lab, config.databases.controlName);
const findings: Array<Record<string, unknown>> = [];
let packetCount = 0;
try {
  const packets = await pool.request()
    .input("run", sql.VarChar(120), run.runId)
    .input("schedule", sql.VarChar(80), scheduleName)
    .query<{ episode_id: string; packet_json: string; packet_sha256: string; scenario_id: string; scenario_group_id: string; injector_request_json: string }>(`
      SELECT p.episode_id,p.packet_json,p.packet_sha256,s.scenario_id,s.scenario_group_id,x.injector_request_json
      FROM ingest.incident_packets AS p
      INNER JOIN workload.injection_executions AS x ON x.episode_id=p.episode_id AND x.run_id=@run
      INNER JOIN workload.schedule_items AS item ON item.schedule_item_id=x.schedule_item_id
      INNER JOIN workload.schedules AS schedule ON schedule.schedule_id=item.schedule_id
      INNER JOIN workload.scenario_variants AS v ON v.scenario_variant_id=item.scenario_variant_id
      INNER JOIN workload.scenario_definitions AS s ON s.scenario_id=v.scenario_id
      WHERE schedule.schedule_name=@schedule;
    `);
  packetCount = packets.recordset.length;
  for (const packet of packets.recordset) {
    const request = JSON.parse(packet.injector_request_json) as { correlationToken: string };
    const recomputed = hashJson(JSON.parse(packet.packet_json));
    if (recomputed !== packet.packet_sha256) findings.push({ episodeId: packet.episode_id, kind: "packet_hash_mismatch" });
    for (const finding of packetLeakageFindings(packet.packet_json, [
      request.correlationToken, request.correlationToken.slice(0, 12),
      packet.scenario_id, packet.scenario_group_id,
    ])) findings.push({ episodeId: packet.episode_id, kind: finding });
  }
  const crossRole = await pool.request().query(`
    SELECT scenario_group_id,COUNT(DISTINCT split_role) AS role_count
    FROM eval.ground_truth_episodes GROUP BY scenario_group_id HAVING COUNT(DISTINCT split_role)>1;
  `);
  for (const row of crossRole.recordset) findings.push({ kind: "scenario_group_crosses_roles", ...row });
  const multiEpisodeEvents = await pool.request().query(`
    SELECT canonical_event_id,COUNT(*) AS episode_count
    FROM ingest.injection_event_links GROUP BY canonical_event_id HAVING COUNT(*)>1;
  `);
  for (const row of multiEpisodeEvents.recordset) findings.push({ kind: "canonical_event_in_multiple_episodes", ...row });
  const unprotected = await pool.request().query(`
    SELECT p.episode_id
    FROM ingest.incident_packets AS p
    LEFT JOIN eval.ground_truth_episodes AS truth ON truth.episode_id=p.episode_id
    WHERE p.is_valid=1 AND truth.episode_id IS NULL;
  `);
  for (const row of unprotected.recordset) findings.push({ kind: "packet_missing_protected_truth", ...row });
  if (findings.length > 0) {
    await pool.request().query("UPDATE ingest.incident_packets SET is_valid=0 WHERE is_valid=1;");
  }
} finally {
  await pool.close();
}

const receiptBody = {
  schemaVersion: 1, runId: run.runId, scheduleName, packetCount,
  findingCount: findings.length, disposition: findings.length === 0 ? "PASS" : "STOP_DATA",
  findings,
};
const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
await atomicWrite(`${runDirectory}/packets/leakage-audit.json`, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({ runId: run.runId, packetCount, findingCount: findings.length, disposition: receipt.disposition, receiptSha256: receipt.receiptSha256 }, null, 2));
if (findings.length > 0) throw new Error(`Packet audit found ${findings.length} issue(s)`);

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
