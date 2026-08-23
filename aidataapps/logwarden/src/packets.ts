import { hashJson } from "./hash.js";

export interface PacketSourceEvent {
  canonicalEventId: string;
  sourceKind: string;
  eventName: string;
  occurredAtUtc: string;
  errorNumber: number | null;
  severity: number | null;
  state: number | null;
  message: string | null;
  clientAppName: string | null;
  fingerprint: string | null;
}

export function redactPacketText(value: string | null, correlationToken: string, episodeId: string): string | null {
  if (value === null) return null;
  return value
    .replace(/\/root\/logwarden-forbidden\/[^\s'"\]]+/gi, "/restricted/backup.bak")
    .replace(/LW_EVT_[0-9a-f]{32}/gi, "SIGNAL_TOKEN")
    .replace(/LW_Missing_[0-9a-f]{16}/gi, "MISSING_OBJECT")
    .replace(/lw_missing_[0-9a-f]{12}/gi, "UNKNOWN_LOGIN")
    .replaceAll(correlationToken, "SIGNAL_TOKEN")
    .replaceAll(correlationToken.slice(0, 12), "SIGNAL_ID")
    .replaceAll(correlationToken.slice(0, 8), "SIGNAL_ID")
    .replaceAll(episodeId, "EPISODE_ID");
}

export function buildIncidentPacket(input: {
  episodeId: string;
  correlationToken: string;
  sourceEvents: PacketSourceEvent[];
  availableTools: string[];
}) {
  const sourceEvents = input.sourceEvents.map((event) => ({
    ...event,
    message: redactPacketText(event.message, input.correlationToken, input.episodeId),
    clientAppName: redactPacketText(event.clientAppName, input.correlationToken, input.episodeId),
  }));
  const sourceSetHash = hashJson(sourceEvents.map((event) => ({
    canonicalEventId: event.canonicalEventId,
    fingerprint: event.fingerprint,
  })));
  return {
    packetVersion: "incident-packet-v1",
    episodeId: input.episodeId,
    anchorTimeUtc: sourceEvents[0]?.occurredAtUtc ?? null,
    sourceEvents,
    recentHistory: { sameFingerprint5m: 0, sameClass1h: 0, openRelatedIncidents: 0 },
    availableTools: [...input.availableTools].sort(),
    frozenToolSnapshots: {},
    sourceDiagnostics: {
      linkedEventCount: sourceEvents.length,
      sources: [...new Set(sourceEvents.map((event) => event.sourceKind))].sort(),
    },
    redactions: ["correlation_token", "disposable_identity", "restricted_path"],
    hashes: { sourceSetSha256: sourceSetHash },
  };
}

export function packetLeakageFindings(packetJson: string, protectedValues: string[]): string[] {
  const findings: string[] = [];
  for (const value of protectedValues.filter((entry) => entry.length >= 4)) {
    if (packetJson.toLowerCase().includes(value.toLowerCase())) findings.push(`protected_value:${value}`);
  }
  for (const key of ["groundTruth", "protectedTruth", "correctActions", "preferredAction", "shouldAbstain", "expectedClass", "expectedSeverity", "scenarioId", "scenarioGroupId"]) {
    if (packetJson.includes(`\"${key}\"`)) findings.push(`protected_key:${key}`);
  }
  return [...new Set(findings)].sort();
}
