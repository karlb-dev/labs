import { buildIncidentPacket, packetLeakageFindings, redactPacketText } from "../src/packets.js";

describe("incident packet boundary", () => {
  const token = "0123456789abcdef0123456789abcdef";
  it("redacts correlation-bearing SQL identities and paths", () => {
    const raw = `LW_EVT_${token} lab.LW_Missing_${token.slice(0, 16)} lw_missing_${token.slice(0, 12)} /root/logwarden-forbidden/LW_EVT_${token}.bak`;
    const redacted = redactPacketText(raw, token, "episode-1")!;
    expect(redacted).not.toContain(token);
    expect(redacted).not.toContain("LW_Missing");
    expect(redacted).not.toContain("lw_missing");
    expect(redacted).toContain("/restricted/backup.bak");
  });

  it("builds a label-free stable packet", () => {
    const packet = buildIncidentPacket({
      episodeId: "episode-1", correlationToken: token, availableTools: [],
      sourceEvents: [{ canonicalEventId: "1", sourceKind: "xe", eventName: "error_reported",
        occurredAtUtc: "2026-01-01T00:00:00.000Z", errorNumber: 245, severity: 16,
        state: 1, message: `LW_EVT_${token}`, clientAppName: "LogWarden-Inject", fingerprint: "a".repeat(64) }],
    });
    expect(packetLeakageFindings(JSON.stringify(packet), [token, "scenario-secret"])).toEqual([]);
  });
});
