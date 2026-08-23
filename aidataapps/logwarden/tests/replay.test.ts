import { describe, expect, it } from "vitest";
import {
  buildReplayCells,
  inferenceDecode,
  parseReplayArms,
  parseReplayRoles,
  replayJobKey,
  replayPrediction,
  type ReplayEpisode,
} from "../src/replay.js";

const episodes: ReplayEpisode[] = [
  { episodeId: "e2", splitRole: "dev", packetSha256: "b".repeat(64), packet: {}, expectedRunbooks: [] },
  { episodeId: "e1", splitRole: "dev", packetSha256: "a".repeat(64), packet: {}, expectedRunbooks: ["TSG-1"] },
];

describe("replay campaign identities", () => {
  it("builds a stable arm ladder and limits retrieval to covered episodes", () => {
    expect(buildReplayCells(episodes, ["A-tools", "A-rag", "A-direct"]).map((cell) => `${cell.arm}:${cell.episodeId}`)).toEqual([
      "A-direct:e1", "A-direct:e2", "A-rag:e1", "A-tools:e1", "A-tools:e2",
    ]);
  });

  it("parses unique governed roles and arms", () => {
    expect(parseReplayRoles("dev,dev,calibration")).toEqual(["dev", "calibration"]);
    expect(parseReplayArms("A-direct,A-tools")).toEqual(["A-direct", "A-tools"]);
    expect(() => parseReplayArms("A-router")).toThrow();
  });

  it("removes the application-only transport marker from vLLM decode", () => {
    expect(inferenceDecode({ temperature: 0, stream: false, transport: "structured_json" })).toEqual({ temperature: 0, stream: false });
  });

  it("hashes exact cell identity", () => {
    const value = { campaignId: 7, runKind: "agent_replay", episodeId: "e1", modelProfileId: "qwen-smoke", agentArmId: "A-direct" as const, decodeConfigId: "primary-json-v3" };
    expect(replayJobKey(value)).toBe(replayJobKey({ ...value }));
    expect(replayJobKey(value)).not.toBe(replayJobKey({ ...value, agentArmId: "A-tools" }));
  });

  it("keeps missing inference as an eligible failure", () => {
    const value = replayPrediction({
      episodeId: "e1", splitRole: "dev", packetSha256: "a".repeat(64), modelProfileId: "qwen-smoke",
      agentArmId: "A-direct", decodeConfigId: "primary-json-v3", jobId: 1, jobAttemptId: 1,
      agentRunId: null, decisionId: null, result: null, decision: null,
      failureClass: "worker_interrupted", failureDetail: "worker stopped",
    });
    expect(value.outcome).toBe("failure");
    expect(value.completeCaseEligible).toBe(false);
    expect(value.failureStage).toBe("orchestration");
  });
});
