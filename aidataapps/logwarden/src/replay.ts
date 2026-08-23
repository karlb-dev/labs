import { z } from "zod";
import type { AgentLoopResult, PrimaryAgentArm } from "./agent-loop.js";
import { hashJson } from "./hash.js";

export const replayRoles = [
  "dev",
  "calibration",
  "test_id",
  "test_variant_holdout",
  "test_unknown",
  "test_live_parity",
  "test_storm",
] as const;

export type ReplayRole = typeof replayRoles[number];

export interface ReplayEpisode {
  episodeId: string;
  splitRole: ReplayRole;
  packetSha256: string;
  packet: unknown;
  expectedRunbooks: string[];
}

export interface ReplayCell {
  episodeId: string;
  splitRole: ReplayRole;
  packetSha256: string;
  arm: PrimaryAgentArm;
}

export interface ReplayPredictionInput {
  episodeId: string;
  splitRole: ReplayRole;
  packetSha256: string;
  modelProfileId: string;
  agentArmId: PrimaryAgentArm;
  decodeConfigId: string;
  jobId: number;
  jobAttemptId: number | null;
  agentRunId: number | null;
  decisionId: number | null;
  result: Pick<AgentLoopResult,
    "status" | "terminalReason" | "traceId" | "modelTurnCount" | "toolCallCount" |
    "toolResultChars" | "cacheHitCount" | "snapshotMissCount" | "elapsedMs"
  > | null;
  decision: {
    incidentClass: string;
    severity: string;
    action: string;
    confidence: number;
    abstained: boolean;
    decisionSha256: string;
  } | null;
  failureClass: string | null;
  failureDetail: string | null;
}

const roleSchema = z.enum(replayRoles);
const armSchema = z.enum(["A-direct", "A-rag", "A-tools"]);

export function parseReplayRoles(source: string): ReplayRole[] {
  const values = uniqueList(source);
  if (values.length === 0) throw new Error("At least one replay role is required");
  return values.map((value) => roleSchema.parse(value));
}

export function parseReplayArms(source: string): PrimaryAgentArm[] {
  const values = uniqueList(source);
  if (values.length === 0) throw new Error("At least one replay arm is required");
  return values.map((value) => armSchema.parse(value));
}

export function buildReplayCells(
  episodes: ReplayEpisode[],
  arms: PrimaryAgentArm[],
): ReplayCell[] {
  const seenEpisodes = new Set<string>();
  for (const episode of episodes) {
    if (seenEpisodes.has(episode.episodeId)) throw new Error(`Duplicate replay episode ${episode.episodeId}`);
    seenEpisodes.add(episode.episodeId);
  }
  return arms.flatMap((arm) => episodes
    .filter((episode) => arm !== "A-rag" || episode.expectedRunbooks.length > 0)
    .map((episode) => ({
      episodeId: episode.episodeId,
      splitRole: episode.splitRole,
      packetSha256: episode.packetSha256,
      arm,
    })))
    .sort((left, right) => left.arm.localeCompare(right.arm)
      || left.splitRole.localeCompare(right.splitRole)
      || left.episodeId.localeCompare(right.episodeId));
}

export function replayJobKey(input: {
  campaignId: number;
  runKind: string;
  episodeId: string;
  modelProfileId: string;
  agentArmId: PrimaryAgentArm;
  decodeConfigId: string;
  sampleIndex?: number;
}): string {
  return hashJson({ schemaVersion: 1, sampleIndex: input.sampleIndex ?? 0, ...input });
}

export function inferenceDecode(value: Record<string, unknown>): Record<string, unknown> {
  const { transport: _transport, ...decode } = value;
  return decode;
}

export function replayPrediction(input: ReplayPredictionInput): {
  envelope: Record<string, unknown>;
  outcome: "decision" | "abstention" | "failure";
  completeCaseEligible: boolean;
  predictedClass: string | null;
  predictedSeverity: string | null;
  predictedAction: string | null;
  confidence: number | null;
  abstained: boolean | null;
  failureStage: string | null;
} {
  const outcome = input.decision === null
    ? "failure"
    : input.decision.abstained ? "abstention" : "decision";
  const failureStage = outcome === "failure" ? failureStageFor(input.result?.status ?? null, input.failureClass) : null;
  const envelope = {
    schemaVersion: 1,
    predictionSource: "agent",
    episodeId: input.episodeId,
    splitRole: input.splitRole,
    packetSha256: input.packetSha256,
    modelProfileId: input.modelProfileId,
    agentArmId: input.agentArmId,
    decodeConfigId: input.decodeConfigId,
    jobId: input.jobId,
    jobAttemptId: input.jobAttemptId,
    agentRunId: input.agentRunId,
    decisionId: input.decisionId,
    outcome,
    failureStage,
    failureClass: outcome === "failure" ? input.failureClass : null,
    failureDetail: outcome === "failure" ? input.failureDetail : null,
    result: input.result === null ? null : {
      status: input.result.status,
      terminalReason: input.result.terminalReason,
      traceId: input.result.traceId,
      modelTurnCount: input.result.modelTurnCount,
      toolCallCount: input.result.toolCallCount,
      toolResultChars: input.result.toolResultChars,
      cacheHitCount: input.result.cacheHitCount,
      snapshotMissCount: input.result.snapshotMissCount,
      elapsedMs: input.result.elapsedMs,
    },
    decision: input.decision,
  };
  return {
    envelope,
    outcome,
    completeCaseEligible: input.decision !== null,
    predictedClass: input.decision?.incidentClass ?? null,
    predictedSeverity: input.decision?.severity ?? null,
    predictedAction: input.decision?.action ?? null,
    confidence: input.decision?.confidence ?? null,
    abstained: input.decision?.abstained ?? null,
    failureStage,
  };
}

function failureStageFor(status: AgentLoopResult["status"] | null, failureClass: string | null): string {
  if (status === "contract_rejected") return "contract_validation";
  if (status === "policy_rejected") return "policy_validation";
  if (status === "model_timeout" || failureClass === "timeout") return "model_transport";
  if (failureClass?.startsWith("tool")) return "tool_execution";
  return status === null ? "orchestration" : "agent_loop";
}

function uniqueList(source: string): string[] {
  return [...new Set(source.split(",").map((value) => value.trim()).filter(Boolean))];
}
