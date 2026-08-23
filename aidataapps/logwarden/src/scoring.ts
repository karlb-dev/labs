import { severities } from "./contracts.js";

export interface ScoreTruth {
  expectedClass: string;
  expectedSeverity: string;
  expectedAction: string;
  acceptableActions: string[];
  shouldAbstain: boolean;
  expectedRunbooks: string[];
  requiredTools: string[];
  optionalTools: string[];
  forbiddenTools: string[];
  costWeights: { miss: number; falseAlarm: number; unnecessaryTool: number };
}

export interface ScorePrediction {
  outcome: "decision" | "abstention" | "failure";
  predictedClass: string | null;
  predictedSeverity: string | null;
  predictedAction: string | null;
  abstained: boolean | null;
  contractValid: boolean;
  policyValid: boolean;
}

export interface ScoreToolInvocation {
  toolName: string;
  status: string;
  policyStatus: string;
  snapshotMiss: boolean;
}

export interface PredictionScore {
  endToEndSuccess: boolean;
  classCorrect: boolean;
  severityCorrect: boolean;
  severityOrdinalCost: number;
  actionCorrect: boolean;
  abstentionCorrect: boolean;
  contractSuccess: boolean;
  requiredToolsSatisfied: boolean;
  forbiddenToolsAvoided: boolean;
  citationPolicySatisfied: boolean;
  compositeScore: number;
  costWeightedLoss: number;
  unnecessaryToolCount: number;
  calledTools: string[];
}

export function scorePrediction(input: {
  truth: ScoreTruth;
  prediction: ScorePrediction;
  armId: string;
  tools: ScoreToolInvocation[];
  citedRunbooks: string[];
  returnedRunbooks: string[];
}): PredictionScore {
  validateTruth(input.truth);
  const successfulTools = input.tools
    .filter(toolCallValid)
    .map((tool) => tool.toolName);
  const calledTools = input.tools.map((tool) => tool.toolName);
  const successful = new Set(successfulTools);
  const requiredToolsSatisfied = input.truth.requiredTools.every((tool) => successful.has(tool));
  const forbiddenToolsAvoided = input.truth.forbiddenTools.every((tool) => !calledTools.includes(tool));
  const expectedToolSet = new Set([...input.truth.requiredTools, ...input.truth.optionalTools]);
  const unnecessaryToolCount = calledTools.filter((tool) => !expectedToolSet.has(tool)).length;
  const classCorrect = input.prediction.predictedClass === input.truth.expectedClass;
  const severityCorrect = input.prediction.predictedSeverity === input.truth.expectedSeverity;
  const severityOrdinalCost = severityCost(input.truth.expectedSeverity, input.prediction.predictedSeverity);
  const actionCorrect = input.prediction.predictedAction !== null && input.truth.acceptableActions.includes(input.prediction.predictedAction);
  const abstentionCorrect = input.prediction.abstained === input.truth.shouldAbstain;
  const contractSuccess = input.prediction.outcome !== "failure" && input.prediction.contractValid;
  const retrievalBearing = input.armId === "A-rag" || input.armId === "A-tools";
  const acceptableRunbooks = new Set(input.truth.expectedRunbooks);
  const returned = new Set(input.returnedRunbooks);
  const citationPolicySatisfied = !retrievalBearing || input.truth.expectedRunbooks.length === 0
    ? input.citedRunbooks.length === 0 || input.citedRunbooks.every((runbook) => returned.has(runbook))
    : input.citedRunbooks.some((runbook) => acceptableRunbooks.has(runbook) && returned.has(runbook))
      && input.citedRunbooks.every((runbook) => returned.has(runbook));
  const endToEndSuccess = contractSuccess && input.prediction.policyValid && classCorrect && actionCorrect
    && abstentionCorrect && requiredToolsSatisfied && forbiddenToolsAvoided && citationPolicySatisfied;
  const components = [
    classCorrect, severityCorrect, actionCorrect, abstentionCorrect, contractSuccess,
    requiredToolsSatisfied, forbiddenToolsAvoided, citationPolicySatisfied,
  ];
  const decisionFailure = !classCorrect || !actionCorrect;
  const falseAlarm = input.truth.expectedClass === "benign_noise" && decisionFailure;
  const costWeightedLoss = (decisionFailure ? (falseAlarm ? input.truth.costWeights.falseAlarm : input.truth.costWeights.miss) : 0)
    + severityOrdinalCost
    + unnecessaryToolCount * input.truth.costWeights.unnecessaryTool
    + (contractSuccess && input.prediction.policyValid ? 0 : input.truth.costWeights.miss);
  return {
    endToEndSuccess,
    classCorrect,
    severityCorrect,
    severityOrdinalCost,
    actionCorrect,
    abstentionCorrect,
    contractSuccess,
    requiredToolsSatisfied,
    forbiddenToolsAvoided,
    citationPolicySatisfied,
    compositeScore: components.filter(Boolean).length / components.length,
    costWeightedLoss,
    unnecessaryToolCount,
    calledTools,
  };
}

export function severityCost(expected: string, predicted: string | null): number {
  const expectedIndex = severities.indexOf(expected as typeof severities[number]);
  const predictedIndex = predicted === null ? -1 : severities.indexOf(predicted as typeof severities[number]);
  if (expectedIndex < 0) throw new Error(`Unknown expected severity ${expected}`);
  if (predictedIndex < 0) return 10;
  const distance = Math.abs(expectedIndex - predictedIndex);
  const undercallCosts = [0, 1, 3, 6, 10] as const;
  const base = undercallCosts[distance] ?? 10;
  return predictedIndex < expectedIndex ? base : base * 0.5;
}

export function toolScoreRows(
  truth: Pick<ScoreTruth, "requiredTools" | "optionalTools" | "forbiddenTools">,
  tools: ScoreToolInvocation[],
): Array<{
  toolName: string;
  expectation: "required" | "optional" | "forbidden" | "unspecified";
  calledCount: number;
  validCallCount: number;
  snapshotMissCount: number;
  toolErrorCount: number;
  correctUse: boolean;
  score: number;
}> {
  const names = [...new Set([...truth.requiredTools, ...truth.optionalTools, ...truth.forbiddenTools, ...tools.map((tool) => tool.toolName)])].sort();
  return names.map((toolName) => {
    const calls = tools.filter((tool) => tool.toolName === toolName);
    const expectation = truth.requiredTools.includes(toolName) ? "required"
      : truth.optionalTools.includes(toolName) ? "optional"
      : truth.forbiddenTools.includes(toolName) ? "forbidden" : "unspecified";
    const validCallCount = calls.filter(toolCallValid).length;
    const correctUse = expectation === "required" ? validCallCount > 0
      : expectation === "forbidden" || expectation === "unspecified" ? calls.length === 0
        : calls.every((tool) => tool.policyStatus === "allow" || tool.policyStatus === "allowed");
    return {
      toolName,
      expectation,
      calledCount: calls.length,
      validCallCount,
      snapshotMissCount: calls.filter((tool) => tool.snapshotMiss).length,
      toolErrorCount: calls.filter((tool) => tool.status !== "success").length,
      correctUse,
      score: correctUse ? 1 : 0,
    };
  });
}

function toolCallValid(tool: ScoreToolInvocation): boolean {
  return tool.status === "success" && (tool.policyStatus === "allow" || tool.policyStatus === "allowed") && !tool.snapshotMiss;
}

function validateTruth(truth: ScoreTruth): void {
  if (!truth.acceptableActions.includes(truth.expectedAction)) throw new Error("Preferred action is absent from acceptable actions");
  for (const [name, value] of Object.entries(truth.costWeights)) {
    if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid cost weight ${name}`);
  }
}
