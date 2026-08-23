export interface CalibrationRow {
  confidence: number;
  correct: boolean;
  actionCorrect: boolean;
}

export interface PlattCalibrationModel {
  algorithm: "logit-platt-newton-l2-v1";
  intercept: number;
  slope: number;
  l2: number;
  iterations: number;
  converged: boolean;
}

export function fitPlattCalibration(rows: CalibrationRow[], l2 = 0.01): PlattCalibrationModel {
  validateRows(rows);
  if (!Number.isFinite(l2) || l2 <= 0) throw new Error("Platt L2 regularization must be positive");
  const features = rows.map((row) => logit(clampProbability(row.confidence)));
  const outcomes = rows.map((row) => Number(row.correct));
  const rate = clampProbability(outcomes.reduce((sum, value) => sum + value, 0) / outcomes.length);
  let intercept = logit(rate);
  let slope = 1;
  let converged = false;
  let iterations = 0;
  for (; iterations < 100; iterations += 1) {
    let gradientIntercept = 0;
    let gradientSlope = l2 * slope;
    let hessianIntercept = 0;
    let hessianCross = 0;
    let hessianSlope = l2;
    for (const [index, feature] of features.entries()) {
      const probability = sigmoid(intercept + slope * feature);
      const residual = probability - outcomes[index]!;
      const weight = Math.max(1e-9, probability * (1 - probability));
      gradientIntercept += residual;
      gradientSlope += residual * feature;
      hessianIntercept += weight;
      hessianCross += weight * feature;
      hessianSlope += weight * feature * feature;
    }
    const determinant = hessianIntercept * hessianSlope - hessianCross * hessianCross;
    if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) throw new Error("Platt Hessian is singular");
    const deltaIntercept = (hessianSlope * gradientIntercept - hessianCross * gradientSlope) / determinant;
    const deltaSlope = (-hessianCross * gradientIntercept + hessianIntercept * gradientSlope) / determinant;
    intercept -= deltaIntercept;
    slope -= deltaSlope;
    if (!Number.isFinite(intercept) || !Number.isFinite(slope)) throw new Error("Platt optimization diverged");
    if (Math.max(Math.abs(deltaIntercept), Math.abs(deltaSlope)) < 1e-10) {
      converged = true;
      iterations += 1;
      break;
    }
  }
  return { algorithm: "logit-platt-newton-l2-v1", intercept: round(intercept), slope: round(slope), l2, iterations, converged };
}

export function calibratedConfidence(model: PlattCalibrationModel, confidence: number): number {
  return round(sigmoid(model.intercept + model.slope * logit(clampProbability(confidence))));
}

export function calibrationMetrics(rows: CalibrationRow[], probabilities: number[], bins = 10): { brier: number; ece: number } {
  validateRows(rows);
  if (probabilities.length !== rows.length || probabilities.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new Error("Calibration probabilities do not align with rows");
  }
  if (!Number.isSafeInteger(bins) || bins < 2 || bins > 100) throw new Error("Calibration bins must be an integer from 2 through 100");
  const brier = probabilities.reduce((sum, probability, index) => sum + (probability - Number(rows[index]!.correct)) ** 2, 0) / rows.length;
  let ece = 0;
  for (let bin = 0; bin < bins; bin += 1) {
    const lower = bin / bins;
    const upper = (bin + 1) / bins;
    const selected = rows.map((row, index) => ({ row, probability: probabilities[index]! }))
      .filter(({ probability }) => probability >= lower && (bin === bins - 1 ? probability <= upper : probability < upper));
    if (selected.length === 0) continue;
    const confidence = selected.reduce((sum, value) => sum + value.probability, 0) / selected.length;
    const accuracy = selected.filter((value) => value.row.correct).length / selected.length;
    ece += selected.length / rows.length * Math.abs(accuracy - confidence);
  }
  return { brier: round(brier), ece: round(ece) };
}

export function selectActionThreshold(
  rows: CalibrationRow[], probabilities: number[], targetAccuracy: number, minimumCoverage: number,
): { threshold: number; coverage: number; actionAccuracy: number; meetsTarget: boolean; selectedRows: number } {
  validateRows(rows);
  if (probabilities.length !== rows.length) throw new Error("Threshold probabilities do not align with rows");
  const candidates = [...new Set([1, ...probabilities, 0])].sort((left, right) => right - left);
  const evaluated = candidates.map((threshold) => {
    const selected = rows.filter((_row, index) => probabilities[index]! >= threshold);
    return {
      threshold, selectedRows: selected.length, coverage: selected.length / rows.length,
      actionAccuracy: selected.length === 0 ? 0 : selected.filter((row) => row.actionCorrect).length / selected.length,
    };
  });
  const qualifying = evaluated.filter((value) => value.coverage >= minimumCoverage && value.actionAccuracy >= targetAccuracy)
    .sort((left, right) => right.coverage - left.coverage || left.threshold - right.threshold);
  const selected = qualifying[0] ?? [...evaluated].sort((left, right) => right.actionAccuracy - left.actionAccuracy || right.coverage - left.coverage)[0]!;
  return { ...Object.fromEntries(Object.entries(selected).map(([key, value]) => [key, typeof value === "number" ? round(value) : value])) as typeof selected,
    meetsTarget: qualifying.length > 0 };
}

function validateRows(rows: CalibrationRow[]): void {
  if (rows.length < 20) throw new Error("Calibration requires at least 20 completed confidence rows");
  for (const [index, row] of rows.entries()) if (!Number.isFinite(row.confidence) || row.confidence < 0 || row.confidence > 1) {
    throw new Error(`Calibration row ${index} has invalid confidence`);
  }
}
function clampProbability(value: number): number { return Math.min(1 - 1e-6, Math.max(1e-6, value)); }
function logit(value: number): number { return Math.log(value / (1 - value)); }
function sigmoid(value: number): number { return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value)); }
function round(value: number): number { return Math.round(value * 1_000_000_000) / 1_000_000_000; }

