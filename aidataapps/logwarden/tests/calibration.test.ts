import { calibratedConfidence, calibrationMetrics, fitPlattCalibration, selectActionThreshold } from "../src/calibration.js";

describe("frozen calibration path", () => {
  const rows = Array.from({ length: 40 }, (_, index) => ({
    confidence: (index + 1) / 42,
    correct: index >= 18,
    actionCorrect: index >= 16,
  }));

  it("fits deterministic finite Platt coefficients", () => {
    const first = fitPlattCalibration(rows);
    const second = fitPlattCalibration(rows);
    expect(first).toEqual(second);
    expect(first.converged).toBe(true);
    expect(calibratedConfidence(first, 0.9)).toBeGreaterThan(calibratedConfidence(first, 0.1));
  });

  it("reports Brier/ECE and freezes a maximum-coverage action threshold", () => {
    const model = fitPlattCalibration(rows);
    const probabilities = rows.map((row) => calibratedConfidence(model, row.confidence));
    const metrics = calibrationMetrics(rows, probabilities);
    expect(metrics.brier).toBeGreaterThanOrEqual(0);
    expect(metrics.ece).toBeGreaterThanOrEqual(0);
    const threshold = selectActionThreshold(rows, probabilities, 0.85, 0.5);
    expect(threshold.coverage).toBeGreaterThanOrEqual(0);
    expect(threshold.coverage).toBeLessThanOrEqual(1);
  });
});

