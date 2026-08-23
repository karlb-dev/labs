import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { hashJson } from "../src/hash.js";

const root = fileURLToPath(new URL("../", import.meta.url));

interface Tier1ReportData {
  primaryAdjudication: string;
  summary: {
    formalContrasts: number;
    formalPositiveCount: number;
    rulesActionAccuracy: number;
  };
  baselines: unknown[];
  agents: Array<{ actionAccuracy: number }>;
  figures: Array<{ dataUri: string }>;
  receiptSha256: string;
}

describe("Tier 1 model-comparison report snapshot", () => {
  const data = JSON.parse(
    readFileSync(`${root}/docs/reports/logwarden-tier1-model-comparison-20260823-data.json`, "utf8"),
  ) as Tier1ReportData;

  it("is rebuilt byte-for-byte from the retained payload and template", () => {
    const template = readFileSync(`${root}/templates/tier1-model-comparison.template.html`, "utf8");
    const actual = readFileSync(`${root}/docs/reports/logwarden-tier1-model-comparison-20260823.html`, "utf8");
    const embedded = JSON.stringify(data, null, 1).replace(/<\/script/gi, "<\\/script");
    expect(template.replace("__DATA_JSON__", embedded)).toBe(actual);
  });

  it("has a valid content receipt", () => {
    const { receiptSha256, ...body } = data;
    expect(hashJson(body)).toBe(receiptSha256);
  });

  it("preserves the declared clean-null result and complete comparison grid", () => {
    expect(data.primaryAdjudication).toBe("CLEAN_NULL");
    expect(data.baselines).toHaveLength(6);
    expect(data.agents).toHaveLength(9);
    expect(data.summary.formalContrasts).toBe(36);
    expect(data.summary.formalPositiveCount).toBe(0);
    expect(data.summary.rulesActionAccuracy).toBeCloseTo(0.916666667, 8);
    expect(data.agents.every((row) => row.actionAccuracy < data.summary.rulesActionAccuracy)).toBe(true);
  });

  it("embeds all six selected frozen figures", () => {
    expect(data.figures).toHaveLength(6);
    expect(data.figures.every((figure) => figure.dataUri.startsWith("data:image/png;base64,"))).toBe(true);
  });
});
