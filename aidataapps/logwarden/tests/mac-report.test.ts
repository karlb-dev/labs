import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));

describe("mac evaluation report snapshot", () => {
  const data = JSON.parse(
    readFileSync(`${root}/docs/reports/mac-eval-bench-20260823-data.json`, "utf8"),
  ) as { models: unknown[]; episodes: unknown[]; grid: Record<string, unknown[]> };

  it("is rebuilt byte-for-byte from the retained data and template", () => {
    const template = readFileSync(`${root}/templates/mac-eval-bench.template.html`, "utf8");
    const actual = readFileSync(`${root}/docs/reports/mac-eval-bench-20260823.html`, "utf8");
    expect(template.replace("__DATA_JSON__", JSON.stringify(data, null, 1))).toBe(actual);
  });

  it("contains a complete nine-serving by sixteen-episode grid", () => {
    expect(data.models).toHaveLength(9);
    expect(data.episodes).toHaveLength(16);
    expect(Object.values(data.grid)).toHaveLength(9);
    expect(Object.values(data.grid).every((states) => states.length === 16)).toBe(true);
  });
});
