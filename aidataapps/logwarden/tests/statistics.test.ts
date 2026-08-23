import { execFileSync } from "node:child_process";

describe("paired power implementation", () => {
  it("passes its deterministic numerical self-test", () => {
    const output = execFileSync("python3", ["analysis/statistics.py", "--self-test"], { encoding: "utf8" });
    expect(JSON.parse(output)).toEqual({ selfTest: "PASS" });
  });
});
