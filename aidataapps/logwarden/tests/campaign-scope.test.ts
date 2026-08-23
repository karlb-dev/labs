import { readFile } from "node:fs/promises";

const standardScopedScripts = [
  "scripts/retrieval-evaluate.ts",
  "scripts/search-freeze.ts",
  "scripts/baseline-rules-run.ts",
  "scripts/derive-campaign-arms.ts",
  "scripts/replay-campaign.ts",
  "scripts/qwen-smoke-e2e.ts",
  "scripts/campaign-freeze.ts",
] as const;

describe("standard campaign schedule isolation", () => {
  it.each(standardScopedScripts)("keeps smoke packets out of %s", async (path) => {
    const source = await readFile(path, "utf8");
    expect(source).toContain("schedule.schedule_name='standard-v1'");
  });

  it("qualifies joined replay recovery timestamps", async () => {
    const source = await readFile("scripts/replay-campaign.ts", "utf8");
    expect(source).toContain("COALESCE(job.completed_at_utc,SYSUTCDATETIME())");
    expect(source).toContain("COALESCE(item.completed_at_utc,SYSUTCDATETIME())");
  });
});
