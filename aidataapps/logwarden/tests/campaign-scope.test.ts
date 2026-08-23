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

  it("projects worker telemetry serially after parallel model execution", async () => {
    const source = await readFile("scripts/replay-campaign.ts", "utf8");
    expect(source).toContain("for (const execution of workerExecutions)");
    expect(source).toContain("execution.journalPath");
    expect(source).toContain("replay workers failed: ${details.join");
  });

  it("evaluates resumed qwen pilots from the prediction-selected agent attempts", async () => {
    const source = await readFile("scripts/qwen-smoke-e2e.ts", "utf8");
    expect(source).toContain("WITH current_predictions AS");
    expect(source).toContain("prediction.agent_run_id=agent_run.agent_run_id");
    expect(source).toContain("prediction.agent_run_id=turn.agent_run_id");
  });

  it("tests direct-arm isolation against an untrusted packet tool list", async () => {
    const source = await readFile("scripts/chat-port-gate.ts", "utf8");
    expect(source).toContain('index === fixtures.length - 1 ? ["get_log_space", "runbook_search"] : []');
  });
});
