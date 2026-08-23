import { readFile } from "node:fs/promises";

describe("model-client SQL gate queue hygiene", () => {
  it("retires verified retry fixtures from the shared replay queue", async () => {
    const source = await readFile("scripts/model-client-gate.ts", "utf8");
    expect(source).toContain("await retireVerifiedGateRetries(pool)");
    expect(source).toContain("item.status='retryable_failure'");
    expect(source).toContain("synthetic fixture retired from shared queue");
    expect(source).not.toContain("next_attempt_at_utc=NULL");
  });
});
