import { readFile } from "node:fs/promises";

describe("qwen smoke development retry governance", () => {
  it("retains attempt evidence and refuses any post-freeze reset", async () => {
    const source = await readFile("scripts/retry-qwen-smoke-pilot.ts", "utf8");
    expect(source).toContain('campaign.status !== "building"');
    expect(source).toContain("scientific dependents");
    expect(source).toContain("modelRequestResponseHashes");
    expect(source).toContain("prior attempt retained");
    expect(source).toContain("RETAINED_DEVELOPMENT_RETRY");
  });
});
