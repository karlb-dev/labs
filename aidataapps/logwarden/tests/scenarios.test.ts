import { scenarioCatalogSchema, scenarioIdentity } from "../src/scenarios.js";

describe("scenario identities", () => {
  it("are stable under object key order", () => {
    const scenario = scenarioCatalogSchema.parse({
      schemaVersion: 1,
      catalogId: "test",
      scenarios: [{
        id: "test-scenario", groupId: "test-group", family: "test_family",
        regime: "K", description: "test", injector: "conversion_error",
        expectedClass: "conversion", expectedSeverity: "medium", shouldAbstain: false,
        expectedEvidence: [{ source: "xe", event: "error_reported", errorNumber: 245, minimumCount: 1 }],
      }],
    }).scenarios[0]!;
    expect(scenarioIdentity("test", scenario).sha256).toHaveLength(64);
    expect(scenarioIdentity("test", scenario)).toEqual(scenarioIdentity("test", { ...scenario }));
  });
});
