import { migrationMode, splitSqlBatches } from "../src/migrations.js";

describe("SQL migration batches", () => {
  it("splits only a standalone GO delimiter", () => {
    expect(splitSqlBatches("SELECT 'go';\nGO\nSELECT 2;\ngo -- delimiter\nSELECT 3;")).toEqual([
      "SELECT 'go';",
      "SELECT 2;",
      "SELECT 3;",
    ]);
  });

  it("requires an explicit directive for non-transactional DDL", () => {
    expect(migrationMode("SELECT 1;")).toBe("transactional");
    expect(migrationMode("-- logwarden:migration-mode=non-transactional\nSELECT 1;")).toBe("non_transactional");
  });
});
