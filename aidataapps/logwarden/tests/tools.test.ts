import { canonicalToolArguments, canonicalToolArgumentsSha256, loadToolRegistry, toolRegistrySha256 } from "../src/tools.js";

describe("tool contracts", () => {
  it("fills defaults and canonicalizes identifier case", () => {
    expect(canonicalToolArguments("get_log_space", { databaseName: "LogWardenWorkload" })).toEqual({ databaseName: "logwardenworkload" });
    expect(canonicalToolArguments("get_blocking_snapshot", { databaseName: "LW_DB_1" })).toEqual({ databaseName: "lw_db_1", maxRows: 20 });
    expect(canonicalToolArgumentsSha256("get_log_space", { databaseName: "LogWardenWorkload" }))
      .toBe(canonicalToolArgumentsSha256("get_log_space", { databaseName: "logwardenworkload" }));
  });

  it("rejects raw SQL, unknown keys, and unsafe database identifiers", () => {
    expect(() => canonicalToolArguments("get_log_space", { databaseName: "master" })).toThrow();
    expect(() => canonicalToolArguments("get_log_space", { databaseName: "LogWardenWorkload]; DROP DATABASE x;--" })).toThrow();
    expect(() => canonicalToolArguments("get_deadlock_graph", { maxRows: 5, sql: "select 1" })).toThrow();
  });

  it("keeps the checked-in registry aligned with executable schemas", () => {
    const registry = loadToolRegistry();
    expect(registry.frozen).toBe(false);
    expect(Object.values(registry.tools).every((tool) => tool.mutates === false)).toBe(true);
    expect(toolRegistrySha256()).toMatch(/^[0-9a-f]{64}$/);
  });
});
