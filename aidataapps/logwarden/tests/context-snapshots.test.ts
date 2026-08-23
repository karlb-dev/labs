import { describe, expect, it } from "vitest";
import { materializeSnapshotArguments } from "../src/context-snapshots.js";

describe("context snapshot argument materialization", () => {
  it("substitutes and canonicalizes an opaque disposable database", () => {
    expect(materializeSnapshotArguments(
      "get_log_space",
      { databaseName: "TARGET_DATABASE" },
      "LW_A1b2C3",
    )).toEqual({ databaseName: "lw_a1b2c3" });
  });

  it("fills bounded tool defaults", () => {
    expect(materializeSnapshotArguments(
      "get_blocking_snapshot",
      { databaseName: "LogWardenWorkload" },
    )).toEqual({ databaseName: "logwardenworkload", maxRows: 20 });
  });

  it("fails closed when a placeholder has no episode database", () => {
    expect(() => materializeSnapshotArguments(
      "get_active_transactions",
      { databaseName: "TARGET_DATABASE", maxRows: 5 },
    )).toThrow(/requires a disposable database/);
  });
});
