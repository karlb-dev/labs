import { describe, expect, it } from "vitest";
import { retainedCheckpointReceiptNames } from "../src/checkpoint-retention.js";

describe("checkpoint retention", () => {
  const receipts = ["001", "002", "003", "004"].map((name) => ({ name, receiptSha256: `hash-${name}` }));

  it("retains pinned and newest rolling receipts", () => {
    expect([...retainedCheckpointReceiptNames(receipts, new Set(["hash-001"]), 2)].sort())
      .toEqual(["001", "003", "004"]);
  });

  it("rejects an empty rolling window", () => {
    expect(() => retainedCheckpointReceiptNames(receipts, new Set(), 0)).toThrow(/positive integer/);
  });
});
