import { batchTelemetryLinks, DEFAULT_TELEMETRY_LINK_BATCH_SIZE } from "../src/telemetry-links.js";

describe("batched telemetry span linking", () => {
  it("partitions links without changing their order", () => {
    const links = Array.from({ length: 5 }, (_, index) => ({
      traceId: String(index).padStart(32, "0"),
      spanId: String(index).padStart(16, "0"),
      turnId: index + 1,
    }));
    expect(batchTelemetryLinks(links, 2)).toEqual([
      links.slice(0, 2), links.slice(2, 4), links.slice(4),
    ]);
  });

  it("uses a bounded production batch size and accepts an empty input", () => {
    expect(DEFAULT_TELEMETRY_LINK_BATCH_SIZE).toBe(1_000);
    expect(batchTelemetryLinks([])).toEqual([]);
  });

  it("rejects duplicate span identities before issuing updates", () => {
    const link = { traceId: "a".repeat(32), spanId: "b".repeat(16), turnId: 1 };
    expect(() => batchTelemetryLinks([link, { ...link, modelRequestId: 2 }]))
      .toThrow("Duplicate telemetry span link");
  });

  it("rejects unsafe batch sizes", () => {
    expect(() => batchTelemetryLinks([], 0)).toThrow("Invalid telemetry link batch size");
    expect(() => batchTelemetryLinks([], 10_001)).toThrow("Invalid telemetry link batch size");
  });
});
