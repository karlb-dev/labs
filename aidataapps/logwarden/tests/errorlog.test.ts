import { parseErrorlogFiles } from "../src/errorlog.js";

describe("SQL Server ERRORLOG parser", () => {
  const fixture = [
    "2026-08-23 03:00:00.100 Logon     Error: 18456, Severity: 14, State: 38.",
    "2026-08-23 03:00:00.101 Logon     Login failed for user 'lw_missing'. Reason: Failed to open database. [CLIENT: 127.0.0.1]",
    "2026-08-23 03:00:01.000 spid12s   Recovery completed.",
    "continuation detail",
    "",
  ].join("\n");

  it("preserves rows and carries state to the paired login reason", () => {
    const scan = parseErrorlogFiles([{ path: "/var/opt/mssql/log/errorlog", content: fixture }]);
    expect(scan.records).toHaveLength(3);
    expect(scan.records[1]).toMatchObject({ errorNumber: 18456, severity: 14, state: 38 });
    expect(scan.records[2]!.rawText).toContain("continuation detail");
  });

  it("keeps source keys stable when a generation is renamed by rotation", () => {
    const before = parseErrorlogFiles([{ path: "/var/opt/mssql/log/errorlog", content: fixture }]);
    const after = parseErrorlogFiles([{ path: "/var/opt/mssql/log/errorlog.1", content: fixture }]);
    expect(after.records.map((row) => row.sourcePositionKey)).toEqual(before.records.map((row) => row.sourcePositionKey));
  });
});
