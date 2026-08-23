import { sha256 } from "./hash.js";

export interface ErrorlogFile {
  path: string;
  content: string;
}

export interface ParsedErrorlogRecord {
  sourcePositionKey: string;
  sourceFileName: string;
  sourceFileOffset: number;
  occurredAtUtc: string;
  processInfo: string;
  rawText: string;
  rawSha256: string;
  messageNormalized: string;
  normalizedSha256: string;
  errorNumber: number | null;
  severity: number | null;
  state: number | null;
  canonicalJson: Record<string, unknown>;
}

export interface ParsedErrorlogScan {
  records: ParsedErrorlogRecord[];
  cursor: {
    schemaVersion: 1;
    parserVersion: "errorlog-file-v1";
    generations: Array<{
      identity: string;
      observedFileName: string;
      firstTimestampUtc: string;
      recordCount: number;
      byteCount: number;
      contentSha256: string;
    }>;
  };
}

const entryPattern = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+(\S+)\s+(.*)$/;
const errorPattern = /\bError:\s*(\d+)\s*,\s*Severity:\s*(\d+)\s*,\s*State:\s*(\d+)/i;

export function parseErrorlogFiles(files: ErrorlogFile[]): ParsedErrorlogScan {
  const records: ParsedErrorlogRecord[] = [];
  const generations: ParsedErrorlogScan["cursor"]["generations"] = [];

  for (const file of files) {
    const logical = logicalEntries(file.content);
    if (logical.length === 0) continue;
    const generationIdentity = sha256(`${logical[0]!.timestamp}|${logical[0]!.processInfo}|${logical[0]!.rawText}`);
    const recentErrors = new Map<string, { atMs: number; errorNumber: number; severity: number; state: number }>();
    for (const entry of logical) {
      const occurred = parseTimestamp(entry.timestamp);
      const direct = errorPattern.exec(entry.message);
      let errorNumber = direct === null ? null : Number(direct[1]);
      let severity = direct === null ? null : Number(direct[2]);
      let state = direct === null ? null : Number(direct[3]);
      if (direct !== null) {
        recentErrors.set(entry.processInfo, { atMs: occurred.getTime(), errorNumber: errorNumber!, severity: severity!, state: state! });
      } else {
        const prior = recentErrors.get(entry.processInfo);
        if (prior !== undefined && occurred.getTime() - prior.atMs <= 100) {
          ({ errorNumber, severity, state } = prior);
        }
      }
      const messageNormalized = entry.message.trim().replace(/\s+/g, " ").toLowerCase();
      const rawSha256 = sha256(entry.rawText);
      const sourcePositionKey = sha256([
        "errorlog", generationIdentity, String(entry.lineOrdinal), occurred.toISOString(), entry.processInfo, rawSha256,
      ].join("|"));
      const canonicalJson = {
        schemaVersion: 1,
        source: "errorlog",
        generationIdentity,
        occurredAtUtc: occurred.toISOString(),
        processInfo: entry.processInfo,
        message: entry.message,
        errorNumber,
        severity,
        state,
      };
      records.push({
        sourcePositionKey,
        sourceFileName: file.path,
        sourceFileOffset: entry.lineOrdinal,
        occurredAtUtc: occurred.toISOString(),
        processInfo: entry.processInfo,
        rawText: entry.rawText,
        rawSha256,
        messageNormalized,
        normalizedSha256: sha256(JSON.stringify(canonicalJson)),
        errorNumber,
        severity,
        state,
        canonicalJson,
      });
    }
    generations.push({
      identity: generationIdentity,
      observedFileName: file.path,
      firstTimestampUtc: parseTimestamp(logical[0]!.timestamp).toISOString(),
      recordCount: logical.length,
      byteCount: Buffer.byteLength(file.content),
      contentSha256: sha256(file.content),
    });
  }

  records.sort((a, b) => a.occurredAtUtc.localeCompare(b.occurredAtUtc) ||
    a.sourcePositionKey.localeCompare(b.sourcePositionKey));
  generations.sort((a, b) => a.firstTimestampUtc.localeCompare(b.firstTimestampUtc) ||
    a.identity.localeCompare(b.identity));
  return { records, cursor: { schemaVersion: 1, parserVersion: "errorlog-file-v1", generations } };
}

function logicalEntries(content: string): Array<{
  timestamp: string;
  processInfo: string;
  message: string;
  rawText: string;
  lineOrdinal: number;
}> {
  const entries: Array<{ timestamp: string; processInfo: string; message: string; rawText: string; lineOrdinal: number }> = [];
  for (const [index, line] of content.replaceAll("\r\n", "\n").split("\n").entries()) {
    const match = entryPattern.exec(line);
    if (match !== null) {
      entries.push({ timestamp: match[1]!, processInfo: match[2]!, message: match[3]!, rawText: line, lineOrdinal: index });
    } else if (line.trim() !== "" && entries.length > 0) {
      const prior = entries.at(-1)!;
      prior.message += `\n${line}`;
      prior.rawText += `\n${line}`;
    }
  }
  return entries;
}

function parseTimestamp(value: string): Date {
  const parsed = new Date(`${value.replace(" ", "T")}Z`);
  if (Number.isNaN(parsed.getTime())) throw new Error(`Invalid SQL ERRORLOG timestamp: ${value}`);
  return parsed;
}
