import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

// GT-2-lite: load the package catalog text files into catalog.* tables.
// Raw file text is authoritative (the replay prompts already embed it);
// the structured parse serves the grammar-aware B2 baseline and the
// identifier-binding validators. Tolerant parser: every non-blank line is
// header, parsed object, or catalog.extras — nothing silently dropped.
// Idempotent: wipe-and-reload per invocation.

const CATALOG_DIR = "data/ghosttype_dataset_v2/catalogs";

interface ParsedColumn {
  name: string; isParameter: boolean; type: string | null; nullability: string | null;
  isPk: boolean; fkTarget: string | null; isGenerated: boolean; raw: string;
}
interface ParsedObject {
  kind: string; schemaName: string | null; objectName: string; namesOnly: boolean;
  unavailable: boolean; systemVersioned: boolean; synonymTarget: string | null;
  annotations: string[]; sourceLine: number; raw: string; columns: ParsedColumn[];
}
interface ParsedCatalog { headers: string[]; objects: ParsedObject[]; extras: Array<{ kind: string; line: number; raw: string }> }

// Split on commas at bracket/paren depth 0.
function splitTop(text: string): string[] {
  const parts: string[] = [];
  let depth = 0, inBracket = false, current = "";
  for (const ch of text) {
    if (inBracket) { current += ch; if (ch === "]") inBracket = false; continue; }
    if (ch === "[") { inBracket = true; current += ch; continue; }
    if (ch === "(") depth += 1;
    if (ch === ")") depth -= 1;
    if (ch === "," && depth === 0) { parts.push(current.trim()); current = ""; continue; }
    current += ch;
  }
  if (current.trim() !== "") parts.push(current.trim());
  return parts;
}

// "dbo.[User Profile]" -> {schema:"dbo", name:"User Profile"}; "sys.tables" -> system-style.
function splitQualified(name: string): { schema: string | null; object: string } {
  const parts: string[] = [];
  let current = "", inBracket = false;
  for (const ch of name.trim()) {
    if (inBracket) { if (ch === "]") { inBracket = false; } else current += ch; continue; }
    if (ch === "[") { inBracket = true; continue; }
    if (ch === ".") { parts.push(current); current = ""; continue; }
    current += ch;
  }
  parts.push(current);
  if (parts.length === 1) return { schema: null, object: parts[0] };
  return { schema: parts.slice(0, -1).join("."), object: parts.at(-1)! };
}

function parseColumnEntry(raw: string): ParsedColumn | { annotation: string } {
  const entry = raw.trim();
  if (/^(PERIOD FOR|HISTORY_TABLE=|CONSTRAINT )/i.test(entry)) return { annotation: entry };
  let name: string, rest: string;
  if (entry.startsWith("[")) {
    const close = entry.indexOf("]");
    name = entry.slice(1, close);
    rest = entry.slice(close + 1).trim();
  } else {
    const space = entry.indexOf(" ");
    name = space === -1 ? entry : entry.slice(0, space);
    rest = space === -1 ? "" : entry.slice(space + 1).trim();
  }
  const isParameter = name.startsWith("@");
  const fk = rest.match(/FK->([^\s,]+)/);
  const typeMatch = rest.match(/^[A-Za-z_][\w]*(\s*\([^)]*\))?/);
  return {
    name, isParameter,
    type: typeMatch ? typeMatch[0].replace(/\s+/g, "") : null,
    nullability: /\bNOT NULL\b/.test(rest) ? "NOT NULL" : /\bNULL\b/.test(rest) ? "NULL" : null,
    isPk: /\bPK\b/.test(rest),
    fkTarget: fk ? fk[1] : null,
    isGenerated: /GENERATED ALWAYS AS ROW (START|END)/i.test(rest),
    raw: entry,
  };
}

function parseCatalog(text: string): ParsedCatalog {
  const headers: string[] = [];
  const objects: ParsedObject[] = [];
  const extras: Array<{ kind: string; line: number; raw: string }> = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i].trim();
    const lineNo = i + 1;
    if (line === "") continue;
    if (line.startsWith("--")) { headers.push(line); continue; }

    const namesList = line.match(/^(TABLE|VIEW|ROUTINE) NAMES(?::\s*(.+)|\s+(\S+)\s*\((.+)\))$/);
    if (namesList) {
      const kind = namesList[1] === "TABLE" ? "table" : namesList[1] === "VIEW" ? "view" : "procedure";
      if (namesList[2] !== undefined) {
        for (const qualified of splitTop(namesList[2])) {
          const { schema, object } = splitQualified(qualified);
          objects.push({ kind, schemaName: schema, objectName: object, namesOnly: true, unavailable: false, systemVersioned: false, synonymTarget: null, annotations: [], sourceLine: lineNo, raw: line, columns: [] });
        }
      } else {
        for (const name of splitTop(namesList[4])) {
          objects.push({ kind, schemaName: namesList[3], objectName: name, namesOnly: true, unavailable: false, systemVersioned: false, synonymTarget: null, annotations: [], sourceLine: lineNo, raw: line, columns: [] });
        }
      }
      continue;
    }

    // Object definitions: name runs to the first "(" (names may contain
    // unbracketed spaces, e.g. Northwind's "VIEW dbo.Orders Qry (...)");
    // the column list runs to the last ")"; any tail (RETURNS ...) is an
    // annotation.
    const objectKindWord = line.match(/^(TABLE|VIEW|PROCEDURE|SCALAR FUNCTION)\s+/);
    if (objectKindWord && line.includes("(")) {
      const kindWord = objectKindWord[1];
      const kind = kindWord === "TABLE" ? "table" : kindWord === "VIEW" ? "view" : kindWord === "PROCEDURE" ? "procedure" : "function";
      const parenStart = line.indexOf("(");
      const parenEnd = line.lastIndexOf(")");
      let head = line.slice(objectKindWord[0].length, parenStart).trim();
      const body = parenEnd > parenStart ? line.slice(parenStart + 1, parenEnd) : line.slice(parenStart + 1);
      const tail = parenEnd > parenStart ? line.slice(parenEnd + 1).trim() : "";
      const systemVersioned = /\bSYSTEM_VERSIONED$/.test(head);
      if (systemVersioned) head = head.replace(/\s*SYSTEM_VERSIONED$/, "");
      const { schema, object } = splitQualified(head);
      const annotations: string[] = [];
      if (tail !== "") annotations.push(tail);
      const columns: ParsedColumn[] = [];
      for (const entry of splitTop(body)) {
        const parsed = parseColumnEntry(entry);
        if ("annotation" in parsed) annotations.push(parsed.annotation);
        else columns.push(parsed);
      }
      objects.push({ kind, schemaName: schema, objectName: object, namesOnly: false, unavailable: false, systemVersioned, synonymTarget: null, annotations, sourceLine: lineNo, raw: line, columns });
      continue;
    }

    const synonym = line.match(/^SYNONYM\s+(\S+)\s*->\s*(\S+)$/);
    if (synonym) {
      const { schema, object } = splitQualified(synonym[1]);
      objects.push({ kind: "synonym", schemaName: schema, objectName: object, namesOnly: false, unavailable: false, systemVersioned: false, synonymTarget: synonym[2], annotations: [], sourceLine: lineNo, raw: line, columns: [] });
      continue;
    }

    const systemList = line.match(/^(SYSTEM OBJECTS|UNAVAILABLE SERVER OBJECTS):\s*(.+)$/);
    if (systemList) {
      const unavailable = systemList[1].startsWith("UNAVAILABLE");
      for (const name of splitTop(systemList[2])) {
        objects.push({ kind: "system", schemaName: null, objectName: name, namesOnly: true, unavailable, systemVersioned: false, synonymTarget: null, annotations: [], sourceLine: lineNo, raw: line, columns: [] });
      }
      continue;
    }

    // Bare schema-qualified shape lines (sys.*, INFORMATION_SCHEMA.*, Fabric
    // queryinsights.*, ...) — object-kind keyword lines never reach here.
    const systemShape = line.match(/^([A-Za-z_][\w]*\.[^\s(]+)\s*\((.+)\)$/);
    if (systemShape) {
      const columns: ParsedColumn[] = [];
      for (const entry of splitTop(systemShape[2])) {
        const parsed = parseColumnEntry(entry);
        if (!("annotation" in parsed)) columns.push(parsed);
      }
      objects.push({ kind: "system", schemaName: null, objectName: systemShape[1], namesOnly: false, unavailable: false, systemVersioned: false, synonymTarget: null, annotations: [], sourceLine: lineNo, raw: line, columns });
      continue;
    }

    const extraKind = line.startsWith("FULLTEXT INDEX:") ? "fulltext_index"
      : line.startsWith("CHANGE TRACKING:") ? "change_tracking"
      : /^INDEX\b/.test(line) ? "index" : "unparsed";
    extras.push({ kind: extraKind, line: lineNo, raw: line });
  }
  return { headers, objects, extras };
}

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const pool = await new sql.ConnectionPool({ ...config.databases.admin, database: config.databases.controlName }).connect();

const snapshotIds = (await pool.request().query<{ catalog_snapshot_id: string; n: number }>(
  "SELECT catalog_snapshot_id, COUNT(*) n FROM dataset.cases GROUP BY catalog_snapshot_id")).recordset;
const toFile = (snapshotId: string) => snapshotId.replace(/-(v2|legacy-v1\.1\.0)$/, "");

const failures: Array<Record<string, unknown>> = [];
const perSnapshot: Array<Record<string, unknown>> = [];

const writer = new sql.Transaction(pool);
await writer.begin();
try {
  for (const table of ["catalog.columns", "catalog.objects", "catalog.extras", "catalog.snapshots"]) {
    await new sql.Request(writer).query(`DELETE FROM ${table}`);
  }
  for (const { catalog_snapshot_id: snapshotId, n } of snapshotIds) {
    const base = toFile(snapshotId);
    if (base === snapshotId) failures.push({ gate: "snapshot-id-suffix", snapshotId });
    let text: string;
    try {
      text = readFileSync(`${CATALOG_DIR}/${base}.txt`, "utf8");
    } catch {
      failures.push({ gate: "catalog-file-missing", snapshotId, file: `${base}.txt` });
      continue;
    }
    const parsed = parseCatalog(text);
    const sha = createHash("sha256").update(text, "utf8").digest("hex");
    await new sql.Request(writer)
      .input("id", sql.VarChar(120), snapshotId).input("name", sql.VarChar(80), base)
      .input("file", sql.VarChar(200), `${CATALOG_DIR}/${base}.txt`)
      .input("hdr", sql.NVarChar(sql.MAX), JSON.stringify(parsed.headers))
      .input("raw", sql.NVarChar(sql.MAX), text).input("sha", sql.Char(64), sha)
      .input("lines", sql.Int, text.split("\n").length)
      .input("objs", sql.Int, parsed.objects.length).input("extras", sql.Int, parsed.extras.length)
      .query(`INSERT catalog.snapshots(snapshot_id,catalog_name,source_file,header_json,raw_text,raw_sha256,line_count,object_count,extra_line_count)
              VALUES(@id,@name,@file,@hdr,@raw,@sha,@lines,@objs,@extras)`);
    for (const object of parsed.objects) {
      const inserted = await new sql.Request(writer)
        .input("sid", sql.VarChar(120), snapshotId).input("kind", sql.VarChar(30), object.kind)
        .input("schema", sql.NVarChar(128), object.schemaName).input("name", sql.NVarChar(256), object.objectName)
        .input("namesOnly", sql.Bit, object.namesOnly).input("unavail", sql.Bit, object.unavailable)
        .input("sysver", sql.Bit, object.systemVersioned).input("syn", sql.NVarChar(400), object.synonymTarget)
        .input("ann", sql.NVarChar(sql.MAX), object.annotations.length > 0 ? JSON.stringify(object.annotations) : null)
        .input("line", sql.Int, object.sourceLine).input("raw", sql.NVarChar(sql.MAX), object.raw)
        .query(`INSERT catalog.objects(snapshot_id,object_kind,schema_name,object_name,names_only,unavailable,is_system_versioned,synonym_target,annotations_json,source_line,raw_def)
                OUTPUT INSERTED.object_pk VALUES(@sid,@kind,@schema,@name,@namesOnly,@unavail,@sysver,@syn,@ann,@line,@raw)`);
      const objectPk = inserted.recordset[0].object_pk as number;
      for (let ordinal = 0; ordinal < object.columns.length; ordinal += 1) {
        const column = object.columns[ordinal];
        await new sql.Request(writer)
          .input("opk", sql.Int, objectPk).input("ord", sql.Int, ordinal + 1)
          .input("name", sql.NVarChar(256), column.name).input("param", sql.Bit, column.isParameter)
          .input("type", sql.NVarChar(200), column.type).input("nul", sql.VarChar(10), column.nullability)
          .input("pk", sql.Bit, column.isPk).input("fk", sql.NVarChar(400), column.fkTarget)
          .input("gen", sql.Bit, column.isGenerated).input("raw", sql.NVarChar(sql.MAX), column.raw)
          .query(`INSERT catalog.columns(object_pk,ordinal,column_name,is_parameter,type_name,nullability,is_pk,fk_target,is_generated,raw_entry)
                  VALUES(@opk,@ord,@name,@param,@type,@nul,@pk,@fk,@gen,@raw)`);
      }
    }
    for (const extra of parsed.extras) {
      await new sql.Request(writer)
        .input("sid", sql.VarChar(120), snapshotId).input("kind", sql.VarChar(40), extra.kind)
        .input("line", sql.Int, extra.line).input("raw", sql.NVarChar(sql.MAX), extra.raw)
        .query("INSERT catalog.extras(snapshot_id,kind,source_line,raw_line) VALUES(@sid,@kind,@line,@raw)");
    }
    if (parsed.objects.length === 0) failures.push({ gate: "empty-parse", snapshotId });
    const unparsed = parsed.extras.filter((e) => e.kind === "unparsed");
    perSnapshot.push({
      snapshotId, cases: n, objects: parsed.objects.length,
      columns: parsed.objects.reduce((sum, o) => sum + o.columns.length, 0),
      extras: parsed.extras.length, unparsed: unparsed.length,
      unparsedSamples: unparsed.slice(0, 3).map((e) => e.raw.slice(0, 80)),
    });
  }
  await writer.commit();
} catch (error) {
  await writer.rollback();
  throw error;
}

const summary = { schemaVersion: 1, stage: "GT-2", snapshots: perSnapshot, failures };
await atomicWrite(`${runDirectory}/manifests/catalog-snapshot.json`, `${JSON.stringify(summary, null, 2)}\n`);
await pool.request()
  .input("key", sql.VarChar(160), `gt2-catalog-snapshot:${hashJson(summary).slice(0, 16)}`)
  .input("run", sql.VarChar(120), runDirectory.split("/").at(-1))
  .input("stage", sql.VarChar(40), "GT-2")
  .input("disp", sql.VarChar(40), failures.length === 0 ? "PASS" : "FAIL")
  .input("json", sql.NVarChar(sql.MAX), JSON.stringify(summary))
  .input("hash", sql.Char(64), hashJson(summary))
  .query(`IF NOT EXISTS (SELECT 1 FROM control.evidence_events WHERE event_key=@key)
      INSERT control.evidence_events(event_key,run_id,stage,disposition,detail_json,detail_sha256) VALUES(@key,@run,@stage,@disp,@json,@hash)`);
console.log(JSON.stringify({ snapshots: perSnapshot.length, failures: failures.length, perSnapshot }, null, 2));
if (failures.length > 0) process.exitCode = 2;
await pool.close();
