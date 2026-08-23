import { writeFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { valueAfter, resolveRunDirectory } from "../src/run.js";

const left = valueAfter("--left") ?? "ModelPrint";
const right = valueAfter("--right");
for (const value of [left, right]) if (!value || !/^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(value)) throw new Error("--left and --right must be safe database names");

const queries: Record<string, string> = {
  tables: `SELECT s.name schema_name,t.name table_name,t.temporal_type,t.is_memory_optimized,t.durability_desc
    FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id
    WHERE t.is_ms_shipped=0 AND t.name NOT LIKE '\\_\\_mp\\_%' ESCAPE '\\' ORDER BY s.name,t.name`,
  columns: `SELECT s.name schema_name,t.name table_name,c.column_id,c.name column_name,ts.name type_schema,ty.name type_name,
    c.max_length,c.precision,c.scale,c.is_nullable,c.is_identity,c.is_computed,c.collation_name,cc.definition computed_definition
    FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id JOIN sys.columns c ON c.object_id=t.object_id
    JOIN sys.types ty ON ty.user_type_id=c.user_type_id JOIN sys.schemas ts ON ts.schema_id=ty.schema_id
    LEFT JOIN sys.computed_columns cc ON cc.object_id=c.object_id AND cc.column_id=c.column_id
    WHERE t.is_ms_shipped=0 AND t.name NOT LIKE '\\_\\_mp\\_%' ESCAPE '\\' ORDER BY s.name,t.name,c.column_id`,
  indexes: `SELECT s.name schema_name,t.name table_name,
    CASE WHEN LEFT(i.name,4)='PK__' THEN '<AUTO_PRIMARY_KEY>' WHEN LEFT(i.name,4)='UQ__' THEN '<AUTO_UNIQUE>' ELSE i.name END index_name,
    i.type_desc,i.is_unique,i.is_primary_key,i.is_unique_constraint,
    i.has_filter,i.filter_definition,ic.key_ordinal,ic.is_descending_key,ic.is_included_column,c.name column_name
    FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id JOIN sys.indexes i ON i.object_id=t.object_id
    LEFT JOIN sys.index_columns ic ON ic.object_id=i.object_id AND ic.index_id=i.index_id
    LEFT JOIN sys.columns c ON c.object_id=ic.object_id AND c.column_id=ic.column_id
    WHERE t.is_ms_shipped=0 AND t.name NOT LIKE '\\_\\_mp\\_%' ESCAPE '\\' AND i.is_hypothetical=0
    ORDER BY s.name,t.name,i.index_id,ic.index_column_id`,
  foreignKeys: `SELECT ps.name parent_schema,pt.name parent_table,
    CASE WHEN LEFT(fk.name,4)='FK__' THEN '<AUTO_FOREIGN_KEY>' ELSE fk.name END constraint_name,fkc.constraint_column_id,
    pc.name parent_column,rs.name referenced_schema,rt.name referenced_table,rc.name referenced_column,
    fk.delete_referential_action_desc,fk.update_referential_action_desc,fk.is_disabled,fk.is_not_trusted
    FROM sys.foreign_keys fk JOIN sys.foreign_key_columns fkc ON fkc.constraint_object_id=fk.object_id
    JOIN sys.tables pt ON pt.object_id=fk.parent_object_id JOIN sys.schemas ps ON ps.schema_id=pt.schema_id JOIN sys.columns pc ON pc.object_id=pt.object_id AND pc.column_id=fkc.parent_column_id
    JOIN sys.tables rt ON rt.object_id=fk.referenced_object_id JOIN sys.schemas rs ON rs.schema_id=rt.schema_id JOIN sys.columns rc ON rc.object_id=rt.object_id AND rc.column_id=fkc.referenced_column_id
    WHERE pt.name NOT LIKE '\\_\\_mp\\_%' ESCAPE '\\' ORDER BY ps.name,pt.name,fk.name,fkc.constraint_column_id`,
  constraints: `SELECT s.name schema_name,t.name table_name,'CHECK' constraint_kind,
    CASE WHEN LEFT(cc.name,4)='CK__' THEN '<AUTO_CHECK>' ELSE cc.name END constraint_name,cc.definition,cc.is_disabled,cc.is_not_trusted
    FROM sys.check_constraints cc JOIN sys.tables t ON t.object_id=cc.parent_object_id JOIN sys.schemas s ON s.schema_id=t.schema_id
    WHERE t.name NOT LIKE '\\_\\_mp\\_%' ESCAPE '\\'
    UNION ALL SELECT s.name,t.name,'DEFAULT',CASE WHEN LEFT(dc.name,4)='DF__' THEN '<AUTO_DEFAULT>' ELSE dc.name END,dc.definition,CONVERT(bit,0),CONVERT(bit,0)
    FROM sys.default_constraints dc JOIN sys.tables t ON t.object_id=dc.parent_object_id JOIN sys.schemas s ON s.schema_id=t.schema_id
    WHERE t.name NOT LIKE '\\_\\_mp\\_%' ESCAPE '\\' ORDER BY schema_name,table_name,constraint_kind,constraint_name`,
};

const config = loadConfig().database;
async function snapshot(database: string) {
  const pool = await new sql.ConnectionPool({ server: config.server, port: config.port, user: config.user, password: config.password,
    database, options: { encrypt: false, trustServerCertificate: true }, requestTimeout: 600_000 }).connect();
  try {
    const output: Record<string, unknown[]> = {};
    for (const [name, query] of Object.entries(queries)) output[name] = (await pool.request().query(query)).recordset;
    return output;
  } finally { await pool.close(); }
}

const leftSnapshot = await snapshot(left);
const rightSnapshot = await snapshot(right!);
const differences: Record<string, { leftOnly: string[]; rightOnly: string[] }> = {};
for (const name of Object.keys(queries)) {
  const leftRows = new Set((leftSnapshot[name] ?? []).map((row) => JSON.stringify(row)));
  const rightRows = new Set((rightSnapshot[name] ?? []).map((row) => JSON.stringify(row)));
  const leftOnly = [...leftRows].filter((row) => !rightRows.has(row));
  const rightOnly = [...rightRows].filter((row) => !leftRows.has(row));
  if (leftOnly.length || rightOnly.length) differences[name] = { leftOnly, rightOnly };
}
const result = { schemaVersion: 1, checkedAt: new Date().toISOString(), status: Object.keys(differences).length ? "FAIL" : "PASS",
  left, right, sections: Object.fromEntries(Object.entries(leftSnapshot).map(([name, rows]) => [name, rows.length])), differences };
await writeFile(`${resolveRunDirectory()}/environment/database-schema-parity.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(result, null, 2));
if (result.status !== "PASS") process.exitCode = 2;
