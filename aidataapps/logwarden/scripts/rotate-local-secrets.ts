import { randomBytes } from "node:crypto";
import { readFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { LAB_ROOT, atomicWrite } from "../src/run.js";
import { withPool } from "../src/repository.js";

const envPath = `${LAB_ROOT}/.env`;
const config = loadConfig();
const newSaPassword = `LogWarden!${randomBytes(24).toString("hex")}`;
const newLabPassword = `LwLab!${randomBytes(24).toString("hex")}`;
const newAgentPassword = `LwAgent!${randomBytes(24).toString("hex")}`;

await withPool(config.databases.admin, "master", async (pool) => {
  await pool.request()
    .input("new_password", sql.NVarChar(128), newSaPassword)
    .query(`
      DECLARE @statement nvarchar(max) =
        N'ALTER LOGIN [sa] WITH PASSWORD = ' + QUOTENAME(@new_password, N'''');
      EXEC sys.sp_executesql @statement;
    `);
});

let content = await readFile(envPath, "utf8");
const replacements: Record<string, string> = {
  MSSQL_SA_PASSWORD: newSaPassword,
  LW_LAB_PASSWORD: newLabPassword,
  LW_AGENT_PASSWORD: newAgentPassword,
};
for (const [key, value] of Object.entries(replacements)) {
  const pattern = new RegExp(`^${key}=.*$`, "m");
  if (!pattern.test(content)) throw new Error(`Missing ${key} in ${envPath}`);
  content = content.replace(pattern, `${key}=${value}`);
}
await atomicWrite(envPath, content, 0o600);
console.log("Rotated local SA/lab/agent credentials; recreate SQL Server so its health check receives the new SA value.");
