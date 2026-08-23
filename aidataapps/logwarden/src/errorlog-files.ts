import { execFile } from "node:child_process";
import type { ErrorlogFile } from "./errorlog.js";

export async function readErrorlogFilesFromContainer(container: string): Promise<ErrorlogFile[]> {
  const listing = await runDocker(container, ["sh", "-lc",
    "ls -1 /var/opt/mssql/log/errorlog /var/opt/mssql/log/errorlog.[0-9]* 2>/dev/null || true"]);
  const paths = [...new Set(listing.split("\n").map((line) => line.trim()).filter(Boolean))];
  const files: ErrorlogFile[] = [];
  for (const path of paths) files.push({ path, content: await runDocker(container, ["cat", path]) });
  if (files.length === 0) throw new Error("No SQL Server ERRORLOG files were readable through the container");
  return files;
}

export async function runDocker(container: string, command: string[], retries = 3): Promise<string> {
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= retries; attempt += 1) {
    try {
      return await new Promise<string>((resolve, reject) => {
        execFile("docker", ["exec", container, ...command], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 }, (error, stdout, stderr) => {
          if (error !== null) reject(new Error(`docker exec failed: ${stderr.trim() || error.message}`));
          else resolve(stdout);
        });
      });
    } catch (error) {
      lastError = error as Error;
      if (attempt < retries) await new Promise((resolve) => setTimeout(resolve, 250 * attempt));
    }
  }
  throw lastError ?? new Error("docker exec failed");
}
