import { existsSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

export const LAB_ROOT = fileURLToPath(new URL("../", import.meta.url)).replace(/\/$/, "");
export const RUNS_ROOT = `${LAB_ROOT}/runs`;

export function valueAfter(flag: string, argv = process.argv): string | undefined {
  const index = argv.indexOf(flag);
  return index >= 0 ? argv[index + 1] : undefined;
}

export function resolveRunDirectory(argv = process.argv): string {
  const explicit = valueAfter("--run", argv) ?? process.env.MODELPRINT_RUN_DIR;
  const recorded = existsSync(`${LAB_ROOT}/.current-run`) ? readFileSync(`${LAB_ROOT}/.current-run`, "utf8").trim() : undefined;
  const selected = explicit || recorded;
  if (!selected) throw new Error("No run selected. Run npm run run:init -- --campaign full or pass --run.");
  const absolute = selected.startsWith("/") ? selected : `${LAB_ROOT}/${selected.replace(/^\.\//, "")}`;
  const normalized = absolute.replace(/\/$/, "");
  if (!normalized.startsWith(`${RUNS_ROOT}/`)) throw new Error(`Run directory must be beneath ${RUNS_ROOT}`);
  return normalized;
}

export async function ensureRunLayout(runDirectory: string): Promise<void> {
  for (const name of ["environment", "manifests", "raw", "tables", "metrics", "figures", "reports", "database", "logs", "checkpoints"]) {
    await mkdir(`${runDirectory}/${name}`, { recursive: true });
  }
}

export async function appendExperimentLog(message: string): Promise<void> {
  const path = `${LAB_ROOT}/EXPERIMENT_LOG.md`;
  const line = `\n- ${new Date().toISOString()} — ${message}\n`;
  const current = existsSync(path) ? readFileSync(path, "utf8") : "# ModelPrint Experiment Log\n\nAppend-only operator record.\n";
  await writeFile(path, current + line);
}
