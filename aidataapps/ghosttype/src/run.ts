import { existsSync, readFileSync } from "node:fs";
import { appendFile, mkdir, rename, writeFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const LAB_ROOT = fileURLToPath(new URL("../", import.meta.url)).replace(/\/$/, "");
export const RUNS_ROOT = `${LAB_ROOT}/runs`;

export function valueAfter(flag: string, argv = process.argv): string | undefined {
  const index = argv.indexOf(flag);
  return index >= 0 ? argv[index + 1] : undefined;
}

export function assertRunDirectory(path: string): string {
  const absolute = resolve(path);
  const relativePath = relative(resolve(RUNS_ROOT), absolute);
  if (!relativePath || relativePath.startsWith("..") || relativePath.startsWith("/")) {
    throw new Error(`Run directory must be a child of ${RUNS_ROOT}: ${absolute}`);
  }
  return absolute;
}

export function resolveRunDirectory(argv = process.argv): string {
  const explicit = valueAfter("--run", argv) ?? process.env.LOGWARDEN_RUN_DIR;
  const recorded = existsSync(`${LAB_ROOT}/.current-run`)
    ? readFileSync(`${LAB_ROOT}/.current-run`, "utf8").trim()
    : undefined;
  const selected = explicit || recorded;
  if (!selected) {
    throw new Error("No run selected. Run npm run run:init -- --campaign smoke or pass --run.");
  }
  return assertRunDirectory(selected.startsWith("/") ? selected : `${LAB_ROOT}/${selected}`);
}

export async function ensureRunLayout(runDirectory: string): Promise<void> {
  for (const name of [
    "environment",
    "manifests",
    "raw",
    "tables",
    "metrics",
    "figures",
    "reports",
    "database",
    "logs",
    "telemetry",
    "repro",
    "checkpoints",
  ]) {
    await mkdir(`${runDirectory}/${name}`, { recursive: true });
  }
}

export async function atomicWrite(path: string, content: string | Buffer, mode = 0o644): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporary, content, { mode });
  await rename(temporary, path);
}

export async function appendExperimentLog(message: string): Promise<void> {
  await appendFile(`${LAB_ROOT}/EXPERIMENT_LOG.md`, `\n- ${new Date().toISOString()} — ${message}\n`);
}
