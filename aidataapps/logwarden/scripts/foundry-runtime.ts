import { execFileSync, spawn } from "node:child_process";

// Shared Azure Foundry Local plumbing for the mac profile (docs/MAC_PROFILE.md).
// Used by mac-model.ts (serve + canary) and mac-eval.ts (model comparison).

export const chatPort = Number(process.env.CHAT_PORT ?? 8010);
export const openAiBaseUrl = `http://127.0.0.1:${chatPort}/v1`;

export interface ServerStatus {
  running: boolean;
  state: string;
  pid?: number;
  webUrls?: string[];
}

export function foundry(args: string[], timeout = 600_000): string {
  return execFileSync("foundry", args, { encoding: "utf8", timeout });
}

export function serverStatus(): ServerStatus {
  try {
    return JSON.parse(foundry(["server", "status", "-o", "json"], 30_000)) as ServerStatus;
  } catch {
    return { running: false, state: "unknown" };
  }
}

const onPinnedPort = (candidate: ServerStatus) =>
  (candidate.webUrls ?? []).some((url) => url.endsWith(`:${chatPort}`));

export async function ensureServer(): Promise<ServerStatus> {
  let status = serverStatus();
  if (status.running && status.state === "ready" && onPinnedPort(status)) return status;
  if (status.running) {
    console.log(`Foundry daemon is on ${status.webUrls?.join(", ")}; restarting on :${chatPort}`);
    spawn("foundry", ["server", "restart", "-p", String(chatPort)], { detached: true, stdio: "ignore" }).unref();
  } else {
    console.log(`Starting Foundry daemon on :${chatPort}`);
    spawn("foundry", ["server", "start", "-p", String(chatPort)], { detached: true, stdio: "ignore" }).unref();
  }
  for (let attempt = 0; attempt < 90; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 2_000));
    status = serverStatus();
    if (status.running && status.state === "ready" && onPinnedPort(status)) return status;
  }
  throw new Error(`Foundry daemon did not become ready on :${chatPort}`);
}

export async function restartServer(): Promise<ServerStatus> {
  spawn("foundry", ["server", "restart", "-p", String(chatPort)], { detached: true, stdio: "ignore" }).unref();
  for (let attempt = 0; attempt < 90; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 2_000));
    const status = serverStatus();
    if (status.running && status.state === "ready" && onPinnedPort(status)) return status;
  }
  throw new Error(`Foundry daemon did not come back on :${chatPort}`);
}

export function resolveVariantId(alias: string, pinned: string): string {
  try {
    const info = JSON.parse(foundry(["model", "info", alias, "-o", "json"], 60_000)) as {
      model: { id: string };
    };
    if (pinned && info.model.id !== pinned) {
      console.warn(`Pinned variant ${pinned} differs from catalog default ${info.model.id}; using the pin.`);
      return pinned;
    }
    return pinned || info.model.id;
  } catch {
    // Custom-cache models (e.g. the Muse bundle) are absent from the catalog.
    if (!pinned) throw new Error(`No catalog entry and no pinned variant for ${alias}`);
    return pinned;
  }
}

export function cacheLocation(): string {
  const raw = foundry(["cache", "location"], 30_000);
  const match = /(\/[^\s]+)/u.exec(raw);
  if (!match?.[1]) throw new Error(`Cannot parse cache location from: ${raw}`);
  return match[1];
}

export function setCacheLocation(path: string): void {
  // --force: the confirmation prompt cannot be answered from a script.
  foundry(["cache", "cd", path, "--force"], 60_000);
}

export function loadModelTimed(alias: string): number {
  const startedAt = Date.now();
  foundry(["model", "load", alias], 1_800_000);
  return Date.now() - startedAt;
}

export function unloadModel(alias: string): void {
  try {
    foundry(["model", "unload", alias], 300_000);
  } catch {
    /* not loaded */
  }
}
