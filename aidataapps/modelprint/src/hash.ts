import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

export function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

export function canonicalJson(value: unknown): string {
  return JSON.stringify(sortValue(value));
}

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, child]) => [key, sortValue(child)]),
    );
  }
  return value;
}

export function hashJson(value: unknown): string {
  return sha256(canonicalJson(value));
}

export async function hashFile(path: string | URL): Promise<string> {
  return sha256(await readFile(path));
}
