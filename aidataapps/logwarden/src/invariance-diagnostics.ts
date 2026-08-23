import { createHash } from "node:crypto";
import { canonicalJson, hashJson } from "./hash.js";

export interface NormalizedModelResponse {
  choices: unknown;
  usage: unknown;
}

export function normalizeModelResponse(body: Buffer | string): NormalizedModelResponse {
  const parsed = JSON.parse(typeof body === "string" ? body : body.toString("utf8")) as Record<string, unknown>;
  return {
    choices: parsed.choices ?? null,
    usage: parsed.usage ?? null,
  };
}

export function normalizeModelRequest(body: Buffer | string): Record<string, unknown> {
  const parsed = JSON.parse(typeof body === "string" ? body : body.toString("utf8")) as Record<string, unknown>;
  const normalized = structuredClone(parsed);
  const messages = normalized.messages;
  if (Array.isArray(messages)) {
    normalized.messages = messages.map((value) => normalizeMessage(value));
  }
  return normalized;
}

export function normalizedResponseSha256(body: Buffer | string): string {
  return hashJson(normalizeModelResponse(body));
}

export function normalizedRequestSha256(body: Buffer | string): string {
  return hashJson(normalizeModelRequest(body));
}

export function generatedContentHashes(body: Buffer | string): { contentSha256: string; reasoningSha256: string } {
  const parsed = JSON.parse(typeof body === "string" ? body : body.toString("utf8")) as {
    choices?: Array<{ message?: { content?: string | null; reasoning?: string | null; reasoning_content?: string | null } }>;
  };
  const message = parsed.choices?.[0]?.message;
  return {
    contentSha256: textSha256(message?.content ?? ""),
    reasoningSha256: textSha256(message?.reasoning_content ?? message?.reasoning ?? ""),
  };
}

function normalizeMessage(value: unknown): unknown {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return value;
  const message = structuredClone(value) as Record<string, unknown>;
  if (typeof message.content !== "string") return message;
  try {
    const content = JSON.parse(message.content) as unknown;
    message.content = canonicalJson(removeExecutionIdentities(content));
  } catch {
    // Ordinary prompt and assistant text is intentionally preserved byte-for-byte.
  }
  return message;
}

function removeExecutionIdentities(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((entry) => removeExecutionIdentities(entry));
  if (value === null || typeof value !== "object") return value;
  const output: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (key === "callId") output[key] = "<execution-call-id>";
    else if (key === "retrievalRunId") output[key] = "<retrieval-run-id>";
    else output[key] = removeExecutionIdentities(entry);
  }
  return output;
}

function textSha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
