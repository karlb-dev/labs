import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { hashJson, sha256 } from "./hash.js";
import type { ModelPrintSplit } from "./types.js";

const carrierSchema = z.object({
  maxTokens: z.number().int().positive(),
  heldOut: z.boolean().optional(),
  evaluationOnly: z.boolean().optional(),
  template: z.string(),
});

const carriersSchema = z.object({
  schemaVersion: z.literal(1),
  primaryMessagePolicy: z.literal("single-user-turn-v1"),
  carriers: z.record(z.string(), carrierSchema),
});

export type Carrier = z.infer<typeof carrierSchema> & { key: string };

const defaultCarriersPath = fileURLToPath(new URL("../config/carriers.json", import.meta.url));

export function loadCarriers(path = defaultCarriersPath): Map<string, Carrier> {
  const parsed = carriersSchema.parse(JSON.parse(readFileSync(path, "utf8")));
  return new Map(Object.entries(parsed.carriers).map(([key, value]) => [key, { key, ...value }]));
}

export function renderCarrier(carrier: Carrier, content: string, context = ""): string {
  const rendered = carrier.template
    .replaceAll("{{content}}", content.trim())
    .replaceAll("{{context}}", context.trim())
    .replace(/\r\n/g, "\n")
    .trim();
  if (/^\s*assistant\s*:/im.test(rendered)) {
    throw new Error(`Carrier ${carrier.key} contains "assistant" as a role label`);
  }
  if (scanSelfName(rendered).selfNameFound) {
    throw new Error(`Carrier ${carrier.key} contains a target model or vendor name`);
  }
  return rendered;
}

export function relationCompletionToQuestion(prompt: string): string {
  const text = prompt.trim().replace(/[.\s]+$/, "");
  const theMatch = /^The ([\w -]+) of (.+?) is$/i.exec(text);
  if (theMatch) return `What is the ${theMatch[1]} of ${theMatch[2]}?`;
  const possessiveMatch = /^(.+?)'s ([\w -]+) is$/i.exec(text);
  if (possessiveMatch) return `What is ${possessiveMatch[1]}'s ${possessiveMatch[2]}?`;
  const locatedMatch = /^(.+?) is located in$/i.exec(text);
  if (locatedMatch) return `Where is ${locatedMatch[1]} located?`;
  const isMatch = /^(.+?) is$/i.exec(text);
  if (isMatch) return `What correctly completes this statement: “${isMatch[1]} is …”?`;
  return `Answer the question implied by this completion-style prompt, then explain briefly: “${text} …”`;
}

const residuePatterns: Array<[string, RegExp]> = [
  ["special-token", /<\|[^>]+\|>/i],
  ["think-tag", /<\/?think>/i],
  ["reasoning-tag", /<\/?reasoning>/i],
  ["tool-call", /<\/?tool_call>/i],
  ["role-label", /^(?:assistant|system|user)\s*:/im],
];

const modelNames = [
  "Muse Glimmer", "Muse-Glimmer", "Glimmer", "Gemma", "OLMo", "Qwen",
  "meta-models", "Google", "AllenAI", "Alibaba", "Hugging Face", "vLLM",
];

export function scanTemplateResidue(text: string) {
  const matches = residuePatterns.filter(([, pattern]) => pattern.test(text)).map(([name]) => name);
  return { templateResidueFound: matches.length > 0, residueKinds: matches };
}

export function scanSelfName(text: string) {
  const names = modelNames.filter((name) => new RegExp(`\\b${escapeRegex(name)}\\b`, "i").test(text));
  return { selfNameFound: names.length > 0, names };
}

export function maskNames(text: string): string {
  let masked = text;
  for (const name of [...modelNames].sort((a, b) => b.length - a.length)) {
    const replacement = ["Google", "AllenAI", "Alibaba", "meta-models", "Hugging Face", "vLLM"].includes(name)
      ? "[VENDOR]"
      : "[MODEL]";
    masked = masked.replace(new RegExp(`\\b${escapeRegex(name)}\\b`, "gi"), replacement);
  }
  return masked;
}

export function stripTemplateResidue(text: string): string {
  return text
    .replace(/<\|[^>]+\|>/g, "")
    .replace(/<\/?(?:think|reasoning|tool_call)>/gi, "")
    .replace(/^(?:assistant|system|user)\s*:\s*/gim, "")
    .trim();
}

export function normalizeText(text: string): string {
  return text.normalize("NFC").replace(/\r\n/g, "\n").trim();
}

export function assignSplit(promptGroupId: string, sourceId: string): ModelPrintSplit {
  if (sourceId === "oasst1") return "test_source_holdout";
  const value = Number.parseInt(sha256(promptGroupId).slice(0, 8), 16) / 0x1_0000_0000;
  if (value < 0.68) return "train";
  if (value < 0.80) return "calibration";
  return "test_id";
}

export function promptVariantId(promptGroupId: string, carrierId: string, renderedText: string): string {
  return `pv_${hashJson({ promptGroupId, carrierId, renderedText }).slice(0, 24)}`;
}

export function charShingles(text: string, width = 5): Set<string> {
  const normalized = normalizeText(text).toLowerCase().replace(/\s+/g, " ");
  if (normalized.length <= width) return new Set([normalized]);
  const result = new Set<string>();
  for (let i = 0; i <= normalized.length - width; i += 1) result.add(normalized.slice(i, i + width));
  return result;
}

export function jaccard(left: Set<string>, right: Set<string>): number {
  if (left.size === 0 && right.size === 0) return 1;
  let intersection = 0;
  for (const item of left) if (right.has(item)) intersection += 1;
  return intersection / (left.size + right.size - intersection);
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
