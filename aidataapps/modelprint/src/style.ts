import { createHash } from "node:crypto";
import { hashJson } from "./hash.js";

const functionWords = [
  "a", "about", "after", "again", "all", "also", "an", "and", "any", "are", "as", "at", "be", "because",
  "been", "before", "between", "both", "but", "by", "can", "could", "did", "do", "does", "each", "either",
  "for", "from", "had", "has", "have", "he", "her", "here", "hers", "him", "his", "how", "however", "i",
  "if", "in", "into", "is", "it", "its", "may", "might", "more", "most", "not", "of", "on", "one", "or",
  "other", "our", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there",
  "these", "they", "this", "those", "through", "to", "too", "under", "up", "was", "we", "were", "what",
  "when", "where", "which", "while", "who", "will", "with", "would", "you", "your",
] as const;

export const STYLE_SCHEMA = {
  id: "style512-v1",
  seed: 0x4d4f4445,
  blocks: { length: 48, layout: 48, functionWords: 96, charNgrams: 160, wordNgrams: 160 },
  functionWords,
};
export const STYLE_SCHEMA_HASH = hashJson(STYLE_SCHEMA);

export function styleVector(text: string): number[] {
  const normalized = text.normalize("NFC");
  const words = normalized.toLowerCase().match(/[\p{L}\p{N}'’-]+/gu) ?? [];
  const sentences = normalized.split(/(?<=[.!?])\s+/).filter(Boolean);
  const paragraphs = normalized.split(/\n\s*\n/).filter((part) => part.trim());
  const lines = normalized.split("\n");
  const wordCount = Math.max(words.length, 1);
  const charCount = Math.max(normalized.length, 1);
  const sentenceCount = Math.max(sentences.length, 1);
  const paragraphCount = Math.max(paragraphs.length, 1);
  const unique = new Set(words);

  const length = pad([
    Math.log1p(normalized.length), Math.log1p(words.length), Math.log1p(sentences.length), Math.log1p(paragraphs.length),
    words.length / sentenceCount, normalized.length / wordCount, sentences.length / paragraphCount, lines.length / paragraphCount,
    unique.size / wordCount, hapax(words) / wordCount, mean(words.map((w) => w.length)), std(words.map((w) => w.length)),
    mean(sentences.map((s) => s.length)), std(sentences.map((s) => s.length)),
    mean(paragraphs.map((p) => p.length)), std(paragraphs.map((p) => p.length)),
    count(normalized, /\d/g) / charCount, count(normalized, /[A-Z]/g) / charCount,
    count(normalized, /[a-z]/g) / charCount, count(normalized, /\s/g) / charCount,
    count(normalized, /\p{L}/gu) / charCount, count(normalized, /\p{N}/gu) / charCount,
    words.filter((w) => w.length <= 3).length / wordCount, words.filter((w) => w.length >= 8).length / wordCount,
    words.filter((w) => w.length >= 12).length / wordCount, contractions(words) / wordCount,
    count(normalized, /\b(?:I|me|my|mine|we|us|our|ours)\b/gi) / wordCount,
    count(normalized, /\b(?:you|your|yours)\b/gi) / wordCount,
    count(normalized, /\b(?:may|might|could|perhaps|possibly|likely|appears?|seems?)\b/gi) / wordCount,
    count(normalized, /\b(?:certainly|definitely|clearly|obviously|always|never)\b/gi) / wordCount,
    count(normalized, /\b(?:this|it|here)\b/gi) / sentenceCount,
    count(normalized, /\b(?:however|additionally|overall|therefore|moreover|finally)\b/gi) / sentenceCount,
  ], 48);

  const layout = pad([
    count(normalized, /,/g) / charCount, count(normalized, /\./g) / charCount, count(normalized, /;/g) / charCount,
    count(normalized, /:/g) / charCount, count(normalized, /!/g) / charCount, count(normalized, /\?/g) / charCount,
    count(normalized, /—/g) / charCount, count(normalized, /–/g) / charCount, count(normalized, /-/g) / charCount,
    count(normalized, /…/g) / charCount, count(normalized, /\.\.\./g) / charCount,
    count(normalized, /[“”]/g) / charCount, count(normalized, /[‘’]/g) / charCount,
    count(normalized, /"/g) / charCount, count(normalized, /'/g) / charCount,
    count(normalized, /\(/g) / charCount, count(normalized, /\)/g) / charCount,
    count(normalized, /\n/g) / charCount, count(normalized, /\n\s*\n/g) / charCount,
    count(normalized, /^#{1,6}\s/gm) / Math.max(lines.length, 1),
    count(normalized, /^\s*[-*•]\s/gm) / Math.max(lines.length, 1),
    count(normalized, /^\s*\d+[.)]\s/gm) / Math.max(lines.length, 1),
    count(normalized, /\*\*[^*]+\*\*/g) / wordCount, count(normalized, /`{3}/g) / 2,
    count(normalized, /`[^`]+`/g) / wordCount, count(normalized, /```[A-Za-z0-9_+-]+/g),
    count(normalized, /\p{Extended_Pictographic}/gu) / charCount,
    count(normalized, /  +/g) / charCount, count(normalized, /[ \t]+$/gm) / Math.max(lines.length, 1),
    /^(?:Certainly|Sure|Great question)\b/i.test(normalized) ? 1 : 0,
    /(?:Let me know|I hope this helps)[.!]?\s*$/i.test(normalized) ? 1 : 0,
    /^\s*\*\*[A-Za-z][^*]{0,40}:\*\*/m.test(normalized) ? 1 : 0,
  ], 48);

  const frequencies = new Map<string, number>();
  for (const word of words) frequencies.set(word, (frequencies.get(word) ?? 0) + 1);
  const functionBlock = pad([
    ...functionWords.map((word) => (frequencies.get(word) ?? 0) / wordCount),
    count(normalized, /\b(?:and|or|but|yet|nor)\b/gi) / wordCount,
    count(normalized, /\b(?:because|although|while|unless|whereas)\b/gi) / wordCount,
    count(normalized, /\b(?:can|could|may|might|must|shall|should|will|would)\b/gi) / wordCount,
    count(normalized, /\b(?:very|really|quite|rather|extremely)\b/gi) / wordCount,
  ], 96);

  const charBlock = hashedNgrams([...ngrams([...normalized.toLowerCase()], 3, 5)], 160, STYLE_SCHEMA.seed);
  const wordBlock = hashedNgrams([...ngrams(words, 1, 3)].map((gram) => gram.join(" ")), 160, STYLE_SCHEMA.seed ^ 0x9e3779b9);
  const vector = [...length, ...layout, ...functionBlock, ...charBlock, ...wordBlock];
  if (vector.length !== 512 || !vector.every(Number.isFinite)) throw new Error("style512 invariant failed");
  return l2(vector);
}

function pad(values: number[], size: number): number[] {
  return [...values.slice(0, size), ...Array(Math.max(0, size - values.length)).fill(0)];
}
function count(text: string, pattern: RegExp): number { return [...text.matchAll(pattern)].length; }
function mean(values: number[]): number { return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0; }
function std(values: number[]): number { const m = mean(values); return Math.sqrt(mean(values.map((v) => (v - m) ** 2))); }
function hapax(words: string[]): number { const m = new Map<string, number>(); for (const w of words) m.set(w, (m.get(w) ?? 0) + 1); return [...m.values()].filter((v) => v === 1).length; }
function contractions(words: string[]): number { return words.filter((w) => /['’]/.test(w)).length; }
function* ngrams<T>(items: T[], min: number, max: number): Generator<T[]> { for (let n = min; n <= max; n += 1) for (let i = 0; i + n <= items.length; i += 1) yield items.slice(i, i + n); }
function hashedNgrams(items: Iterable<string | string[]>, size: number, seed: number): number[] {
  const result = Array<number>(size).fill(0);
  let total = 0;
  for (const item of items) {
    const value = Array.isArray(item) ? item.join("") : item;
    const digest = createHash("sha256").update(`${seed}:${value}`).digest();
    const bucket = digest.readUInt32LE(0) % size;
    result[bucket] = (result[bucket] ?? 0) + (digest[4]! & 1 ? 1 : -1);
    total += 1;
  }
  return result.map((value) => value / Math.max(total, 1));
}
function l2(values: number[]): number[] { const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)); return norm ? values.map((value) => value / norm) : values; }
