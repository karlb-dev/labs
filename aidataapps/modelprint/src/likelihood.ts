export interface TokenLogprobCandidate { logprob?: number; decoded_token?: string }

export interface AssistantSpanSlice {
  values: number[];
  tokenCount: number;
  assistantCharStart: number;
  assistantCharEnd: number;
  firstTokenIndex: number;
  reconstructedPrompt: string;
  alignment: "exact" | "unicode-byte-fallback";
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function locateAssistantText(reconstructed: string, assistantText: string): { start: number; end: number; alignment: AssistantSpanSlice["alignment"] } | null {
  const exactStart = reconstructed.lastIndexOf(assistantText);
  if (exactStart >= 0) return { start: exactStart, end: exactStart + assistantText.length, alignment: "exact" };

  // vLLM exposes each token's decoded_token independently. Byte-fallback
  // pieces for a non-ASCII code point can therefore appear as one or more
  // U+FFFD characters even though decoding the complete token sequence is
  // lossless. Keep every ASCII character exact and allow that substitution
  // only where the governed assistant text itself contains non-ASCII.
  const characters = Array.from(assistantText); const parts: string[] = [];
  for (let index = 0; index < characters.length; index += 1) {
    const character = characters[index]!; const literal = escapeRegex(character);
    if (/^[\x00-\x7F]$/.test(character)) { parts.push(literal); continue; }
    if (characters[index - 1] === " ") {
      parts.pop();
      parts.push(`(?: ${literal}| ?\\uFFFD+)`);
    } else parts.push(`(?:${literal}|\\uFFFD+)`);
  }
  const pattern = parts.join("");
  const expression = new RegExp(pattern, "gu");
  let match: RegExpExecArray | null; let last: RegExpExecArray | null = null;
  while ((match = expression.exec(reconstructed)) !== null) last = match;
  return last ? { start: last.index, end: last.index + last[0].length, alignment: "unicode-byte-fallback" } : null;
}

/** Slice by decoded overlap because the first assistant token may merge with the chat-template junction. */
export function sliceAssistantLogprobs(
  tokenIds: number[],
  promptLogprobs: Array<Record<string, TokenLogprobCandidate> | null>,
  assistantText: string,
): AssistantSpanSlice | null {
  const rows: Array<{ index: number; start: number; end: number; value: number | undefined; decoded: string }> = [];
  let reconstructed = "";
  for (let index = 0; index < Math.min(tokenIds.length, promptLogprobs.length); index += 1) {
    const candidates = promptLogprobs[index];
    const actual = candidates?.[String(tokenIds[index])] ?? (candidates ? Object.values(candidates).find((row) => row.decoded_token !== undefined) : undefined);
    const decoded = actual?.decoded_token ?? ""; const start = reconstructed.length; reconstructed += decoded;
    rows.push({ index, start, end: reconstructed.length, value: actual?.logprob, decoded });
  }
  const location = locateAssistantText(reconstructed, assistantText);
  if (!location) return null;
  const assistantCharStart = location.start;
  const assistantCharEnd = location.end;
  const selected = rows.filter((row) => row.end > assistantCharStart && row.start < assistantCharEnd
    && Number.isFinite(row.value) && !/<\|(?:im_end|eot_id|end_of_turn)[^>]*\|>|<end_of_turn>/i.test(row.decoded));
  if (!selected.length) return null;
  return { values: selected.map((row) => row.value!), tokenCount: selected.length, assistantCharStart, assistantCharEnd,
    firstTokenIndex: selected[0]!.index, reconstructedPrompt: reconstructed, alignment: location.alignment };
}
