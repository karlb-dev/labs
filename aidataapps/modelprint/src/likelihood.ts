export interface TokenLogprobCandidate { logprob?: number; decoded_token?: string }

export interface AssistantSpanSlice {
  values: number[];
  tokenCount: number;
  assistantCharStart: number;
  assistantCharEnd: number;
  firstTokenIndex: number;
  reconstructedPrompt: string;
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
  const assistantCharStart = reconstructed.lastIndexOf(assistantText);
  if (assistantCharStart < 0) return null;
  const assistantCharEnd = assistantCharStart + assistantText.length;
  const selected = rows.filter((row) => row.end > assistantCharStart && row.start < assistantCharEnd
    && Number.isFinite(row.value) && !/<\|(?:im_end|eot_id|end_of_turn)[^>]*\|>|<end_of_turn>/i.test(row.decoded));
  if (!selected.length) return null;
  return { values: selected.map((row) => row.value!), tokenCount: selected.length, assistantCharStart, assistantCharEnd,
    firstTokenIndex: selected[0]!.index, reconstructedPrompt: reconstructed };
}
