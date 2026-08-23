import { VllmGateway } from "../src/inference.js";

export interface EmbeddingCompatibilityProfile {
  baseUrl: string;
  modelId: string;
  dimensions: number;
  truncatePromptTokens?: number | undefined;
}

export function repairUnpairedSurrogates(value: string): { text: string; replacedCodeUnits: number } {
  let text = ""; let replacedCodeUnits = 0;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) { text += value[index]! + value[index + 1]!; index += 1; }
      else { text += "\ufffd"; replacedCodeUnits += 1; }
    } else if (code >= 0xdc00 && code <= 0xdfff) { text += "\ufffd"; replacedCodeUnits += 1; }
    else text += value[index]!;
  }
  return { text, replacedCodeUnits };
}

export async function embedWithCompatibility(gateway: VllmGateway, profile: EmbeddingCompatibilityProfile, input: string[]): Promise<{
  vectors: number[][];
  repairedInputs: Array<{ index: number; replacedCodeUnits: number }>;
}> {
  const repairedInputs: Array<{ index: number; replacedCodeUnits: number }> = [];
  const normalized = input.map((value, index) => {
    const repaired = repairUnpairedSurrogates(value);
    if (repaired.replacedCodeUnits) repairedInputs.push({ index, replacedCodeUnits: repaired.replacedCodeUnits });
    return repaired.text;
  });
  if (!profile.truncatePromptTokens) {
    return { vectors: await gateway.embed(profile.baseUrl, profile.modelId, profile.dimensions, normalized), repairedInputs };
  }
  const response = await fetch(`${profile.baseUrl}/embeddings`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: profile.modelId, input: normalized, truncate_prompt_tokens: profile.truncatePromptTokens, truncation_side: "right" }),
    signal: AbortSignal.timeout(900_000),
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`${profile.baseUrl}/embeddings returned HTTP ${response.status}: ${text.slice(0, 4000)}`);
  const body = JSON.parse(text) as { data?: Array<{ index?: number; embedding?: number[] }> };
  const ordered = [...(body.data ?? [])].sort((left, right) => (left.index ?? 0) - (right.index ?? 0));
  if (ordered.length !== normalized.length) throw new Error(`Expected ${normalized.length} embeddings, received ${ordered.length}`);
  const vectors = ordered.map((row, index) => {
    if (!row.embedding || row.embedding.length !== profile.dimensions || !row.embedding.every(Number.isFinite)) {
      throw new Error(`Embedding ${index} is invalid; expected ${profile.dimensions} finite dimensions`);
    }
    return row.embedding;
  });
  return { vectors, repairedInputs };
}
