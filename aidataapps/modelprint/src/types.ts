export type ModelPrintSplit =
  | "train"
  | "calibration"
  | "test_id"
  | "test_source_holdout";

export type Decision =
  | "attributed"
  | "ambiguous"
  | "unknown_or_out_of_distribution"
  | "insufficient_text"
  | "unsupported_language";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface DecodeConfig {
  key: string;
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repetition_penalty: number;
  presence_penalty: number;
  frequency_penalty: number;
  seed: number;
  max_tokens: number;
  n: 1;
  stop: string[];
}

export const decodeCell = (
  key: "det" | "nat-0" | "nat-1" | "hv",
  maxTokens: number,
): DecodeConfig => {
  const cells = {
    det: { temperature: 0, top_p: 1, seed: 0 },
    "nat-0": { temperature: 0.7, top_p: 0.9, seed: 0 },
    "nat-1": { temperature: 0.7, top_p: 0.9, seed: 1 },
    hv: { temperature: 1, top_p: 0.95, seed: 2 },
  } as const;
  return {
    key,
    ...cells[key],
    // vLLM 0.27.x uses 0 as the effective disabled value. Earlier API
    // examples accepted -1 but normalized it to 0, which fails the campaign's
    // request-versus-effective-parameters equality gate.
    top_k: 0,
    min_p: 0,
    repetition_penalty: 1,
    presence_penalty: 0,
    frequency_penalty: 0,
    max_tokens: maxTokens,
    n: 1,
    stop: [],
  };
};

export interface PromptGroup {
  promptGroupId: string;
  sourceId: string;
  sourceRowId: string;
  family: string;
  domain: string;
  stratum: string;
  canonicalText: string;
  split: ModelPrintSplit;
  metadata: Record<string, unknown>;
}

export interface PromptVariant {
  promptVariantId: string;
  promptGroupId: string;
  carrierId: string;
  renderedText: string;
  renderSha256: string;
  maxTokens: number;
  evaluationOnly: boolean;
  tierMembership: string[];
  metadata: Record<string, unknown>;
}

export interface GenerationResult {
  request: Record<string, unknown>;
  rawResponse: Record<string, unknown>;
  finalText: string;
  reasoningText: string | null;
  inputTokens: number | null;
  outputTokens: number | null;
  finishReason: string | null;
  httpStatus: number;
  latencyMs: number;
}
