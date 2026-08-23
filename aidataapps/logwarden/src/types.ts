export type SplitRole =
  | "dev"
  | "calibration"
  | "test_id"
  | "test_variant_holdout"
  | "test_unknown"
  | "test_live_parity"
  | "test_storm";

export type SourceMode = "replay" | "live";

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

export const deterministicDecode = (maxTokens = 900): DecodeConfig => ({
  key: "det",
  temperature: 0,
  top_p: 1,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 0,
  frequency_penalty: 0,
  seed: 0,
  max_tokens: maxTokens,
  n: 1,
  stop: [],
});

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}
