import { performance } from "node:perf_hooks";
import type { ModelProfile } from "./models.js";
import type { ChatMessage, DecodeConfig, GenerationResult } from "./types.js";

interface ChatChoice {
  finish_reason?: string | null;
  message?: {
    content?: string | null;
    reasoning?: string | null;
    reasoning_content?: string | null;
  };
}

interface ChatResponse {
  choices?: ChatChoice[];
  usage?: { prompt_tokens?: number; completion_tokens?: number };
  [key: string]: unknown;
}

interface EmbeddingResponse {
  data?: Array<{ index?: number; embedding?: number[] }>;
}

export function buildChatRequest(profile: ModelProfile, messages: ChatMessage[], decode: DecodeConfig): Record<string, unknown> {
  if (messages.length !== 1 || messages[0]?.role !== "user") {
    throw new Error("Primary ModelPrint generations require exactly one user message");
  }
  return {
    model: profile.key,
    messages,
    temperature: decode.temperature,
    top_p: decode.top_p,
    top_k: decode.top_k,
    min_p: decode.min_p,
    repetition_penalty: decode.repetition_penalty,
    presence_penalty: decode.presence_penalty,
    frequency_penalty: decode.frequency_penalty,
    seed: decode.seed,
    max_tokens: decode.max_tokens,
    n: decode.n,
    stop: decode.stop,
    chat_template_kwargs: profile.chatTemplateKwargs,
  };
}

export class ExternalServiceError extends Error {
  constructor(message: string, readonly status?: number, readonly detail?: string) {
    super(message);
    this.name = "ExternalServiceError";
  }
}

export class VllmGateway {
  constructor(private readonly timeoutMs = 900_000) {}

  async completeMessages(
    baseUrl: string,
    profile: ModelProfile,
    messages: ChatMessage[],
    decode: DecodeConfig,
  ): Promise<GenerationResult> {
    const request = buildChatRequest(profile, messages, decode);
    const started = performance.now();
    const { body, status } = await this.post<ChatResponse>(`${baseUrl}/chat/completions`, request);
    const choice = body.choices?.[0];
    const finalText = choice?.message?.content?.trim() ?? "";
    const reasoningText =
      choice?.message?.reasoning_content?.trim() || choice?.message?.reasoning?.trim() || null;
    return {
      request,
      rawResponse: body,
      finalText,
      reasoningText,
      inputTokens: body.usage?.prompt_tokens ?? null,
      outputTokens: body.usage?.completion_tokens ?? null,
      finishReason: choice?.finish_reason ?? null,
      httpStatus: status,
      latencyMs: Number((performance.now() - started).toFixed(2)),
    };
  }

  async embed(baseUrl: string, model: string, dimensions: number, input: string[]): Promise<number[][]> {
    if (input.length === 0) return [];
    const { body } = await this.post<EmbeddingResponse>(`${baseUrl}/embeddings`, { model, input });
    const ordered = [...(body.data ?? [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    if (ordered.length !== input.length) {
      throw new ExternalServiceError(`Expected ${input.length} embeddings, received ${ordered.length}`);
    }
    return ordered.map((row, index) => {
      if (!row.embedding || row.embedding.length !== dimensions) {
        throw new ExternalServiceError(`Embedding ${index} has ${row.embedding?.length ?? 0} dimensions; expected ${dimensions}`);
      }
      if (!row.embedding.every(Number.isFinite)) throw new ExternalServiceError(`Embedding ${index} is non-finite`);
      return row.embedding;
    });
  }

  async tokenize(baseUrl: string, model: string, messages: ChatMessage[]): Promise<Record<string, unknown>> {
    const root = baseUrl.replace(/\/v1$/, "");
    const { body } = await this.post<Record<string, unknown>>(`${root}/tokenize`, {
      model,
      messages,
      add_generation_prompt: true,
    });
    return body;
  }

  async ready(baseUrl: string): Promise<void> {
    const response = await fetch(`${baseUrl}/models`, { signal: AbortSignal.timeout(this.timeoutMs) });
    if (!response.ok) throw new ExternalServiceError(`${baseUrl}/models returned ${response.status}`, response.status);
  }

  private async post<T>(url: string, request: unknown): Promise<{ body: T; status: number }> {
    const response = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    const text = await response.text();
    if (!response.ok) {
      throw new ExternalServiceError(`${url} returned HTTP ${response.status}`, response.status, text.slice(0, 4000));
    }
    try {
      return { body: JSON.parse(text) as T, status: response.status };
    } catch {
      throw new ExternalServiceError(`${url} returned invalid JSON`, response.status, text.slice(0, 4000));
    }
  }
}
