import type { ModelProfile } from "./models.js";
import type { InferenceGateway } from "./types.js";

interface GatewayOptions {
  chatBaseUrl: string;
  embeddingBaseUrl: string;
  embeddingModel: string;
  embeddingDimensions: number;
  model: ModelProfile;
  timeoutMs?: number;
}

interface EmbeddingResponse {
  data?: Array<{ index?: number; embedding?: number[] }>;
}

interface ChatResponse {
  choices?: Array<{ message?: { content?: string | null } }>;
}

export class ExternalServiceError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ExternalServiceError";
  }
}

export class VllmGateway implements InferenceGateway {
  private readonly timeoutMs: number;

  constructor(private readonly options: GatewayOptions) {
    this.timeoutMs = options.timeoutMs ?? 120_000;
  }

  async embed(input: string[]): Promise<number[][]> {
    if (input.length === 0) return [];
    const response = await this.post<EmbeddingResponse>(
      `${this.options.embeddingBaseUrl}/embeddings`,
      { model: this.options.embeddingModel, input },
    );
    const ordered = [...(response.data ?? [])].sort(
      (left, right) => (left.index ?? 0) - (right.index ?? 0),
    );
    if (ordered.length !== input.length) {
      throw new ExternalServiceError(
        `Embedding service returned ${ordered.length} vectors for ${input.length} inputs`,
      );
    }
    return ordered.map((item, index) => {
      const vector = item.embedding;
      if (!vector || vector.length !== this.options.embeddingDimensions) {
        throw new ExternalServiceError(
          `Embedding ${index} has ${vector?.length ?? 0} dimensions; expected ${this.options.embeddingDimensions}`,
        );
      }
      return vector;
    });
  }

  async complete(systemPrompt: string, userPrompt: string): Promise<string> {
    const messages = this.options.model.systemRole
      ? [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ]
      : [
          {
            role: "user",
            content: `Instructions:\n${systemPrompt}\n\nRequest:\n${userPrompt}`,
          },
        ];
    const response = await this.post<ChatResponse>(
      `${this.options.chatBaseUrl}/chat/completions`,
      {
        model: this.options.model.key,
        messages,
        temperature: 0.1,
        max_tokens: 700,
        response_format: { type: "json_object" },
        chat_template_kwargs: this.options.model.chatTemplateKwargs,
      },
    );
    const content = response.choices?.[0]?.message?.content;
    if (!content) {
      throw new ExternalServiceError("Chat service returned no message content");
    }
    return content;
  }

  async ready(): Promise<void> {
    await this.get(`${this.options.chatBaseUrl}/models`);
    await this.get(`${this.options.embeddingBaseUrl}/models`);
  }

  private async get(url: string): Promise<void> {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!response.ok) {
      throw new ExternalServiceError(
        `${url} returned HTTP ${response.status}`,
        response.status,
      );
    }
  }

  private async post<T>(url: string, body: unknown): Promise<T> {
    const response = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!response.ok) {
      const detail = (await response.text()).slice(0, 1000);
      throw new ExternalServiceError(
        `${url} returned HTTP ${response.status}: ${detail}`,
        response.status,
      );
    }
    return (await response.json()) as T;
  }
}
