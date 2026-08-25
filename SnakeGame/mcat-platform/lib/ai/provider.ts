import "server-only";

export type AiProviderInput = {
  system: string;
  messages: { role: "user" | "assistant"; content: string }[];
  maxTokens?: number;
  temperature?: number;
};

export interface AiProvider {
  generate(input: AiProviderInput): Promise<string>;
}
