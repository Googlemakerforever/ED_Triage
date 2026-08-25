import "server-only";

import Anthropic from "@anthropic-ai/sdk";
import { env } from "@/lib/config/env";
import { AiProvider, AiProviderInput } from "@/lib/ai/provider";

export class ClaudeProvider implements AiProvider {
  private client: Anthropic | null;

  constructor() {
    this.client = env.CLAUDE_API_KEY ? new Anthropic({ apiKey: env.CLAUDE_API_KEY }) : null;
  }

  private async generateWithGoogle(input: AiProviderInput): Promise<string> {
    if (!env.GOOGLE_API_KEY) {
      return "AI is currently in offline mode. Use this as a structured review plan: identify claim, evidence, reasoning, and elimination logic before checking the explanation.";
    }

    const joinedHistory = input.messages.map((m) => `${m.role.toUpperCase()}: ${m.content}`).join("\n\n");
    const prompt = `SYSTEM INSTRUCTIONS:\n${input.system}\n\nCONVERSATION:\n${joinedHistory}`;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20_000);

    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(env.GOOGLE_MODEL)}:generateContent?key=${encodeURIComponent(env.GOOGLE_API_KEY)}`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            generationConfig: {
              temperature: input.temperature ?? 0.3,
              maxOutputTokens: input.maxTokens ?? 600
            },
            contents: [
              {
                role: "user",
                parts: [{ text: prompt }]
              }
            ]
          }),
          signal: controller.signal
        }
      );

      if (!response.ok) {
        return "I hit a temporary AI issue. Continue with this fallback: summarize the core concept, identify your elimination rule, then solve one similar example under 90 seconds.";
      }

      const data = (await response.json()) as {
        candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
      };
      const text = data.candidates?.[0]?.content?.parts?.find((part) => typeof part.text === "string")?.text;
      return text?.slice(0, 6000) ?? "No response generated.";
    } catch {
      return "I hit a temporary AI issue. Continue with this fallback: summarize the core concept, identify your elimination rule, then solve one similar example under 90 seconds.";
    } finally {
      clearTimeout(timeout);
    }
  }

  async generate(input: AiProviderInput) {
    if (!this.client) {
      return this.generateWithGoogle(input);
    }

    try {
      const response = await Promise.race([
        this.client.messages.create({
          model: env.CLAUDE_MODEL,
          max_tokens: input.maxTokens ?? 600,
          temperature: input.temperature ?? 0.3,
          system: input.system,
          messages: input.messages.map((m) => ({ role: m.role, content: m.content }))
        }),
        new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error("AI_TIMEOUT")), 20_000);
        })
      ]);

      const textBlock = response.content.find((b) => b.type === "text");
      return textBlock?.text?.slice(0, 6000) ?? "No response generated.";
    } catch {
      return this.generateWithGoogle(input);
    }
  }
}
