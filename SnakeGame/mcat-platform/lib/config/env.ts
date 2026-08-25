import { z } from "zod";

const envSchema = z.object({
  DATABASE_URL: z.string().min(1),
  JWT_SECRET: z.string().min(32),
  APP_ORIGIN: z.string().url(),
  CLAUDE_API_KEY: z.string().optional(),
  CLAUDE_MODEL: z.string().default("claude-3-7-sonnet-latest"),
  GOOGLE_API_KEY: z.string().optional(),
  GOOGLE_MODEL: z.string().default("gemini-1.5-flash")
});

export const env = envSchema.parse({
  DATABASE_URL: process.env.DATABASE_URL,
  JWT_SECRET: process.env.JWT_SECRET,
  APP_ORIGIN: process.env.APP_ORIGIN,
  CLAUDE_API_KEY: process.env.CLAUDE_API_KEY,
  CLAUDE_MODEL: process.env.CLAUDE_MODEL,
  GOOGLE_API_KEY: process.env.GOOGLE_API_KEY,
  GOOGLE_MODEL: process.env.GOOGLE_MODEL
});
