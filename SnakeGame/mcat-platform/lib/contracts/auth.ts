import { z } from "zod";

export const signupSchema = z.object({
  email: z.string().trim().toLowerCase().email(),
  password: z.string().min(10).max(128),
  fullName: z.string().trim().min(2).max(120)
});

export const loginSchema = z.object({
  email: z.string().trim().toLowerCase().email(),
  password: z.string().min(8).max(128)
});

export const authUserSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  fullName: z.string(),
  onboardingComplete: z.boolean()
});

export type SignupInput = z.infer<typeof signupSchema>;
export type LoginInput = z.infer<typeof loginSchema>;
export type AuthUser = z.infer<typeof authUserSchema>;
