import { z } from "zod";

export const createConversationSchema = z.object({
  mode: z.enum(["teach", "hint", "socratic", "review", "drill", "cars-coach", "plan"]).default("teach")
});

export const createMessageSchema = z.object({
  message: z.string().min(1).max(1200),
  contextType: z.enum(["question", "passage", "error_log", "topic", "study_plan"]).optional(),
  contextId: z.string().optional()
}).superRefine((value, ctx) => {
  if (value.contextType && !value.contextId) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "contextId is required when contextType is provided"
    });
  }
});

export type CreateConversationInput = z.infer<typeof createConversationSchema>;
export type CreateMessageInput = z.infer<typeof createMessageSchema>;
