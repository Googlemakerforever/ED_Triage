import { z } from "zod";

export const startSessionSchema = z.object({
  kind: z.enum(["diagnostic", "question", "passage"]),
  section: z.enum(["CP", "CARS", "BB", "PS"]).optional(),
  timed: z.boolean().default(true),
  itemCount: z.number().int().min(5).max(59).default(15)
});

export const submitSessionSchema = z.object({
  answers: z.array(
    z.object({
      sessionItemId: z.string().min(10),
      selectedChoiceId: z.string().min(10),
      confidence: z.number().int().min(1).max(5),
      elapsedSeconds: z.number().int().min(1).max(600)
    })
  ).min(1)
});

export type StartSessionInput = z.infer<typeof startSessionSchema>;
export type SubmitSessionInput = z.infer<typeof submitSessionSchema>;
