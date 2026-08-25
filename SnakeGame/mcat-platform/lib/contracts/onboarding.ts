import { z } from "zod";
import { sectionSchema } from "@/lib/validation/shared";

export const onboardingSchema = z.object({
  targetScore: z.number().int().min(472).max(528),
  testDate: z.string().date(),
  diagnosticScore: z.number().int().min(472).max(528).optional(),
  strongestSection: sectionSchema,
  weakestSection: sectionSchema,
  weeklyHours: z.number().int().min(1).max(80),
  studyPhase: z.enum(["foundation", "mixed", "intensive", "final_review"])
}).superRefine((value, ctx) => {
  const testDate = new Date(value.testDate);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  if (testDate < today) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["testDate"],
      message: "Test date must be today or in the future"
    });
  }
});

export type OnboardingInput = z.infer<typeof onboardingSchema>;
