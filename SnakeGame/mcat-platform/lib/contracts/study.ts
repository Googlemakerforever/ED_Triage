import { z } from "zod";

export const taskPatchSchema = z.object({
  status: z.enum(["todo", "in_progress", "done"]).optional(),
  dueDate: z.string().date().optional(),
  priority: z.number().int().min(1).max(5).optional()
});

export const studyTaskSchema = z.object({
  id: z.string(),
  title: z.string(),
  section: z.enum(["CP", "CARS", "BB", "PS"]),
  topic: z.string(),
  dueDate: z.string(),
  durationMinutes: z.number(),
  priority: z.number(),
  status: z.enum(["todo", "in_progress", "done"])
});

export type TaskPatchInput = z.infer<typeof taskPatchSchema>;
export type StudyTaskDto = z.infer<typeof studyTaskSchema>;
