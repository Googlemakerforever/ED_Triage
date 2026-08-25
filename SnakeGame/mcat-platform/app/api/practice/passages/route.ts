import { requireSession } from "@/lib/auth/guard";
import { practiceRepository } from "@/lib/repositories/practice-repository";
import { fail, ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";
import { z } from "zod";

const querySchema = z.object({
  section: z.enum(["CP", "CARS", "BB", "PS"]).optional(),
  topic: z.string().min(2).max(120).optional()
});

export async function GET(request: Request) {
  try {
    await requireSession();
    const { searchParams } = new URL(request.url);
    const parsed = querySchema.safeParse({
      section: searchParams.get("section") ?? undefined,
      topic: searchParams.get("topic") ?? undefined
    });
    if (!parsed.success) {
      return fail("VALIDATION_ERROR", parsed.error.issues[0]?.message ?? "Invalid query", 422);
    }
    const passages = await practiceRepository.getPassages(parsed.data);
    return ok(passages);
  } catch (error) {
    return toApiError(error);
  }
}
