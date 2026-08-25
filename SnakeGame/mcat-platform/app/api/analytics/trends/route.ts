import { requireSession } from "@/lib/auth/guard";
import { analyticsRepository } from "@/lib/repositories/analytics-repository";
import { fail, ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";
import { z } from "zod";

const querySchema = z.object({
  days: z.coerce.number().int().min(7).max(90).default(30)
});

export async function GET(request: Request) {
  try {
    const session = await requireSession();
    const { searchParams } = new URL(request.url);
    const parsed = querySchema.safeParse({ days: searchParams.get("days") ?? undefined });
    if (!parsed.success) {
      return fail("VALIDATION_ERROR", parsed.error.issues[0]?.message ?? "Invalid query", 422);
    }
    const data = await analyticsRepository.getTrends(session.sub, parsed.data.days);
    return ok(data);
  } catch (error) {
    return toApiError(error);
  }
}
