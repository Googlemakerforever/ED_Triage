import { requireSession } from "@/lib/auth/guard";
import { analyticsRepository } from "@/lib/repositories/analytics-repository";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";

export async function GET() {
  try {
    const session = await requireSession();
    const data = await analyticsRepository.getMastery(session.sub);
    return ok(data);
  } catch (error) {
    return toApiError(error);
  }
}
