import { requireSession } from "@/lib/auth/guard";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";
import { practiceSessionService } from "@/lib/services/practice-session-service";

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    const session = await requireSession();
    const data = await practiceSessionService.getReview(session.sub, id);
    return ok(data);
  } catch (error) {
    return toApiError(error);
  }
}
