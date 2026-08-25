import { requireSession } from "@/lib/auth/guard";
import { parseJson } from "@/lib/api/parse";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";
import { enforceTrustedMutation } from "@/lib/api/route-guards";
import { submitSessionSchema } from "@/lib/contracts/practice-session";
import { practiceSessionService } from "@/lib/services/practice-session-service";

export async function POST(request: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const session = await requireSession();
    const parsed = await parseJson(request, submitSessionSchema);
    if (parsed.error) return parsed.error;

    const result = await practiceSessionService.submit(session.sub, id, parsed.data.answers);
    return ok(result);
  } catch (error) {
    return toApiError(error);
  }
}
