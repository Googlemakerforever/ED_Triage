import { requireSession } from "@/lib/auth/guard";
import { parseJson } from "@/lib/api/parse";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";
import { enforceRateLimit, enforceTrustedMutation } from "@/lib/api/route-guards";
import { startSessionSchema } from "@/lib/contracts/practice-session";
import { practiceSessionService } from "@/lib/services/practice-session-service";

export async function POST(request: Request) {
  try {
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const rate = enforceRateLimit(request, { scope: "practice-session-create", limit: 60, windowMs: 60_000 });
    if (rate) return rate;
    const session = await requireSession();
    const parsed = await parseJson(request, startSessionSchema);
    if (parsed.error) return parsed.error;

    const created = await practiceSessionService.start({
      userId: session.sub,
      kind: parsed.data.kind,
      section: parsed.data.section,
      timed: parsed.data.timed ?? true,
      itemCount: parsed.data.itemCount ?? 15
    });
    return ok(created, 201);
  } catch (error) {
    return toApiError(error);
  }
}
