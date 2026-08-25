import { requireSession } from "@/lib/auth/guard";
import { aiTutorService } from "@/lib/services/ai-tutor-service";
import { parseJson } from "@/lib/api/parse";
import { createMessageSchema } from "@/lib/contracts/ai";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";
import { enforceRateLimit, enforceTrustedMutation } from "@/lib/api/route-guards";

export async function POST(request: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const session = await requireSession();
    const rate = enforceRateLimit(request, { scope: "ai-message", userKey: session.sub, limit: 40, windowMs: 60_000 });
    if (rate) return rate;
    const parsed = await parseJson(request, createMessageSchema);
    if (parsed.error) return parsed.error;

    const data = await aiTutorService.reply({
      userId: session.sub,
      conversationId: id,
      message: parsed.data.message,
      contextType: parsed.data.contextType,
      contextId: parsed.data.contextId
    });

    return ok(data, 201);
  } catch (error) {
    return toApiError(error);
  }
}
