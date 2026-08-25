import { requireSession } from "@/lib/auth/guard";
import { aiTutorService } from "@/lib/services/ai-tutor-service";
import { parseJson } from "@/lib/api/parse";
import { createConversationSchema } from "@/lib/contracts/ai";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";
import { enforceRateLimit, enforceTrustedMutation } from "@/lib/api/route-guards";

export async function GET() {
  try {
    const session = await requireSession();
    const conversations = await aiTutorService.listConversations(session.sub);
    return ok(conversations);
  } catch (error) {
    return toApiError(error);
  }
}

export async function POST(request: Request) {
  try {
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const session = await requireSession();
    const rate = enforceRateLimit(request, { scope: "ai-conversation-create", userKey: session.sub, limit: 20, windowMs: 60_000 });
    if (rate) return rate;
    const parsed = await parseJson(request, createConversationSchema);
    if (parsed.error) return parsed.error;

    const conversation = await aiTutorService.createConversation(session.sub, parsed.data.mode ?? "teach");
    return ok(conversation, 201);
  } catch (error) {
    return toApiError(error);
  }
}
