import { requireSession } from "@/lib/auth/guard";
import { parseJson } from "@/lib/api/parse";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";
import { enforceTrustedMutation } from "@/lib/api/route-guards";
import { practiceSessionService } from "@/lib/services/practice-session-service";
import { z } from "zod";

const startSchema = z.object({
  timed: z.boolean().default(true),
  itemCount: z.number().int().min(8).max(59).default(24)
});

export async function POST(request: Request) {
  try {
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const session = await requireSession();
    const parsed = await parseJson(request, startSchema);
    if (parsed.error) return parsed.error;

    const created = await practiceSessionService.start({
      userId: session.sub,
      kind: "diagnostic",
      timed: parsed.data.timed ?? true,
      itemCount: parsed.data.itemCount ?? 24
    });

    return ok(created, 201);
  } catch (error) {
    return toApiError(error);
  }
}
