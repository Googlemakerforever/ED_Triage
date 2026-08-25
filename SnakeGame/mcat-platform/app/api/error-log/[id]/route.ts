import { z } from "zod";
import { requireSession } from "@/lib/auth/guard";
import { prisma } from "@/lib/db/prisma";
import { parseJson } from "@/lib/api/parse";
import { ok, fail } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";
import { enforceTrustedMutation } from "@/lib/api/route-guards";

const patchSchema = z.object({
  status: z.enum(["unresolved", "in_progress", "resolved"]).optional(),
  note: z.string().min(2).optional(),
  tags: z.array(z.string()).optional()
});

export async function PATCH(request: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const session = await requireSession();
    const parsed = await parseJson(request, patchSchema);
    if (parsed.error) return parsed.error;

    const result = await prisma.errorLogEntry.updateMany({ where: { id, userId: session.sub }, data: parsed.data });
    if (!result.count) {
      return fail("NOT_FOUND", "Error log entry not found", 404);
    }
    return ok({ updated: true });
  } catch (error) {
    return toApiError(error);
  }
}
