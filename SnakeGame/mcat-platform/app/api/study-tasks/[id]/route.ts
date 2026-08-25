import { taskPatchSchema } from "@/lib/contracts/study";
import { requireSession } from "@/lib/auth/guard";
import { parseJson } from "@/lib/api/parse";
import { ok, fail } from "@/lib/api/responses";
import { studyRepository } from "@/lib/repositories/study-repository";
import { toApiError } from "@/lib/api/errors";
import { enforceTrustedMutation } from "@/lib/api/route-guards";

export async function PATCH(request: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const session = await requireSession();
    const parsed = await parseJson(request, taskPatchSchema);
    if (parsed.error) return parsed.error;

    const result = await studyRepository.patchTask(session.sub, id, parsed.data);
    if (result.count === 0) {
      return fail("NOT_FOUND", "Study task not found", 404);
    }

    return ok({ updated: true });
  } catch (error) {
    return toApiError(error);
  }
}
