import { requireSession } from "@/lib/auth/guard";
import { studyRepository } from "@/lib/repositories/study-repository";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";
import { enforceTrustedMutation } from "@/lib/api/route-guards";

export async function POST(request: Request) {
  try {
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const session = await requireSession();
    const plan = await studyRepository.generatePlan(session.sub);
    return ok(plan, 201);
  } catch (error) {
    return toApiError(error);
  }
}
