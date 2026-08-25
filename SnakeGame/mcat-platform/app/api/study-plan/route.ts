import { requireSession } from "@/lib/auth/guard";
import { studyRepository } from "@/lib/repositories/study-repository";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";

export async function GET() {
  try {
    const session = await requireSession();
    const existing = await studyRepository.getLatestPlan(session.sub);
    const plan = existing ?? (await studyRepository.generatePlan(session.sub));
    return ok(plan);
  } catch (error) {
    return toApiError(error);
  }
}
