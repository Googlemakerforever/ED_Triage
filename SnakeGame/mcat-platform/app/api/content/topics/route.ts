import { requireSession } from "@/lib/auth/guard";
import { contentRepository } from "@/lib/repositories/content-repository";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";

export async function GET() {
  try {
    await requireSession();
    const topics = await contentRepository.getTopics();
    return ok(topics);
  } catch (error) {
    return toApiError(error);
  }
}
