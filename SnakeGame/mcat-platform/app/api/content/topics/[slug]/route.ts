import { requireSession } from "@/lib/auth/guard";
import { contentRepository } from "@/lib/repositories/content-repository";
import { ok, fail } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";

export async function GET(_request: Request, context: { params: Promise<{ slug: string }> }) {
  try {
    const { slug } = await context.params;
    await requireSession();
    const topic = await contentRepository.getTopicBySlug(slug);
    if (!topic) return fail("NOT_FOUND", "Topic not found", 404);
    return ok(topic);
  } catch (error) {
    return toApiError(error);
  }
}
