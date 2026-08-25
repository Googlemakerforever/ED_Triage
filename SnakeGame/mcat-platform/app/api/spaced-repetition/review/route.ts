import { z } from "zod";
import { requireSession } from "@/lib/auth/guard";
import { prisma } from "@/lib/db/prisma";
import { parseJson } from "@/lib/api/parse";
import { ok, fail } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";
import { enforceTrustedMutation } from "@/lib/api/route-guards";

const reviewSchema = z.object({
  itemId: z.string(),
  quality: z.enum(["again", "hard", "good", "easy"])
});

const multipliers = { again: 0.4, hard: 0.8, good: 1.0, easy: 1.35 } as const;

export async function POST(request: Request) {
  try {
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const session = await requireSession();
    const parsed = await parseJson(request, reviewSchema);
    if (parsed.error) return parsed.error;

    const current = await prisma.spacedRepetitionItem.findFirst({ where: { id: parsed.data.itemId, userId: session.sub } });
    if (!current) {
      return fail("NOT_FOUND", "Card not found", 404);
    }

    const nextInterval = Math.max(1, Math.round(current.intervalDays * multipliers[parsed.data.quality] * current.ease));
    const ease = parsed.data.quality === "again" ? Math.max(1.3, current.ease - 0.2) : Math.min(3.2, current.ease + 0.05);

    const updated = await prisma.spacedRepetitionItem.update({
      where: { id: current.id },
      data: {
        ease,
        intervalDays: nextInterval,
        reviewCount: current.reviewCount + 1,
        dueDate: new Date(Date.now() + nextInterval * 24 * 3600 * 1000)
      }
    });

    return ok(updated);
  } catch (error) {
    return toApiError(error);
  }
}
