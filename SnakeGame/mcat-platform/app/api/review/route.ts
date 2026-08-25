import { requireSession } from "@/lib/auth/guard";
import { prisma } from "@/lib/db/prisma";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";

export async function GET() {
  try {
    const session = await requireSession();
    const [items, recentSessions] = await Promise.all([
      prisma.reviewItem.findMany({ where: { userId: session.sub }, orderBy: { createdAt: "desc" }, take: 30 }),
      prisma.practiceSession.findMany({
        where: { userId: session.sub, status: "submitted" },
        orderBy: { submittedAt: "desc" },
        take: 10,
        select: { id: true, kind: true, scorePercent: true, submittedAt: true }
      })
    ]);
    return ok({ items, recentSessions });
  } catch (error) {
    return toApiError(error);
  }
}
