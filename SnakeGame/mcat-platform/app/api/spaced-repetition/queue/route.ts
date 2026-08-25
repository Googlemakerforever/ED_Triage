import { requireSession } from "@/lib/auth/guard";
import { prisma } from "@/lib/db/prisma";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";

export async function GET() {
  try {
    const session = await requireSession();
    const now = new Date();
    const [due, overdue, mastered] = await Promise.all([
      prisma.spacedRepetitionItem.findMany({ where: { userId: session.sub, dueDate: { lte: now } }, take: 20, orderBy: { dueDate: "asc" } }),
      prisma.spacedRepetitionItem.count({ where: { userId: session.sub, dueDate: { lt: new Date(now.getTime() - 24 * 3600 * 1000) } } }),
      prisma.spacedRepetitionItem.findMany({ where: { userId: session.sub, intervalDays: { gte: 14 } }, take: 6, orderBy: { updatedAt: "desc" } })
    ]);

    return ok({ dueToday: due, overdueCount: overdue, masteredRecently: mastered });
  } catch (error) {
    return toApiError(error);
  }
}
