import { subDays } from "date-fns";
import { prisma } from "@/lib/db/prisma";

export const analyticsRepository = {
  async getOverview(userId: string) {
    const [masteries, attempts, predictions, practiceAnswers, recentDiagnostic] = await Promise.all([
      prisma.topicMastery.findMany({ where: { userId }, include: { topic: true } }),
      prisma.questionAttempt.findMany({ where: { userId } }),
      prisma.scorePrediction.findMany({ where: { userId }, orderBy: { generatedAt: "desc" }, take: 1 }),
      prisma.practiceAnswer.findMany({ where: { userId } }),
      prisma.practiceSession.findFirst({ where: { userId, kind: "diagnostic", status: "submitted" }, orderBy: { submittedAt: "desc" } })
    ]);

    const accuracy = attempts.length
      ? Math.round((attempts.filter((a) => a.isCorrect).length / attempts.length) * 100)
      : 0;

    return {
      prediction: predictions[0] ?? null,
      accuracy,
      masteryAvg: masteries.length ? Math.round(masteries.reduce((sum, m) => sum + m.mastery, 0) / masteries.length) : 0,
      weakest: masteries.sort((a, b) => a.mastery - b.mastery).slice(0, 4),
      confidenceVsAccuracy: {
        avgConfidence: practiceAnswers.length ? Math.round(practiceAnswers.reduce((sum, p) => sum + p.confidence, 0) / practiceAnswers.length) : 0,
        accuracy: practiceAnswers.length ? Math.round((practiceAnswers.filter((p) => p.isCorrect).length / practiceAnswers.length) * 100) : 0
      },
      timingVsAccuracy: {
        avgTimeSeconds: practiceAnswers.length ? Math.round(practiceAnswers.reduce((sum, p) => sum + p.elapsedSeconds, 0) / practiceAnswers.length) : 0,
        accuracy: practiceAnswers.length ? Math.round((practiceAnswers.filter((p) => p.isCorrect).length / practiceAnswers.length) * 100) : 0
      },
      classifications: {
        contentGaps: practiceAnswers.filter((p) => p.missClass === "content_gap").length,
        reasoningGaps: practiceAnswers.filter((p) => p.missClass === "reasoning_gap").length,
        executionGaps: practiceAnswers.filter((p) => p.missClass === "execution_gap").length
      },
      latestDiagnostic: recentDiagnostic
    };
  },

  async getTrends(userId: string, days = 30) {
    const since = subDays(new Date(), days);
    const attempts = await prisma.practiceAnswer.findMany({ where: { userId, createdAt: { gte: since } } });

    const byDay = new Map<string, { total: number; correct: number; avgTime: number }>();
    for (const attempt of attempts) {
      const day = attempt.createdAt.toISOString().slice(0, 10);
      const item = byDay.get(day) ?? { total: 0, correct: 0, avgTime: 0 };
      item.total += 1;
      item.correct += attempt.isCorrect ? 1 : 0;
      item.avgTime += attempt.elapsedSeconds;
      byDay.set(day, item);
    }

    return [...byDay.entries()].map(([date, value]) => ({
      date,
      accuracy: Math.round((value.correct / Math.max(value.total, 1)) * 100),
      avgTime: Math.round(value.avgTime / Math.max(value.total, 1))
    }));
  },

  getMastery(userId: string) {
    return prisma.topicMastery.findMany({ where: { userId }, include: { topic: true }, orderBy: { mastery: "asc" } });
  }
};
