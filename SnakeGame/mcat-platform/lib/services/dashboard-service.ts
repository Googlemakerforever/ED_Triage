import { prisma } from "@/lib/db/prisma";
import { studyRepository } from "@/lib/repositories/study-repository";
import { analyticsRepository } from "@/lib/repositories/analytics-repository";

export const dashboardService = {
  async getDashboard(userId: string) {
    const [overview, latestPlan, dueCards, errorPatterns, milestones] = await Promise.all([
      analyticsRepository.getOverview(userId),
      studyRepository.getLatestPlan(userId),
      prisma.spacedRepetitionItem.count({ where: { userId, dueDate: { lte: new Date() } } }),
      prisma.errorLogEntry.findMany({ where: { userId }, orderBy: { recurrenceCount: "desc" }, take: 5 }),
      prisma.userProfile.findUnique({ where: { userId } })
    ]);

    return {
      predictedScoreRange: overview.prediction ? `${overview.prediction.low}-${overview.prediction.high}` : "Not enough data",
      sectionSnapshot: {
        accuracy: overview.accuracy,
        masteryAvg: overview.masteryAvg,
        avgConfidence: overview.confidenceVsAccuracy.avgConfidence
      },
      todaysPlan: latestPlan?.tasks.slice(0, 4) ?? [],
      weakestTopics: overview.weakest.map((w) => ({ topic: w.topic.title, mastery: w.mastery })),
      dueReviewItems: dueCards,
      recurringPatterns: errorPatterns,
      recommendedAction: "Complete 2 timed CARS passages, then review every wrong answer with elimination notes.",
      upcomingMilestones: milestones?.testDate ? [{ label: "Test date", date: milestones.testDate }] : [],
      diagnostic: overview.latestDiagnostic
        ? {
            scorePercent: overview.latestDiagnostic.scorePercent,
            totalItems: overview.latestDiagnostic.totalItems,
            completedAt: overview.latestDiagnostic.submittedAt
          }
        : null,
      classifications: overview.classifications
    };
  }
};
