import { prisma } from "@/lib/db/prisma";

export const profileService = {
  getProfile(userId: string) {
    return prisma.userProfile.findUnique({ where: { userId } });
  },

  updateProfile(userId: string, patch: { targetScore?: number; testDate?: string; weeklyStudyHours?: number }) {
    return prisma.userProfile.update({
      where: { userId },
      data: {
        targetScore: patch.targetScore,
        testDate: patch.testDate ? new Date(patch.testDate) : undefined,
        weeklyStudyHours: patch.weeklyStudyHours
      }
    });
  },

  completeOnboarding(userId: string, payload: {
    targetScore: number;
    testDate: string;
    diagnosticScore?: number;
    strongestSection: "CP" | "CARS" | "BB" | "PS";
    weakestSection: "CP" | "CARS" | "BB" | "PS";
    weeklyHours: number;
    studyPhase: "foundation" | "mixed" | "intensive" | "final_review";
  }) {
    return prisma.userProfile.update({
      where: { userId },
      data: {
        targetScore: payload.targetScore,
        testDate: new Date(payload.testDate),
        diagnosticScore: payload.diagnosticScore,
        strongestSection: payload.strongestSection,
        weakestSection: payload.weakestSection,
        weeklyStudyHours: payload.weeklyHours,
        studyPhase: payload.studyPhase
      }
    });
  }
};
