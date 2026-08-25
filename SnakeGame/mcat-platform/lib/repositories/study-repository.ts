import { prisma } from "@/lib/db/prisma";

export const studyRepository = {
  getLatestPlan(userId: string) {
    return prisma.studyPlan.findFirst({
      where: { userId },
      include: { tasks: { orderBy: [{ dueDate: "asc" }, { priority: "desc" }] } },
      orderBy: { generatedAt: "desc" }
    });
  },

  async generatePlan(userId: string) {
    const now = new Date();
    const monday = new Date(now);
    monday.setDate(now.getDate() - ((now.getDay() + 6) % 7));
    const [profile, weakPatterns] = await Promise.all([
      prisma.userProfile.findUnique({ where: { userId } }),
      prisma.errorLogEntry.findMany({ where: { userId }, orderBy: { recurrenceCount: "desc" }, take: 4 })
    ]);

    const weakSection = profile?.weakestSection ?? "CARS";
    const weakTopic = weakPatterns[0]?.topic ?? "Argument structure";
    const secondaryTopic = weakPatterns[1]?.topic ?? "Experimental analysis";

    return prisma.studyPlan.create({
      data: {
        userId,
        weekStartDate: monday,
        tasks: {
          create: [
            {
              title: `${weakSection} precision block: high-yield weak skill`,
              section: weakSection,
              topic: weakTopic,
              dueDate: new Date(now.getTime() + 3600 * 1000 * 24),
              durationMinutes: 80,
              priority: 5
            },
            {
              title: "Targeted review + reinforcement set",
              section: weakSection === "BB" ? "CP" : "BB",
              topic: secondaryTopic,
              dueDate: new Date(now.getTime() + 3600 * 1000 * 48),
              durationMinutes: 60,
              priority: 4
            },
            {
              title: "Timed mixed passage block",
              section: "CARS",
              topic: "Inference and elimination",
              dueDate: new Date(now.getTime() + 3600 * 1000 * 72),
              durationMinutes: 75,
              priority: 4
            },
            {
              title: "Spaced repetition + error-log closure",
              section: "PS",
              topic: "Retention and recall",
              dueDate: new Date(now.getTime() + 3600 * 1000 * 96),
              durationMinutes: 45,
              priority: 3
            }
          ]
        }
      },
      include: { tasks: true }
    });
  },

  patchTask(userId: string, taskId: string, patch: { status?: "todo" | "in_progress" | "done"; dueDate?: string; priority?: number }) {
    return prisma.studyTask.updateMany({
      where: { id: taskId, studyPlan: { userId } },
      data: {
        status: patch.status,
        dueDate: patch.dueDate ? new Date(patch.dueDate) : undefined,
        priority: patch.priority
      }
    });
  }
};
