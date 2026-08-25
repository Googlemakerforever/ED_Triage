import { prisma } from "@/lib/db/prisma";

export const contentRepository = {
  getTopics() {
    return prisma.topic.findMany({ orderBy: [{ highYield: "desc" }, { title: "asc" }] });
  },

  getTopicBySlug(slug: string) {
    return prisma.topic.findUnique({
      where: { slug },
      include: { questions: { include: { answerChoices: true }, take: 8 }, passages: { take: 4 } }
    });
  }
};
