import { prisma } from "@/lib/db/prisma";

export const practiceRepository = {
  getQuestions(params: { section?: string; topic?: string; difficulty?: number }) {
    return prisma.question.findMany({
      where: {
        topic: {
          section: params.section as never | undefined,
          title: params.topic ? { contains: params.topic, mode: "insensitive" } : undefined
        },
        difficulty: params.difficulty
      },
      include: {
        answerChoices: { select: { id: true, label: true, text: true } },
        topic: true
      },
      take: 30
    });
  },

  getPassages(params: { section?: string; topic?: string }) {
    return prisma.passage.findMany({
      where: {
        section: params.section as never | undefined,
        topic: params.topic ? { title: { contains: params.topic, mode: "insensitive" } } : undefined
      },
      include: {
        questions: {
          include: {
            answerChoices: { select: { id: true, label: true, text: true } }
          }
        },
        topic: true
      },
      take: 20
    });
  },

  createQuestionAttempt(input: {
    userId: string;
    questionId: string;
    selectedChoiceId: string;
    confidence: number;
    elapsedSeconds: number;
  }) {
    return prisma.questionAttempt.create({
      data: {
        userId: input.userId,
        questionId: input.questionId,
        selectedChoiceId: input.selectedChoiceId,
        confidence: input.confidence,
        elapsedSeconds: input.elapsedSeconds,
        isCorrect: false
      }
    });
  }
};
