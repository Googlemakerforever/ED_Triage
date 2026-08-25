import { MistakeType, SessionKind } from "@prisma/client";
import { prisma } from "@/lib/db/prisma";

type StartSessionInput = {
  userId: string;
  kind: "diagnostic" | "question" | "passage";
  section?: "CP" | "CARS" | "BB" | "PS";
  timed: boolean;
  itemCount: number;
};

type SubmitAnswer = {
  sessionItemId: string;
  selectedChoiceId: string;
  confidence: number;
  elapsedSeconds: number;
};

function classifyMiss(input: { isCorrect: boolean; confidence: number; elapsedSeconds: number }): {
  missClass: "content_gap" | "reasoning_gap" | "execution_gap" | null;
  mistakeType: MistakeType | null;
} {
  if (input.isCorrect) return { missClass: null, mistakeType: null };

  if (input.elapsedSeconds > 115) {
    return { missClass: "execution_gap", mistakeType: "timing_issue" };
  }

  if (input.confidence >= 4) {
    return { missClass: "reasoning_gap", mistakeType: "reasoning_error" };
  }

  return { missClass: "content_gap", mistakeType: "content_gap" };
}

function sessionKind(kind: StartSessionInput["kind"]): SessionKind {
  if (kind === "diagnostic") return "diagnostic";
  if (kind === "passage") return "passage";
  return "question";
}

export const practiceSessionService = {
  async start(input: StartSessionInput) {
    const questions = await prisma.question.findMany({
      where: {
        topic: {
          section: input.section
        }
      },
      include: { answerChoices: true, topic: true },
      orderBy: { createdAt: "asc" },
      take: input.itemCount
    });

    if (!questions.length) {
      throw new Error("NO_QUESTIONS");
    }

    const session = await prisma.practiceSession.create({
      data: {
        userId: input.userId,
        kind: sessionKind(input.kind),
        section: input.section,
        timed: input.timed,
        status: "in_progress",
        totalItems: questions.length,
        items: {
          create: questions.map((q, idx) => ({
            orderIndex: idx,
            section: q.topic.section,
            topicId: q.topicId,
            questionId: q.id,
            blueprintCategory: q.topic.blueprintCategory ?? q.topic.title,
            reasoningSkill: q.topic.reasoningSkill ?? "Scientific Reasoning"
          }))
        }
      },
      include: {
        items: {
          orderBy: { orderIndex: "asc" },
          include: {
            question: {
              include: {
                answerChoices: {
                  select: { id: true, label: true, text: true }
                },
                topic: true
              }
            }
          }
        }
      }
    });

    return {
      id: session.id,
      kind: session.kind,
      timed: session.timed,
      totalItems: session.totalItems,
      items: session.items.map((item) => ({
        sessionItemId: item.id,
        orderIndex: item.orderIndex,
        section: item.section,
        blueprintCategory: item.blueprintCategory,
        reasoningSkill: item.reasoningSkill,
        question: item.question
          ? {
              id: item.question.id,
              stem: item.question.stem,
              topic: item.question.topic.title,
              difficulty: item.question.difficulty,
              answerChoices: item.question.answerChoices
            }
          : null
      }))
    };
  },

  async submit(userId: string, sessionId: string, answers: SubmitAnswer[]) {
    const session = await prisma.practiceSession.findFirst({
      where: { id: sessionId, userId },
      include: {
        items: {
          include: {
            question: {
              include: {
                answerChoices: true,
                topic: true
              }
            }
          }
        }
      }
    });

    if (!session) throw new Error("NOT_FOUND");
    if (session.status === "submitted") throw new Error("ALREADY_SUBMITTED");

    const itemById = new Map(session.items.map((item) => [item.id, item]));

    const answerWrites = [] as {
      sessionId: string;
      userId: string;
      sessionItemId: string;
      section: "CP" | "CARS" | "BB" | "PS";
      questionId?: string;
      selectedChoiceId: string;
      isCorrect: boolean;
      confidence: number;
      elapsedSeconds: number;
      mistakeType?: MistakeType;
      missClass?: string;
      blueprintCategory?: string;
      reasoningSkill?: string;
    }[];

    let totalCorrect = 0;
    let totalElapsedSeconds = 0;

    for (const answer of answers) {
      const item = itemById.get(answer.sessionItemId);
      if (!item || !item.question) {
        throw new Error("INVALID_SESSION_ITEM");
      }

      const choice = item.question.answerChoices.find((c) => c.id === answer.selectedChoiceId);
      if (!choice) {
        throw new Error("INVALID_CHOICE");
      }

      const isCorrect = choice.isCorrect;
      const classification = classifyMiss({
        isCorrect,
        confidence: answer.confidence,
        elapsedSeconds: answer.elapsedSeconds
      });

      if (isCorrect) totalCorrect += 1;
      totalElapsedSeconds += answer.elapsedSeconds;

      answerWrites.push({
        sessionId: session.id,
        userId,
        sessionItemId: item.id,
        section: item.section,
        questionId: item.questionId ?? undefined,
        selectedChoiceId: answer.selectedChoiceId,
        isCorrect,
        confidence: answer.confidence,
        elapsedSeconds: answer.elapsedSeconds,
        mistakeType: classification.mistakeType ?? undefined,
        missClass: classification.missClass ?? undefined,
        blueprintCategory: item.blueprintCategory ?? undefined,
        reasoningSkill: item.reasoningSkill ?? undefined
      });
    }

    const scorePercent = Math.round((totalCorrect / Math.max(answers.length, 1)) * 100);

    await prisma.$transaction(async (tx) => {
      await tx.practiceAnswer.createMany({ data: answerWrites });

      await tx.practiceSession.update({
        where: { id: session.id },
        data: {
          status: "submitted",
          submittedAt: new Date(),
          totalCorrect,
          totalElapsedSeconds,
          scorePercent
        }
      });

      for (const item of session.items) {
        const answer = answerWrites.find((a) => a.sessionItemId === item.id);
        if (!answer || !item.question) continue;

        await tx.questionAttempt.create({
          data: {
            userId,
            questionId: item.question.id,
            selectedChoiceId: answer.selectedChoiceId,
            isCorrect: answer.isCorrect,
            confidence: answer.confidence,
            elapsedSeconds: answer.elapsedSeconds
          }
        });

        if (!answer.isCorrect) {
          await tx.reviewItem.create({
            data: {
              userId,
              summary: `${item.question.stem.slice(0, 140)}...`,
              keyTakeaway: item.question.mistakeRule
            }
          });

          await tx.errorLogEntry.create({
            data: {
              userId,
              section: item.section,
              topic: item.question.topic.title,
              mistakeType: answer.mistakeType ?? "content_gap",
              note: `Missed in ${session.kind} session. Confidence ${answer.confidence}/5, ${answer.elapsedSeconds}s.`,
              tags: [session.kind, answer.blueprintCategory ?? item.question.topic.title],
              recurrenceCount: 1
            }
          });

          await tx.spacedRepetitionItem.create({
            data: {
              userId,
              prompt: item.question.stem,
              answer: item.question.explanation,
              topic: item.question.topic.title,
              section: item.section,
              dueDate: new Date(Date.now() + 24 * 3600 * 1000)
            }
          });
        }
      }

      const sectionScores = session.items.reduce<Record<string, { total: number; correct: number }>>((acc, item) => {
        const ans = answerWrites.find((a) => a.sessionItemId === item.id);
        if (!ans) return acc;
        const key = item.section;
        acc[key] ??= { total: 0, correct: 0 };
        acc[key].total += 1;
        acc[key].correct += ans.isCorrect ? 1 : 0;
        return acc;
      }, {});

      const cp = sectionScores.CP ? Math.round((sectionScores.CP.correct / sectionScores.CP.total) * 14 + 118) : 124;
      const cars = sectionScores.CARS ? Math.round((sectionScores.CARS.correct / sectionScores.CARS.total) * 14 + 118) : 124;
      const bb = sectionScores.BB ? Math.round((sectionScores.BB.correct / sectionScores.BB.total) * 14 + 118) : 124;
      const ps = sectionScores.PS ? Math.round((sectionScores.PS.correct / sectionScores.PS.total) * 14 + 118) : 124;
      const total = cp + cars + bb + ps;

      await tx.scorePrediction.create({
        data: {
          userId,
          low: Math.max(472, total - 3),
          high: Math.min(528, total + 3),
          confidence: Math.min(95, 60 + Math.round(scorePercent / 2))
        }
      });

      if (session.kind === "diagnostic") {
        const sections: ("CP" | "CARS" | "BB" | "PS")[] = ["CP", "CARS", "BB", "PS"];
        await tx.userProfile.update({
          where: { userId },
          data: {
            diagnosticScore: total,
            weakestSection: [...sections]
              .sort((a, b) => (sectionScores[a]?.correct ?? 0) / Math.max(1, sectionScores[a]?.total ?? 1) - (sectionScores[b]?.correct ?? 0) / Math.max(1, sectionScores[b]?.total ?? 1))[0],
            strongestSection: [...sections]
              .sort((a, b) => (sectionScores[b]?.correct ?? 0) / Math.max(1, sectionScores[b]?.total ?? 1) - (sectionScores[a]?.correct ?? 0) / Math.max(1, sectionScores[a]?.total ?? 1))[0]
          }
        });
      }
    });

    return {
      sessionId: session.id,
      submitted: true,
      scorePercent,
      totalCorrect,
      totalAnswered: answers.length,
      totalElapsedSeconds,
      classifications: {
        contentGaps: answerWrites.filter((a) => a.missClass === "content_gap").length,
        reasoningGaps: answerWrites.filter((a) => a.missClass === "reasoning_gap").length,
        executionGaps: answerWrites.filter((a) => a.missClass === "execution_gap").length
      }
    };
  },

  async getReview(userId: string, sessionId: string) {
    const session = await prisma.practiceSession.findFirst({
      where: { id: sessionId, userId },
      include: {
        answers: {
          include: {
            item: {
              include: {
                question: {
                  include: { answerChoices: true, topic: true }
                }
              }
            }
          },
          orderBy: { createdAt: "asc" }
        }
      }
    });

    if (!session) throw new Error("NOT_FOUND");

    return {
      sessionId: session.id,
      kind: session.kind,
      scorePercent: session.scorePercent,
      items: session.answers.map((ans) => {
        const question = ans.item.question;
        const selected = question?.answerChoices.find((c) => c.id === ans.selectedChoiceId);
        const correct = question?.answerChoices.find((c) => c.isCorrect);
        return {
          sessionItemId: ans.sessionItemId,
          question: question?.stem ?? "Question unavailable",
          section: ans.section,
          topic: question?.topic.title,
          blueprintCategory: ans.blueprintCategory,
          reasoningSkill: ans.reasoningSkill,
          selectedAnswer: selected ? `${selected.label}. ${selected.text}` : "N/A",
          correctAnswer: correct ? `${correct.label}. ${correct.text}` : "N/A",
          explanation: question?.explanation ?? "No explanation available.",
          whyWrong: question?.wrongRationale ?? "No rationale available.",
          mistakeType: ans.mistakeType,
          missClass: ans.missClass,
          confidence: ans.confidence,
          elapsedSeconds: ans.elapsedSeconds,
          nextTimeRule: question?.mistakeRule ?? "Identify the exact evidence line before committing."
        };
      })
    };
  }
};
