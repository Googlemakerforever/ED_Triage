import { aiRepository } from "@/lib/repositories/ai-repository";
import { buildTutorSystemPrompt } from "@/lib/ai/prompt-builder";
import { ClaudeProvider } from "@/lib/ai/claude-provider";
import { prisma } from "@/lib/db/prisma";
import { TutorMode } from "@/types/domain";
import { looksLikePromptInjection, sanitizeAiText } from "@/lib/security/ai-safety";

const provider = new ClaudeProvider();
const modeMap: Record<string, TutorMode> = {
  teach: "teach",
  hint: "hint",
  socratic: "socratic",
  review: "review",
  drill: "drill",
  cars_coach: "cars-coach",
  plan: "plan"
};

async function resolveContextSnippet(input: { userId: string; contextType?: string; contextId?: string }) {
  if (!input.contextType || !input.contextId) return null;

  if (input.contextType === "error_log") {
    const entry = await prisma.errorLogEntry.findFirst({ where: { id: input.contextId, userId: input.userId } });
    if (!entry) throw new Error("CONTEXT_NOT_FOUND");
    return `Error log context: ${entry.topic} (${entry.mistakeType}) - ${entry.note}`;
  }

  if (input.contextType === "study_plan") {
    const plan = await prisma.studyPlan.findFirst({ where: { id: input.contextId, userId: input.userId }, include: { tasks: true } });
    if (!plan) throw new Error("CONTEXT_NOT_FOUND");
    return `Study plan context: ${plan.tasks.slice(0, 5).map((t) => t.title).join("; ")}`;
  }

  if (input.contextType === "topic") {
    const topic = await prisma.topic.findUnique({ where: { id: input.contextId } });
    if (!topic) throw new Error("CONTEXT_NOT_FOUND");
    return `Topic context: ${topic.title} - ${topic.summary}`;
  }

  if (input.contextType === "question") {
    const question = await prisma.question.findUnique({ where: { id: input.contextId } });
    if (!question) throw new Error("CONTEXT_NOT_FOUND");
    return `Question context: ${question.stem}`;
  }

  if (input.contextType === "passage") {
    const passage = await prisma.passage.findUnique({ where: { id: input.contextId } });
    if (!passage) throw new Error("CONTEXT_NOT_FOUND");
    return `Passage context: ${passage.title}`;
  }

  return null;
}

function fallbackTitleFromMessage(message: string) {
  const normalized = message.replace(/\s+/g, " ").trim();
  if (!normalized) return "New conversation";
  const words = normalized.split(" ").slice(0, 7).join(" ");
  return words.length > 72 ? `${words.slice(0, 69)}...` : words;
}

export const aiTutorService = {
  listConversations(userId: string) {
    return aiRepository.listConversations(userId);
  },

  createConversation(userId: string, mode: TutorMode) {
    return aiRepository.createConversation(userId, mode);
  },

  async reply(input: {
    userId: string;
    conversationId: string;
    message: string;
    contextType?: string;
    contextId?: string;
  }) {
    const [conversation, profile, weakTopics, recentAnswers] = await Promise.all([
      aiRepository.getConversation(input.userId, input.conversationId),
      prisma.userProfile.findUnique({ where: { userId: input.userId } }),
      prisma.topicMastery.findMany({ where: { userId: input.userId }, include: { topic: true }, orderBy: { mastery: "asc" }, take: 4 }),
      prisma.practiceAnswer.findMany({
        where: { userId: input.userId },
        orderBy: { createdAt: "desc" },
        take: 20,
        select: { isCorrect: true, confidence: true, elapsedSeconds: true, section: true, missClass: true }
      })
    ]);

    if (!conversation) {
      throw new Error("NOT_FOUND");
    }

    const cleanMessage = sanitizeAiText(input.message);
    const shouldTitleConversation = conversation.title === "New conversation" || !conversation.messages.some((m) => m.role === "user");
    const contextSnippet = await resolveContextSnippet(input);

    const savedUserMessage = await aiRepository.addMessage(conversation.id, "user", cleanMessage, input.contextType, input.contextId);
    let nextConversationTitle = conversation.title;

    if (looksLikePromptInjection(cleanMessage)) {
      const blocked = await aiRepository.addMessage(
        conversation.id,
        "assistant",
        "I can help with MCAT learning, but I cannot follow requests to bypass system safeguards. Ask a content or strategy question.",
        input.contextType,
        input.contextId
      );
      if (shouldTitleConversation) {
        nextConversationTitle = fallbackTitleFromMessage(cleanMessage);
        await aiRepository.updateConversationTitle(conversation.id, nextConversationTitle);
      }
      return {
        userMessage: savedUserMessage,
        message: blocked,
        conversationId: conversation.id,
        conversationTitle: nextConversationTitle
      };
    }

    const system = buildTutorSystemPrompt({
      mode: modeMap[conversation.mode] ?? "teach",
      userContext: {
        targetScore: profile?.targetScore,
        weakSection: profile?.weakestSection,
        weakTopics: weakTopics.map((t) => t.topic.title)
      }
    });
    const recentAccuracy = recentAnswers.length
      ? Math.round((recentAnswers.filter((a) => a.isCorrect).length / recentAnswers.length) * 100)
      : null;
    const recentExecutionGaps = recentAnswers.filter((a) => a.missClass === "execution_gap").length;
    const historyMessages: { role: "user" | "assistant"; content: string }[] = conversation.messages
      .slice(-10)
      .map((m) => ({ role: (m.role === "assistant" ? "assistant" : "user") as "user" | "assistant", content: m.content }));

    const responseText = await provider.generate({
      system: [
        system,
        `Recent accuracy: ${recentAccuracy ?? "unknown"}%`,
        `Recent execution gaps: ${recentExecutionGaps}`,
        "When useful, include a concrete next-day action plan with durations."
      ].join("\\n"),
      maxTokens: 700,
      messages: historyMessages.concat([
        { role: "user", content: contextSnippet ? `${cleanMessage}\n\n${contextSnippet}` : cleanMessage }
      ])
    });

    const saved = await aiRepository.addMessage(conversation.id, "assistant", responseText, input.contextType, input.contextId);
    if (shouldTitleConversation) {
      nextConversationTitle = fallbackTitleFromMessage(cleanMessage);
      await aiRepository.updateConversationTitle(conversation.id, nextConversationTitle);
    }

    return {
      userMessage: savedUserMessage,
      message: saved,
      conversationId: conversation.id,
      conversationTitle: nextConversationTitle
    };
  }
};
