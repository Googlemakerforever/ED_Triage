import { prisma } from "@/lib/db/prisma";

export const aiRepository = {
  listConversations(userId: string) {
    return prisma.aiConversation.findMany({
      where: { userId },
      orderBy: { updatedAt: "desc" },
      take: 20,
      include: { messages: { orderBy: { createdAt: "asc" } } }
    });
  },

  createConversation(userId: string, mode: "teach" | "hint" | "socratic" | "review" | "drill" | "cars-coach" | "plan") {
    return prisma.aiConversation.create({
      data: {
        userId,
        title: "New conversation",
        mode: mode === "cars-coach" ? "cars_coach" : mode
      },
      include: { messages: { orderBy: { createdAt: "asc" } } }
    });
  },

  getConversation(userId: string, id: string) {
    return prisma.aiConversation.findFirst({ where: { id, userId }, include: { messages: { orderBy: { createdAt: "asc" } } } });
  },

  async addMessage(conversationId: string, role: "user" | "assistant", content: string, contextType?: string, contextId?: string) {
    const [message] = await prisma.$transaction([
      prisma.aiMessage.create({
        data: { conversationId, role, content, contextType, contextId }
      }),
      prisma.aiConversation.update({
        where: { id: conversationId },
        data: { updatedAt: new Date() }
      })
    ]);
    return message;
  },

  updateConversationTitle(conversationId: string, title: string) {
    return prisma.aiConversation.update({
      where: { id: conversationId },
      data: { title }
    });
  }
};
