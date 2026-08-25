import { prisma } from "@/lib/db/prisma";

export const userRepository = {
  findByEmail(email: string) {
    return prisma.user.findUnique({ where: { email }, include: { profile: true } });
  },

  findById(id: string) {
    return prisma.user.findUnique({ where: { id }, include: { profile: true } });
  },

  async create(input: { email: string; fullName: string; passwordHash: string }) {
    return prisma.user.create({
      data: {
        email: input.email,
        fullName: input.fullName,
        passwordHash: input.passwordHash,
        profile: { create: {} }
      },
      include: { profile: true }
    });
  }
};
