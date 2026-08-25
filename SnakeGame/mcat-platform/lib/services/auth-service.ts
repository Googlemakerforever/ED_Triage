import { userRepository } from "@/lib/repositories/user-repository";
import { hashPassword, verifyPassword } from "@/lib/security/password";

export const authService = {
  async signup(input: { email: string; password: string; fullName: string }) {
    const existing = await userRepository.findByEmail(input.email);
    if (existing) {
      throw new Error("EMAIL_EXISTS");
    }

    const passwordHash = await hashPassword(input.password);
    return userRepository.create({ email: input.email, fullName: input.fullName, passwordHash });
  },

  async login(input: { email: string; password: string }) {
    const user = await userRepository.findByEmail(input.email);
    if (!user) {
      throw new Error("INVALID_CREDENTIALS");
    }

    const valid = await verifyPassword(input.password, user.passwordHash);
    if (!valid) {
      throw new Error("INVALID_CREDENTIALS");
    }

    return user;
  }
};
