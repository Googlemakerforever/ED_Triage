import { getSession } from "@/lib/auth/session";
import { userRepository } from "@/lib/repositories/user-repository";
import { ok } from "@/lib/api/responses";

export async function GET() {
  const session = await getSession();
  if (!session) return ok(null);

  const user = await userRepository.findById(session.sub);
  if (!user) return ok(null);

  return ok({
    id: user.id,
    email: user.email,
    fullName: user.fullName,
    onboardingComplete: Boolean(user.profile?.targetScore)
  });
}
