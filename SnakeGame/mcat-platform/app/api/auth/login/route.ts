import { loginSchema } from "@/lib/contracts/auth";
import { parseJson } from "@/lib/api/parse";
import { ok, fail } from "@/lib/api/responses";
import { authService } from "@/lib/services/auth-service";
import { createSession } from "@/lib/auth/session";
import { toApiError } from "@/lib/api/errors";
import { enforceRateLimit, enforceTrustedMutation } from "@/lib/api/route-guards";

export async function POST(request: Request) {
  try {
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const rate = enforceRateLimit(request, { scope: "auth-login", limit: 12, windowMs: 60_000 });
    if (rate) return rate;

    const parsed = await parseJson(request, loginSchema);
    if (parsed.error) return parsed.error;

    const user = await authService.login(parsed.data);
    await createSession({ sub: user.id, email: user.email, fullName: user.fullName });

    return ok({ id: user.id, email: user.email, fullName: user.fullName, onboardingComplete: Boolean(user.profile?.targetScore) });
  } catch (error) {
    if (error instanceof Error && error.message === "INVALID_CREDENTIALS") {
      return fail("INVALID_CREDENTIALS", "Email or password is incorrect.", 401);
    }
    return toApiError(error);
  }
}
