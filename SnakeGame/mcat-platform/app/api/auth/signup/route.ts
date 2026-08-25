import { signupSchema } from "@/lib/contracts/auth";
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
    const rate = enforceRateLimit(request, { scope: "auth-signup", limit: 10, windowMs: 60_000 });
    if (rate) return rate;

    const parsed = await parseJson(request, signupSchema);
    if (parsed.error) return parsed.error;

    const user = await authService.signup(parsed.data);
    await createSession({ sub: user.id, email: user.email, fullName: user.fullName });

    return ok({ id: user.id, email: user.email, fullName: user.fullName, onboardingComplete: Boolean(user.profile?.targetScore) }, 201);
  } catch (error) {
    if (error instanceof Error && error.message === "EMAIL_EXISTS") {
      return fail("EMAIL_EXISTS", "An account with that email already exists.", 409);
    }
    return toApiError(error);
  }
}
