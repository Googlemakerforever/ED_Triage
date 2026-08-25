import { onboardingSchema } from "@/lib/contracts/onboarding";
import { parseJson } from "@/lib/api/parse";
import { ok } from "@/lib/api/responses";
import { requireSession } from "@/lib/auth/guard";
import { profileService } from "@/lib/services/profile-service";
import { toApiError } from "@/lib/api/errors";
import { enforceTrustedMutation } from "@/lib/api/route-guards";

export async function POST(request: Request) {
  try {
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const session = await requireSession();
    const parsed = await parseJson(request, onboardingSchema);
    if (parsed.error) return parsed.error;

    const profile = await profileService.completeOnboarding(session.sub, parsed.data);
    return ok(profile, 201);
  } catch (error) {
    return toApiError(error);
  }
}
