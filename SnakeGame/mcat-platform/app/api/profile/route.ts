import { z } from "zod";
import { requireSession } from "@/lib/auth/guard";
import { profileService } from "@/lib/services/profile-service";
import { ok } from "@/lib/api/responses";
import { parseJson } from "@/lib/api/parse";
import { toApiError } from "@/lib/api/errors";
import { enforceTrustedMutation } from "@/lib/api/route-guards";

const profilePatchSchema = z.object({
  targetScore: z.number().int().min(472).max(528).optional(),
  testDate: z.string().date().optional(),
  weeklyStudyHours: z.number().int().min(1).max(80).optional()
});

export async function GET() {
  try {
    const session = await requireSession();
    const profile = await profileService.getProfile(session.sub);
    return ok(profile);
  } catch (error) {
    return toApiError(error);
  }
}

export async function PATCH(request: Request) {
  try {
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const session = await requireSession();
    const parsed = await parseJson(request, profilePatchSchema);
    if (parsed.error) return parsed.error;

    const profile = await profileService.updateProfile(session.sub, parsed.data);
    return ok(profile);
  } catch (error) {
    return toApiError(error);
  }
}
