import { clearSession } from "@/lib/auth/session";
import { ok } from "@/lib/api/responses";
import { enforceTrustedMutation } from "@/lib/api/route-guards";

export async function POST(request: Request) {
  const mutation = enforceTrustedMutation(request);
  if (mutation) return mutation;
  await clearSession();
  return ok({ success: true });
}
