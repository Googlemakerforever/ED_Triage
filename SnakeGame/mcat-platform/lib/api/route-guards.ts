import { fail } from "@/lib/api/responses";
import { getClientIp, isRateLimited } from "@/lib/security/rate-limit";
import { isSafeMethod, isTrustedOrigin } from "@/lib/security/csrf";

export function enforceTrustedMutation(request: Request) {
  if (isSafeMethod(request.method)) return null;
  if (!isTrustedOrigin(request)) {
    return fail("UNTRUSTED_ORIGIN", "Request origin is not allowed", 403);
  }
  return null;
}

export function enforceRateLimit(request: Request, options: { scope: string; limit: number; windowMs: number; userKey?: string }) {
  const ip = getClientIp(request);
  const key = `${options.scope}:${options.userKey ?? "anon"}:${ip}`;
  if (isRateLimited(key, options.limit, options.windowMs)) {
    return fail("RATE_LIMITED", "Too many requests. Please retry shortly.", 429);
  }
  return null;
}
