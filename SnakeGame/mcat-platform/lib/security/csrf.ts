import { env } from "@/lib/config/env";

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

export function isSafeMethod(method: string) {
  return SAFE_METHODS.has(method.toUpperCase());
}

export function isTrustedOrigin(request: Request) {
  const origin = request.headers.get("origin");
  const fetchSite = request.headers.get("sec-fetch-site");

  // Non-browser clients may omit Origin; allow only when sec-fetch-site is also absent.
  if (!origin) {
    return !fetchSite;
  }

  if (fetchSite && fetchSite !== "same-origin" && fetchSite !== "same-site") {
    return false;
  }

  if (origin === env.APP_ORIGIN) return true;
  try {
    const originUrl = new URL(origin);
    const requestUrl = new URL(request.url);
    return originUrl.host === requestUrl.host;
  } catch {
    return false;
  }
}
