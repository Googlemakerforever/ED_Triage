type Bucket = { count: number; resetAt: number };

const buckets = new Map<string, Bucket>();

function nowMs() {
  return Date.now();
}

export function getClientIp(request: Request) {
  const fwd = request.headers.get("x-forwarded-for");
  if (fwd) return fwd.split(",")[0]?.trim() ?? "unknown";
  return request.headers.get("x-real-ip") ?? "unknown";
}

export function isRateLimited(key: string, limit: number, windowMs: number) {
  const now = nowMs();
  const existing = buckets.get(key);

  if (!existing || now >= existing.resetAt) {
    buckets.set(key, { count: 1, resetAt: now + windowMs });
    return false;
  }

  if (existing.count >= limit) {
    return true;
  }

  existing.count += 1;
  buckets.set(key, existing);
  return false;
}
