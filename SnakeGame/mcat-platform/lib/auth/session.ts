import "server-only";

import { createHash } from "crypto";
import { SignJWT, jwtVerify } from "jose";
import { cookies } from "next/headers";
import { env } from "@/lib/config/env";
import { prisma } from "@/lib/db/prisma";

const COOKIE_NAME = "mcat_session";
const key = new TextEncoder().encode(env.JWT_SECRET);

type SessionPayload = { sub: string; email: string; fullName: string; sid: string };

function hashSessionId(sid: string) {
  return createHash("sha256").update(sid).digest("hex");
}

export async function createSession(payload: Omit<SessionPayload, "sid">) {
  const sid = crypto.randomUUID();
  const expiresAt = new Date(Date.now() + 60 * 60 * 24 * 7 * 1000);
  const token = await new SignJWT(payload)
    .setSubject(payload.sub)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("7d")
    .setJti(sid)
    .sign(key);

  await prisma.session.create({
    data: {
      userId: payload.sub,
      tokenHash: hashSessionId(sid),
      expiresAt
    }
  });

  const cookieStore = await cookies();
  cookieStore.set(COOKIE_NAME, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "strict",
    path: "/",
    maxAge: 60 * 60 * 24 * 7
  });
}

export async function clearSession() {
  const cookieStore = await cookies();
  const token = cookieStore.get(COOKIE_NAME)?.value;
  if (token) {
    try {
      const { payload } = await jwtVerify(token, key);
      if (payload.jti && payload.sub) {
        await prisma.session.deleteMany({
          where: {
            userId: String(payload.sub),
            tokenHash: hashSessionId(String(payload.jti))
          }
        });
      }
    } catch {
      // best-effort cleanup
    }
  }
  cookieStore.delete(COOKIE_NAME);
}

export async function getSession() {
  const cookieStore = await cookies();
  const token = cookieStore.get(COOKIE_NAME)?.value;
  if (!token) return null;

  try {
    const { payload } = await jwtVerify(token, key);
    if (!payload.sub || !payload.jti) return null;

    const dbSession = await prisma.session.findFirst({
      where: {
        userId: String(payload.sub),
        tokenHash: hashSessionId(String(payload.jti)),
        expiresAt: { gt: new Date() }
      }
    });

    if (!dbSession) {
      return null;
    }

    return {
      sub: String(payload.sub),
      email: String(payload.email),
      fullName: String(payload.fullName),
      sid: String(payload.jti)
    } as SessionPayload;
  } catch {
    return null;
  }
}
