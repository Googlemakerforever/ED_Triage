import { z } from "zod";
import { requireSession } from "@/lib/auth/guard";
import { prisma } from "@/lib/db/prisma";
import { ok } from "@/lib/api/responses";
import { parseJson } from "@/lib/api/parse";
import { toApiError } from "@/lib/api/errors";
import { enforceTrustedMutation } from "@/lib/api/route-guards";

const createSchema = z.object({
  section: z.enum(["CP", "CARS", "BB", "PS"]),
  topic: z.string().min(2),
  mistakeType: z.enum(["content_gap", "reasoning_error", "timing_issue", "careless_error"]),
  note: z.string().min(2),
  tags: z.array(z.string()).default([])
});

export async function GET() {
  try {
    const session = await requireSession();
    const items = await prisma.errorLogEntry.findMany({ where: { userId: session.sub }, orderBy: { updatedAt: "desc" } });
    return ok(items);
  } catch (error) {
    return toApiError(error);
  }
}

export async function POST(request: Request) {
  try {
    const mutation = enforceTrustedMutation(request);
    if (mutation) return mutation;
    const session = await requireSession();
    const parsed = await parseJson(request, createSchema);
    if (parsed.error) return parsed.error;

    const created = await prisma.errorLogEntry.create({ data: { userId: session.sub, ...parsed.data } });
    return ok(created, 201);
  } catch (error) {
    return toApiError(error);
  }
}
