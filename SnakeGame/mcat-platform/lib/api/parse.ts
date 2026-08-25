import { ZodSchema } from "zod";
import { fail } from "@/lib/api/responses";

export async function parseJson<T>(request: Request, schema: ZodSchema<T>) {
  try {
    const body = await request.json();
    const parsed = schema.safeParse(body);
    if (!parsed.success) {
      return { error: fail("VALIDATION_ERROR", parsed.error.issues[0]?.message ?? "Invalid request", 422) };
    }
    return { data: parsed.data };
  } catch {
    return { error: fail("BAD_REQUEST", "Malformed JSON payload", 400) };
  }
}
