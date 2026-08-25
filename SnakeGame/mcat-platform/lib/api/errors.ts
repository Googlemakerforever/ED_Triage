import { fail } from "@/lib/api/responses";

export function toApiError(error: unknown) {
  if (error instanceof Error && error.message === "UNAUTHORIZED") {
    return fail("UNAUTHORIZED", "Authentication required", 401);
  }
  if (error instanceof Error && error.message === "NOT_FOUND") {
    return fail("NOT_FOUND", "Resource not found", 404);
  }
  if (error instanceof Error && error.message === "CONTEXT_NOT_FOUND") {
    return fail("CONTEXT_NOT_FOUND", "Referenced context was not found or not accessible", 404);
  }
  if (error instanceof Error && error.message === "NO_QUESTIONS") {
    return fail("NO_QUESTIONS", "No questions available for the requested session configuration", 404);
  }
  if (error instanceof Error && error.message === "ALREADY_SUBMITTED") {
    return fail("ALREADY_SUBMITTED", "This session was already submitted", 409);
  }
  if (error instanceof Error && error.message === "INVALID_SESSION_ITEM") {
    return fail("INVALID_SESSION_ITEM", "One or more answers do not belong to this session", 422);
  }
  if (error instanceof Error && error.message === "INVALID_CHOICE") {
    return fail("INVALID_CHOICE", "Answer choice is invalid for the question", 422);
  }
  return fail("INTERNAL_ERROR", "Something went wrong. Please try again.", 500);
}
