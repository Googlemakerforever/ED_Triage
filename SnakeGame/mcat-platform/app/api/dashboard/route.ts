import { requireSession } from "@/lib/auth/guard";
import { dashboardService } from "@/lib/services/dashboard-service";
import { ok } from "@/lib/api/responses";
import { toApiError } from "@/lib/api/errors";

export async function GET() {
  try {
    const session = await requireSession();
    const data = await dashboardService.getDashboard(session.sub);
    return ok(data);
  } catch (error) {
    return toApiError(error);
  }
}
