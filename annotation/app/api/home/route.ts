import { getHomeData } from "@/lib/annotation-io";
import { parseCats } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * GET /api/home?cats=day,ir -> { progress, queue, boxQueue, entry, labelStats }.
 * One server-side pass over the manifest replaces the five separate homepage
 * scans: one round-trip, each sidecar read once. `progress` is global; the rest
 * are scoped to `cats`.
 */
export async function GET(req: Request): Promise<Response> {
  const cats = parseCats(new URL(req.url).searchParams.get("cats"));
  return Response.json(await getHomeData(cats));
}
