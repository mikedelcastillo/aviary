import { snapshot, warm } from "@/lib/hash-indexer";
import { parseCats } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * GET /api/dedupe/progress?cats=day,ir -> { done, total, running }.
 * Also idempotently warms the index for the requested categories (the side
 * effect that makes the homepage auto-warm). Returns coverage cheaply.
 */
export async function GET(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const cats = parseCats(url.searchParams.get("cats"));
  warm(cats);
  const snap = await snapshot(cats);
  return Response.json(snap);
}
