import { snapshot, warm } from "@/lib/hash-indexer";
import { parseCats } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * GET /api/dedupe/progress?cats=day,ir -> { done, total, running }.
 * Also idempotently warms the index for the requested categories (the side
 * effect that makes the homepage auto-warm). Returns coverage cheaply.
 *
 * Snapshot BEFORE warm: `running` must reflect a pre-existing in-flight job, not
 * the one this request just kicked. Warming first would make every snapshot see a
 * freshly-created job (it can't finish before the snapshot reads it), pinning
 * `running` true forever and preventing the homepage poll from ever stopping.
 */
export async function GET(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const cats = parseCats(url.searchParams.get("cats"));
  const snap = await snapshot(cats);
  warm(cats);
  return Response.json(snap);
}
