import { getGeneration, getManifest } from "@/lib/annotation-io";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * GET /api/generation -> { generation, total }. Cheap change token: `generation`
 * bumps whenever the manifest rebuilds (including external imports/deletes
 * detected via directory mtime), so a client can poll this and refetch only when
 * it changes instead of re-pulling the full manifest on a timer.
 */
export async function GET(): Promise<Response> {
  const generation = getGeneration();
  return Response.json({ generation, total: getManifest().length });
}
