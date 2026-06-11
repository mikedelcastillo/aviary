import { getReviewQueue } from "@/lib/annotation-io";
import { parseCats } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(req: Request): Promise<Response> {
  try {
    const url = new URL(req.url);
    const label = url.searchParams.get("label") ?? "";
    const cats = parseCats(url.searchParams.get("cats"));
    return Response.json(await getReviewQueue(label, cats));
  } catch (e) {
    return new Response((e as Error).message, { status: 500 });
  }
}
