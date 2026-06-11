import { getBoxQueue } from "@/lib/annotation-io";
import { parseCats } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(req: Request): Promise<Response> {
  try {
    const cats = parseCats(new URL(req.url).searchParams.get("cats"));
    return Response.json(await getBoxQueue(cats));
  } catch (e) {
    return new Response((e as Error).message, { status: 500 });
  }
}
