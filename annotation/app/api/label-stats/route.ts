import { getLabelStats } from "@/lib/annotation-io";
import { parseCats } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(req: Request) {
  const cats = parseCats(new URL(req.url).searchParams.get("cats"));
  return Response.json(await getLabelStats(cats));
}
