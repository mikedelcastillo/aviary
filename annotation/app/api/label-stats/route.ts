import { getLabelStats } from "@/lib/annotation-io";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  return Response.json(await getLabelStats());
}
