import { getQueue } from "@/lib/annotation-io";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  try {
    return Response.json(await getQueue());
  } catch (e) {
    return new Response((e as Error).message, { status: 500 });
  }
}
