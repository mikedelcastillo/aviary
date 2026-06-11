import { getManifest } from "@/lib/annotation-io";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  try {
    return Response.json(getManifest());
  } catch (e) {
    return new Response((e as Error).message, { status: 500 });
  }
}
