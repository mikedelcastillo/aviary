import { pillsForCategory } from "@/lib/roster";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  try {
    return Response.json({
      day: pillsForCategory("day"),
      ir: pillsForCategory("ir"),
      phone: pillsForCategory("phone"),
    });
  } catch (e) {
    return new Response((e as Error).message, { status: 500 });
  }
}
