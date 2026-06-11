import { isValidCat, isValidName } from "@/lib/paths";
import { readAnnotation, writeAnnotation } from "@/lib/annotation-io";
import type { Annotation, Box } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type RouteParams = { params: Promise<{ cat: string; name: string }> };

export async function GET(_req: Request, { params }: RouteParams): Promise<Response> {
  const { cat, name } = await params;
  if (!isValidCat(cat) || !isValidName(name)) {
    return new Response("bad request", { status: 400 });
  }
  try {
    return Response.json(await readAnnotation(cat, name));
  } catch (e) {
    return new Response((e as Error).message, { status: 500 });
  }
}

export async function PUT(req: Request, { params }: RouteParams): Promise<Response> {
  const { cat, name } = await params;
  if (!isValidCat(cat) || !isValidName(name)) {
    return new Response("bad request", { status: 400 });
  }
  try {
    const body = (await req.json()) as Partial<Annotation>;
    // Defensive coercion — the lib re-cleans on write, so a light check is fine.
    const boxes: Box[] = Array.isArray(body.boxes)
      ? body.boxes.map((b, i) => ({
          id: typeof b?.id === "string" ? b.id : `b-${i}`,
          cx: Number(b?.cx) || 0,
          cy: Number(b?.cy) || 0,
          w: Number(b?.w) || 0,
          h: Number(b?.h) || 0,
          label: typeof b?.label === "string" ? b.label : null,
        }))
      : [];
    const ann: Annotation = { boxed: Boolean(body.boxed), boxes };
    await writeAnnotation(cat, name, ann);
    return Response.json({ ok: true });
  } catch (e) {
    return new Response((e as Error).message, { status: 500 });
  }
}
