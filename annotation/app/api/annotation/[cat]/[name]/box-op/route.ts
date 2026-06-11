import { isValidCat, isValidName } from "@/lib/paths";
import { mutateAnnotation } from "@/lib/annotation-io";
import type { Box } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type RouteParams = { params: Promise<{ cat: string; name: string }> };

/**
 * Single-box mutation for the grid review (which doesn't mount the per-image
 * annotation hook). Atomic + serialized per file via mutateAnnotation, which
 * also regenerates the YOLO .txt.
 *
 *   remove   — delete the box (Unbox)
 *   setLabel — set/clear the box's label (Unlabel sends label:null)
 *   add      — re-insert a full box (Undo of remove): same id, geometry, label
 */
type BoxOp =
  | { op: "remove"; id: string }
  | { op: "setLabel"; id: string; label: string | null }
  | { op: "add"; box: Box };

function coerceBox(b: unknown): Box | null {
  if (!b || typeof b !== "object") return null;
  const r = b as Record<string, unknown>;
  if (typeof r.id !== "string") return null;
  return {
    id: r.id,
    cx: Number(r.cx) || 0,
    cy: Number(r.cy) || 0,
    w: Number(r.w) || 0,
    h: Number(r.h) || 0,
    label: typeof r.label === "string" ? r.label : null,
  };
}

export async function POST(req: Request, { params }: RouteParams): Promise<Response> {
  const { cat, name } = await params;
  if (!isValidCat(cat) || !isValidName(name)) {
    return new Response("bad request", { status: 400 });
  }

  let body: BoxOp;
  try {
    body = (await req.json()) as BoxOp;
  } catch {
    return new Response("bad json", { status: 400 });
  }

  try {
    let annotation;
    switch (body.op) {
      case "remove": {
        if (typeof body.id !== "string") return new Response("missing id", { status: 400 });
        annotation = await mutateAnnotation(cat, name, (boxes) =>
          boxes.filter((b) => b.id !== body.id),
        );
        break;
      }
      case "setLabel": {
        if (typeof body.id !== "string") return new Response("missing id", { status: 400 });
        const label = typeof body.label === "string" ? body.label : null;
        annotation = await mutateAnnotation(cat, name, (boxes) =>
          boxes.map((b) => (b.id === body.id ? { ...b, label } : b)),
        );
        break;
      }
      case "add": {
        const box = coerceBox(body.box);
        if (!box) return new Response("bad box", { status: 400 });
        annotation = await mutateAnnotation(cat, name, (boxes) =>
          boxes.some((b) => b.id === box.id) ? boxes : [...boxes, box],
        );
        break;
      }
      default:
        return new Response("unknown op", { status: 400 });
    }
    return Response.json({ ok: true, annotation });
  } catch (e) {
    return new Response((e as Error).message, { status: 500 });
  }
}
