import { isValidCat, isValidName } from "@/lib/paths";
import { readSuggestions, removeLabelSuggestion, removeSuggestedBox } from "@/lib/suggestions-io";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type RouteParams = { params: Promise<{ cat: string; name: string }> };

/** GET: the image's model suggestions (boxes + per-box label suggestions). */
export async function GET(_req: Request, { params }: RouteParams): Promise<Response> {
  const { cat, name } = await params;
  if (!isValidCat(cat) || !isValidName(name)) {
    return new Response("bad request", { status: 400 });
  }
  try {
    return Response.json(await readSuggestions(cat, name));
  } catch (e) {
    return new Response((e as Error).message, { status: 500 });
  }
}

/**
 * DELETE: drop a single suggestion once the human has acted on it.
 *   ?box=<id>      — a box proposal was accepted (added to the .json) or rejected
 *   ?label=<boxId> — a label suggestion was accepted or overridden
 * Returns the shrunk suggestions so the client can resync.
 */
export async function DELETE(req: Request, { params }: RouteParams): Promise<Response> {
  const { cat, name } = await params;
  if (!isValidCat(cat) || !isValidName(name)) {
    return new Response("bad request", { status: 400 });
  }
  const url = new URL(req.url);
  const boxId = url.searchParams.get("box");
  const labelBoxId = url.searchParams.get("label");
  try {
    if (boxId != null) return Response.json(await removeSuggestedBox(cat, name, boxId));
    if (labelBoxId != null) return Response.json(await removeLabelSuggestion(cat, name, labelBoxId));
    return new Response("missing box or label query param", { status: 400 });
  } catch (e) {
    return new Response((e as Error).message, { status: 500 });
  }
}
