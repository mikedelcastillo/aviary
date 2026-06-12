import { restoreDeletedImage } from "@/lib/annotation-io";
import { isValidCat, isValidName } from "@/lib/paths";
import { type CatId } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface RestoreBody {
  cat?: string;
  name?: string;
}

/** POST /api/delete/restore { cat, name } -> undo a manual delete (trash -> raw). */
export async function POST(req: Request): Promise<Response> {
  let body: RestoreBody;
  try {
    body = (await req.json()) as RestoreBody;
  } catch {
    return new Response("invalid JSON", { status: 400 });
  }
  const { cat, name } = body;
  if (!cat || !isValidCat(cat) || !name || !isValidName(name)) {
    return new Response("bad request", { status: 400 });
  }
  try {
    const res = await restoreDeletedImage(cat as CatId, name);
    return Response.json({ ok: true, moved: res.moved });
  } catch (e) {
    return new Response(`restore failed: ${(e as Error).message}`, { status: 500 });
  }
}
