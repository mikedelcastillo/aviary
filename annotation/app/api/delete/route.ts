import { deleteImage } from "@/lib/annotation-io";
import { isValidCat, isValidName } from "@/lib/paths";
import { type CatId } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface DeleteBody {
  cat?: string;
  name?: string;
}

/** POST /api/delete { cat, name } -> move one image (+ sidecars) to the trash tree. */
export async function POST(req: Request): Promise<Response> {
  let body: DeleteBody;
  try {
    body = (await req.json()) as DeleteBody;
  } catch {
    return new Response("invalid JSON", { status: 400 });
  }
  const { cat, name } = body;
  if (!cat || !isValidCat(cat) || !name || !isValidName(name)) {
    return new Response("bad request", { status: 400 });
  }
  try {
    const res = await deleteImage(cat as CatId, name);
    return Response.json({ ok: true, moved: res.moved });
  } catch (e) {
    return new Response(`delete failed: ${(e as Error).message}`, { status: 500 });
  }
}
