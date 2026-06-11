import { restoreImage } from "@/lib/annotation-io";
import { isValidCat, isValidName } from "@/lib/paths";
import { type CatId } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface RestoreBody {
  cat?: string;
  restore?: string[];
}

/** POST /api/dedupe/restore { cat, restore:[names] } -> undo a soft-delete. */
export async function POST(req: Request): Promise<Response> {
  let body: RestoreBody;
  try {
    body = (await req.json()) as RestoreBody;
  } catch {
    return new Response("invalid JSON", { status: 400 });
  }
  const { cat, restore } = body;
  if (!cat || !isValidCat(cat) || !Array.isArray(restore)) {
    return new Response("bad request", { status: 400 });
  }
  if (!restore.every((n) => typeof n === "string" && isValidName(n))) {
    return new Response("invalid filename in restore list", { status: 400 });
  }

  const moved: { name: string; files: string[] }[] = [];
  try {
    for (const name of restore) {
      const res = await restoreImage(cat as CatId, name);
      moved.push({ name, files: res.moved });
    }
  } catch (e) {
    return new Response(`restore failed: ${(e as Error).message}`, { status: 500 });
  }
  return Response.json({ ok: true, moved });
}
