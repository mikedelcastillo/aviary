import { getDedupeClusters } from "@/lib/dedupe";
import { removeImage } from "@/lib/annotation-io";
import { isValidCat, isValidName } from "@/lib/paths";
import { DEDUPE_DEFAULT_THRESHOLD, parseCats, type CatId } from "@/lib/types";
import { invalidateCat } from "@/lib/hash-indexer";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function clampThreshold(raw: string | null): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return DEDUPE_DEFAULT_THRESHOLD;
  return Math.min(20, Math.max(0, Math.floor(n)));
}

/** GET /api/dedupe?cats=day,ir&threshold=3 -> near-duplicate clusters. */
export async function GET(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const cats = parseCats(url.searchParams.get("cats"));
  const threshold = clampThreshold(url.searchParams.get("threshold"));
  try {
    const clusters = await getDedupeClusters(cats, threshold);
    return Response.json(clusters);
  } catch (e) {
    return new Response(`dedupe failed: ${(e as Error).message}`, { status: 500 });
  }
}

interface CommitBody {
  cat?: string;
  remove?: string[];
}

/** POST /api/dedupe { cat, remove:[names] } -> soft-delete the listed images. */
export async function POST(req: Request): Promise<Response> {
  let body: CommitBody;
  try {
    body = (await req.json()) as CommitBody;
  } catch {
    return new Response("invalid JSON", { status: 400 });
  }
  const { cat, remove } = body;
  if (!cat || !isValidCat(cat) || !Array.isArray(remove)) {
    return new Response("bad request", { status: 400 });
  }
  if (!remove.every((n) => typeof n === "string" && isValidName(n))) {
    return new Response("invalid filename in remove list", { status: 400 });
  }

  const moved: { name: string; files: string[] }[] = [];
  try {
    // Sequential: each removeImage invalidates the manifest after itself.
    for (const name of remove) {
      const res = await removeImage(cat as CatId, name);
      moved.push({ name, files: res.moved });
    }
    invalidateCat(cat as CatId);
  } catch (e) {
    return new Response(`remove failed: ${(e as Error).message}`, { status: 500 });
  }
  return Response.json({ ok: true, moved });
}
