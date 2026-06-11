import sharp from "sharp";
import { isValidCat, isValidName, imageFsPath } from "@/lib/paths";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type RouteParams = { params: Promise<{ cat: string; name: string }> };

function clampWidth(raw: string | null): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return 280;
  return Math.min(1024, Math.max(64, Math.floor(n)));
}

/** GET /api/thumb/:cat/:name?w=280 -> downscaled JPEG (dedupe grid thumbnails). */
export async function GET(req: Request, { params }: RouteParams): Promise<Response> {
  const { cat, name } = await params;
  if (!isValidCat(cat) || !isValidName(name)) {
    return new Response("not found", { status: 404 });
  }
  const width = clampWidth(new URL(req.url).searchParams.get("w"));
  try {
    const buffer = await sharp(imageFsPath(cat, name))
      .resize({ width, withoutEnlargement: true })
      .jpeg({ quality: 70 })
      .toBuffer();
    // Copy into a fresh ArrayBuffer-backed view so the type matches BodyInit:
    // sharp's Buffer<ArrayBufferLike> includes SharedArrayBuffer, which the DOM
    // Response type rejects.
    const body = new Uint8Array(new ArrayBuffer(buffer.byteLength));
    body.set(buffer);
    return new Response(body, {
      headers: {
        "Content-Type": "image/jpeg",
        // no-store: content can change under soft-delete/restore; avoid stale thumbs.
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return new Response("not found", { status: 404 });
  }
}
