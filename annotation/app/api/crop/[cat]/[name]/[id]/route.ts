import sharp from "sharp";
import { isValidCat, isValidName, imageFsPath } from "@/lib/paths";
import { readAnnotation } from "@/lib/annotation-io";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type RouteParams = { params: Promise<{ cat: string; name: string; id: string }> };

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(Math.max(n, lo), hi);
}

/**
 * Square thumbnail of one labeled box, cropped tight to the box and scaled to
 * fill the whole preview (no surrounding context). `fit: "cover"` fills the
 * square without distortion, cropping any overflow on the longer axis centrally.
 *
 * The extract region MUST be integer and fully in-bounds or libvips throws
 * `extract_area: bad extract area`, so the box rect is clamped to the image and
 * to a minimum of 1px per side.
 */
export async function GET(_req: Request, { params }: RouteParams): Promise<Response> {
  const { cat, name, id } = await params;
  if (!isValidCat(cat) || !isValidName(name)) {
    return new Response("not found", { status: 404 });
  }

  try {
    const ann = await readAnnotation(cat, name);
    const box = ann.boxes.find((b) => b.id === id);
    if (!box) return new Response("not found", { status: 404 });

    const url = new URL(_req.url);
    const size = Math.round(clamp(Number(url.searchParams.get("size")) || 160, 48, 320));

    const fsPath = imageFsPath(cat, name);
    const meta = await sharp(fsPath).metadata();
    const W = meta.width ?? 0;
    const H = meta.height ?? 0;
    if (!W || !H) return new Response("bad image", { status: 500 });

    // Tight box rectangle in image pixels, clamped to the image and to >=1px.
    const left = Math.round(clamp((box.cx - box.w / 2) * W, 0, W - 1));
    const top = Math.round(clamp((box.cy - box.h / 2) * H, 0, H - 1));
    const width = Math.round(clamp(box.w * W, 1, W - left));
    const height = Math.round(clamp(box.h * H, 1, H - top));

    const out = await sharp(fsPath)
      .extract({ left, top, width, height })
      .resize(size, size, { fit: "cover" })
      .png()
      .toBuffer();

    // Wrap in a fresh Uint8Array so the body is a BodyInit-compatible
    // ArrayBuffer view (sharp's Buffer<ArrayBufferLike> isn't directly assignable).
    return new Response(new Uint8Array(out), {
      headers: {
        "Content-Type": "image/png",
        // Box geometry can change in Focus Review, so don't immutable-cache by
        // id; the grid drops cells on action so stale crops aren't refetched.
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return new Response("not found", { status: 404 });
  }
}
