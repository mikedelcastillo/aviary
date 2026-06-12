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
 * Square thumbnail of one labeled box. The box is shown whole — `fit: "contain"`
 * letterboxes a tall/wide box inside the square instead of cropping its ends the
 * way `cover` did — with a thin context margin around it and the green box drawn
 * on top so the preview reads like a miniature Focus view.
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
    const bx = Math.round(clamp((box.cx - box.w / 2) * W, 0, W - 1));
    const by = Math.round(clamp((box.cy - box.h / 2) * H, 0, H - 1));
    const bw = Math.round(clamp(box.w * W, 1, W - bx));
    const bh = Math.round(clamp(box.h * H, 1, H - by));

    // Extract region = box plus a thin context margin so the drawn box doesn't
    // sit flush against the frame edge. Padded by a fraction of the box's longer
    // side, then clamped back into the image (an off-image box just gets less
    // margin on the clamped side).
    const pad = Math.round(0.12 * Math.max(bw, bh));
    const left = clamp(bx - pad, 0, W - 1);
    const top = clamp(by - pad, 0, H - 1);
    const width = clamp(bw + 2 * pad, 1, W - left);
    const height = clamp(bh + 2 * pad, 1, H - top);

    // `contain` scale + centering offsets, mirrored here so the green box lands
    // exactly where the box renders after letterboxing.
    const s = Math.min(size / width, size / height);
    const offX = (size - width * s) / 2;
    const offY = (size - height * s) / 2;
    const gx = offX + (bx - left) * s;
    const gy = offY + (by - top) * s;
    const gw = bw * s;
    const gh = bh * s;

    const overlay = Buffer.from(
      `<svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">` +
        `<rect x="${gx.toFixed(1)}" y="${gy.toFixed(1)}" ` +
        `width="${gw.toFixed(1)}" height="${gh.toFixed(1)}" ` +
        `fill="none" stroke="#22c55e" stroke-width="2" />` +
        `</svg>`,
    );

    const out = await sharp(fsPath)
      .extract({ left, top, width, height })
      .resize(size, size, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .composite([{ input: overlay }])
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
