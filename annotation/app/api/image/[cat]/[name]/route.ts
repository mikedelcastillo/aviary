import { readFile } from "node:fs/promises";
import { isValidCat, isValidName, imageFsPath } from "@/lib/paths";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type RouteParams = { params: Promise<{ cat: string; name: string }> };

function contentType(name: string): string {
  return /\.png$/i.test(name) ? "image/png" : "image/jpeg";
}

export async function GET(_req: Request, { params }: RouteParams): Promise<Response> {
  const { cat, name } = await params;
  if (!isValidCat(cat) || !isValidName(name)) {
    return new Response("not found", { status: 404 });
  }
  try {
    const p = imageFsPath(cat, name);
    const buffer = await readFile(p);
    return new Response(buffer, {
      headers: {
        "Content-Type": contentType(name),
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return new Response("not found", { status: 404 });
  }
}
