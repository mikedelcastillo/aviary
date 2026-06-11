// Server-only filesystem data layer. The filesystem IS the database:
//  - <image>.json  : source-of-truth annotation (boxed flag + boxes w/ labels)
//  - <image>.txt   : YOLO export (labeled boxes only, at global roster indices)
import { promises as fs } from "node:fs";
import { existsSync, readdirSync } from "node:fs";
import path from "node:path";
import {
  ALL_CATS,
  CATEGORIES,
  filterByCats,
  type Annotation,
  type Box,
  type CatId,
  type CategoryProgress,
  type ManifestEntry,
} from "./types";
import { categoryDir, imageFsPath, sidecarFsPath } from "./paths";
import { loadRoster, nameToIndex } from "./roster";

const IMG_RE = /\.(jpe?g|png)$/i;

function listImages(cat: CatId): string[] {
  const dir = categoryDir(cat);
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => IMG_RE.test(f))
    .sort((a, b) => a.localeCompare(b));
}

// Global manifest cached per server run (indices stable for a session).
let manifestCache: ManifestEntry[] | null = null;

export function getManifest(): ManifestEntry[] {
  if (manifestCache) return manifestCache;
  const out: ManifestEntry[] = [];
  let n = 0;
  for (const c of CATEGORIES) {
    for (const name of listImages(c.id)) {
      out.push({ n, cat: c.id, name });
      n++;
    }
  }
  manifestCache = out;
  return out;
}

export function getManifestEntry(n: number): ManifestEntry | undefined {
  return getManifest()[n];
}

export function manifestForCat(cat: CatId): ManifestEntry[] {
  return getManifest().filter((e) => e.cat === cat);
}

// --- YOLO <-> Box helpers ---------------------------------------------------

function parseYolo(txt: string): Box[] {
  const boxes: Box[] = [];
  txt.split("\n").forEach((line, i) => {
    const p = line.trim().split(/\s+/);
    if (p.length < 5) return;
    const [, cx, cy, w, h] = p;
    boxes.push({
      id: `seed-${i}`,
      cx: Number(cx),
      cy: Number(cy),
      w: Number(w),
      h: Number(h),
      label: null, // seed boxes start unlabeled regardless of their placeholder class
    });
  });
  return boxes;
}

function f(n: number): string {
  return Math.min(1, Math.max(0, n)).toFixed(6);
}

function toYolo(boxes: Box[]): string {
  const idx = nameToIndex();
  const lines: string[] = [];
  for (const b of boxes) {
    if (b.label == null) continue; // only labeled boxes are exported
    const cls = idx.get(b.label);
    if (cls == null) continue;
    lines.push(`${cls} ${f(b.cx)} ${f(b.cy)} ${f(b.w)} ${f(b.h)}`);
  }
  return lines.length ? lines.join("\n") + "\n" : "";
}

// --- Read / write -----------------------------------------------------------

/**
 * Read an image's annotation. If no JSON sidecar exists yet, seed boxes from the
 * auto-detection .txt (all unlabeled) with boxed=false. NEVER writes on read.
 */
export async function readAnnotation(cat: CatId, name: string): Promise<Annotation> {
  const jsonPath = sidecarFsPath(cat, name, ".json");
  try {
    const raw = await fs.readFile(jsonPath, "utf8");
    const data = JSON.parse(raw) as Annotation;
    return {
      boxed: Boolean(data.boxed),
      boxes: (data.boxes ?? []).map((b, i) => ({
        id: b.id ?? `b-${i}`,
        cx: b.cx,
        cy: b.cy,
        w: b.w,
        h: b.h,
        label: b.label ?? null,
      })),
    };
  } catch {
    // No JSON yet: seed from the auto-detection .txt if present.
    const txtPath = sidecarFsPath(cat, name, ".txt");
    try {
      const txt = await fs.readFile(txtPath, "utf8");
      return { boxed: false, boxes: parseYolo(txt) };
    } catch {
      return { boxed: false, boxes: [] };
    }
  }
}

/** Persist an annotation: atomic JSON write + rewrite the YOLO .txt export. */
export async function writeAnnotation(cat: CatId, name: string, ann: Annotation): Promise<void> {
  const jsonPath = sidecarFsPath(cat, name, ".json");
  const txtPath = sidecarFsPath(cat, name, ".txt");
  const clean: Annotation = {
    boxed: Boolean(ann.boxed),
    boxes: (ann.boxes ?? []).map((b, i) => ({
      id: b.id ?? `b-${i}`,
      cx: b.cx,
      cy: b.cy,
      w: b.w,
      h: b.h,
      label: b.label ?? null,
    })),
  };

  // Atomic JSON write (temp + rename).
  const tmp = `${jsonPath}.tmp-${process.pid}`;
  await fs.writeFile(tmp, JSON.stringify(clean, null, 2), "utf8");
  await fs.rename(tmp, jsonPath);

  // Mirror to YOLO export (overwrite; empty file == reviewed negative).
  await fs.writeFile(txtPath, toYolo(clean.boxes), "utf8");
}

// --- Progress + queue -------------------------------------------------------

async function quickState(cat: CatId, name: string): Promise<{ boxed: boolean; labeled: boolean; hasUnlabeled: boolean }> {
  const jsonPath = sidecarFsPath(cat, name, ".json");
  try {
    const raw = await fs.readFile(jsonPath, "utf8");
    const data = JSON.parse(raw) as Annotation;
    const boxed = Boolean(data.boxed);
    const boxes = data.boxes ?? [];
    const hasUnlabeled = boxes.some((b) => b.label == null);
    const labeled = boxed && !hasUnlabeled;
    return { boxed, labeled, hasUnlabeled };
  } catch {
    return { boxed: false, labeled: false, hasUnlabeled: false };
  }
}

export async function getProgress(): Promise<CategoryProgress[]> {
  const manifest = getManifest();
  const out: CategoryProgress[] = [];
  for (const c of CATEGORIES) {
    const entries = manifest.filter((e) => e.cat === c.id);
    let boxed = 0;
    let labeled = 0;
    await Promise.all(
      entries.map(async (e) => {
        const s = await quickState(c.id, e.name);
        if (s.boxed) boxed++;
        if (s.labeled) labeled++;
      }),
    );
    out.push({
      id: c.id,
      title: c.title,
      subtitle: c.subtitle,
      total: entries.length,
      boxed,
      labeled,
      startN: entries[0]?.n ?? 0,
    });
  }
  return out;
}

/**
 * Global indices eligible for the label queue: an image must be BOXED
 * (human-vetted) AND still have at least one unlabeled box. You cannot label an
 * image that hasn't actually been boxed — auto-detection seed boxes never
 * qualify on their own.
 */
export async function getQueue(cats: CatId[] = ALL_CATS): Promise<number[]> {
  const manifest = filterByCats(getManifest(), cats);
  const flags = await Promise.all(
    manifest.map(async (e) => {
      const jsonPath = sidecarFsPath(e.cat, e.name, ".json");
      try {
        const raw = await fs.readFile(jsonPath, "utf8");
        const data = JSON.parse(raw) as Annotation;
        if (!data.boxed) return false; // not vetted yet -> not labelable
        return (data.boxes ?? []).some((b) => b.label == null);
      } catch {
        return false; // no JSON -> not boxed -> not in queue
      }
    }),
  );
  return manifest.filter((_, i) => flags[i]).map((e) => e.n);
}

/**
 * Global indices still needing box work: images NOT yet human-confirmed
 * (boxed !== true). Mirrors {@link getQueue} but for Box mode — drives the
 * "skip finished images" behavior when boxing in random order.
 */
export async function getBoxQueue(cats: CatId[] = ALL_CATS): Promise<number[]> {
  const manifest = filterByCats(getManifest(), cats);
  const flags = await Promise.all(manifest.map((e) => quickState(e.cat, e.name).then((s) => !s.boxed)));
  return manifest.filter((_, i) => flags[i]).map((e) => e.n);
}

/**
 * Where the home-page mode buttons should jump to:
 *  - box:   first image not yet human-confirmed (boxed !== true)
 *  - label: first image that still has an unlabeled box (queue head)
 * Falls back to 0 when everything is already done.
 */
export async function getEntryPoints(cats: CatId[] = ALL_CATS): Promise<{ box: number; label: number }> {
  // Scope to the selected categories but report GLOBAL manifest indices (the
  // path param), so the box target is the first un-boxed image among `cats`.
  const manifest = filterByCats(getManifest(), cats);
  const boxedFlags = await Promise.all(manifest.map((e) => quickState(e.cat, e.name).then((s) => s.boxed)));
  const firstUnboxed = manifest.find((_, i) => !boxedFlags[i]);
  const queue = await getQueue(cats);
  return { box: firstUnboxed?.n ?? manifest[0]?.n ?? 0, label: queue[0] ?? manifest[0]?.n ?? 0 };
}

/**
 * Global indices of images that contain at least one box labeled `label`
 * (scoped to `cats`). Drives review mode's navigation.
 */
export async function getReviewQueue(label: string, cats: CatId[] = ALL_CATS): Promise<number[]> {
  if (!label) return [];
  const manifest = filterByCats(getManifest(), cats);
  const flags = await Promise.all(
    manifest.map(async (e) => {
      const jsonPath = sidecarFsPath(e.cat, e.name, ".json");
      try {
        const raw = await fs.readFile(jsonPath, "utf8");
        const data = JSON.parse(raw) as Annotation;
        return (data.boxes ?? []).some((b) => b.label === label);
      } catch {
        return false;
      }
    }),
  );
  return manifest.filter((_, i) => flags[i]).map((e) => e.n);
}

export interface LabelStat {
  label: string;
  count: number;
}

/**
 * Total labeled boxes per roster label across all images, ranked desc. Every
 * roster label is included (even 0-count) so the home page can list them all.
 */
export async function getLabelStats(cats: CatId[] = ALL_CATS): Promise<LabelStat[]> {
  const manifest = filterByCats(getManifest(), cats);
  const counts = new Map<string, number>();
  await Promise.all(
    manifest.map(async (e) => {
      const jsonPath = sidecarFsPath(e.cat, e.name, ".json");
      try {
        const raw = await fs.readFile(jsonPath, "utf8");
        const data = JSON.parse(raw) as Annotation;
        for (const b of data.boxes ?? []) {
          if (b.label) counts.set(b.label, (counts.get(b.label) ?? 0) + 1);
        }
      } catch {
        // no annotation yet
      }
    }),
  );
  const stats = loadRoster().map((r) => ({ label: r.name, count: counts.get(r.name) ?? 0 }));
  stats.sort((a, b) => b.count - a.count);
  return stats;
}

export function imageExists(cat: CatId, name: string): boolean {
  try {
    return existsSync(imageFsPath(cat, name));
  } catch {
    return false;
  }
}

export { path };
