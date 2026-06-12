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
  type ReviewBox,
} from "./types";
import {
  categoryDir,
  deletedFsPath,
  deletedSidecarFsPath,
  imageFsPath,
  removedFsPath,
  removedSidecarFsPath,
  sidecarFsPath,
} from "./paths";
import { loadRoster, nameToIndex } from "./roster";
import { withFileLock } from "./file-lock";
import { dropFromCache } from "./hash-cache";

const IMG_RE = /\.(jpe?g|png)$/i;

/** Sorted image basenames in a category — the canonical listing rule (also used
 * by dedupe so it clusters exactly the file set the manifest indexes). */
export function listImages(cat: CatId): string[] {
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

/**
 * Drop the cached manifest. MUST be called whenever images are added or removed
 * from the raw tree (e.g. dedupe soft-delete/restore) — global indices `n` are
 * assigned by scanning the directories, so a stale cache would point Box/Label
 * mode at the wrong files.
 */
export function invalidateManifest(): void {
  manifestCache = null;
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

/**
 * Canonicalize a raw boxes array to the persisted shape: a stable id (falling
 * back to the positional `b-${i}` that the crop/box-op routes key on), the
 * numeric geometry, and an explicit null label. Single source of truth for
 * read, write, mutate, and the grid scan so their ids/labels never diverge.
 */
function normalizeBoxes(boxes: Box[] | undefined): Box[] {
  return (boxes ?? []).map((b, i) => ({
    id: b.id ?? `b-${i}`,
    cx: b.cx,
    cy: b.cy,
    w: b.w,
    h: b.h,
    label: b.label ?? null,
  }));
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
    return { boxed: Boolean(data.boxed), boxes: normalizeBoxes(data.boxes) };
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

/**
 * Persist an annotation: atomic JSON write + rewrite the YOLO .txt export.
 * Serialized per sidecar so concurrent writers (autosave PUT + grid box-ops)
 * can't lose updates to the same image.
 */
export async function writeAnnotation(cat: CatId, name: string, ann: Annotation): Promise<void> {
  const jsonPath = sidecarFsPath(cat, name, ".json");
  const txtPath = sidecarFsPath(cat, name, ".txt");
  const clean: Annotation = { boxed: Boolean(ann.boxed), boxes: normalizeBoxes(ann.boxes) };
  await withFileLock(jsonPath, () => persistAnnotationFiles(jsonPath, txtPath, clean));
}

/**
 * Atomic JSON write (temp + rename) + YOLO .txt mirror. The CALLER must hold the
 * file lock for `jsonPath`; shared by writeAnnotation and mutateAnnotation so the
 * on-disk format and the "empty .txt == reviewed negative" rule live in one place.
 */
async function persistAnnotationFiles(
  jsonPath: string,
  txtPath: string,
  clean: Annotation,
): Promise<void> {
  const tmp = `${jsonPath}.tmp-${process.pid}`;
  await fs.writeFile(tmp, JSON.stringify(clean, null, 2), "utf8");
  await fs.rename(tmp, jsonPath);
  await fs.writeFile(txtPath, toYolo(clean.boxes), "utf8");
}

/**
 * Atomic single-box mutation against an image's sidecar, serialized per file.
 * Reads the current annotation, applies `mut`, and writes back (regenerating the
 * .txt). Used by the grid review's box-op endpoint, where many boxes across many
 * images are unboxed/unlabeled without mounting the per-image annotation hook.
 */
export async function mutateAnnotation(
  cat: CatId,
  name: string,
  mut: (boxes: Box[]) => Box[],
): Promise<Annotation> {
  const jsonPath = sidecarFsPath(cat, name, ".json");
  const txtPath = sidecarFsPath(cat, name, ".txt");
  return withFileLock(jsonPath, async () => {
    const cur = await readAnnotation(cat, name);
    const clean: Annotation = { boxed: Boolean(cur.boxed), boxes: normalizeBoxes(mut(cur.boxes)) };
    // Persist inline: the caller already holds this file's lock, so calling
    // writeAnnotation (which re-acquires the same lock) would just queue behind us.
    await persistAnnotationFiles(jsonPath, txtPath, clean);
    return clean;
  });
}

// --- Reversible moves between the raw tree and a side tree -----------------
// Moving an image + its sidecars is the reversible alternative to deletion. Two
// side trees share this machinery: the dedupe loser tree (REMOVED_ROOT) and the
// manual-delete trash (DELETED_ROOT). An undo (or manual mv) brings files back.

const SIDECAR_EXTS = [".json", ".txt"] as const;

interface Transfer {
  src: string;
  dest: string;
  /** Filename for the result list (image or sidecar). */
  label: string;
}

/** Path resolvers for a side tree (removed/deleted), mirroring the raw helpers. */
interface SideTree {
  img: (cat: string, name: string) => string;
  sidecar: (cat: string, name: string, ext: string) => string;
}

const REMOVED_TREE: SideTree = { img: removedFsPath, sidecar: removedSidecarFsPath };
const DELETED_TREE: SideTree = { img: deletedFsPath, sidecar: deletedSidecarFsPath };

/**
 * Build the move pairs (only for files that exist) between the raw tree and the
 * given side tree. dir "out" = raw -> side (remove/delete); "in" = side -> raw
 * (restore). Image extension is derived by the *FsPath helpers; never assume ".jpg".
 */
function transferPairs(cat: CatId, name: string, side: SideTree, dir: "out" | "in"): Transfer[] {
  const pairs: Transfer[] = [];
  const add = (src: string, dest: string, label: string) => {
    if (existsSync(src)) pairs.push({ src, dest, label });
  };
  if (dir === "out") {
    add(imageFsPath(cat, name), side.img(cat, name), name);
    for (const ext of SIDECAR_EXTS) {
      add(sidecarFsPath(cat, name, ext), side.sidecar(cat, name, ext), name.replace(IMG_RE, ext));
    }
  } else {
    add(side.img(cat, name), imageFsPath(cat, name), name);
    for (const ext of SIDECAR_EXTS) {
      add(side.sidecar(cat, name, ext), sidecarFsPath(cat, name, ext), name.replace(IMG_RE, ext));
    }
  }
  return pairs;
}

/** Move one file: rename, falling back to copy+unlink across filesystems (EXDEV). */
async function moveFile(src: string, dest: string): Promise<void> {
  try {
    await fs.rename(src, dest);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "EXDEV") {
      await fs.copyFile(src, dest);
      await fs.unlink(src);
    } else {
      throw err;
    }
  }
}

/**
 * Move a bundle of files, refusing to clobber any existing destination. If a
 * move fails partway, roll back the moves already made (LIFO, best effort) so a
 * bundle is never left split across the raw and removed trees. EXDEV-safe.
 */
async function moveBundle(pairs: Transfer[]): Promise<string[]> {
  // Precheck all destinations first so a clobber aborts before anything moves.
  for (const p of pairs) {
    if (existsSync(p.dest)) throw new Error(`refusing to overwrite existing file: ${p.dest}`);
  }
  const moved: string[] = [];
  const done: Transfer[] = [];
  try {
    for (const p of pairs) {
      await fs.mkdir(path.dirname(p.dest), { recursive: true });
      await moveFile(p.src, p.dest);
      done.push(p);
      moved.push(p.label);
    }
  } catch (err) {
    for (const p of done.reverse()) {
      try {
        await moveFile(p.dest, p.src);
      } catch {
        /* leave this one where it landed; the original error is what matters */
      }
    }
    throw err;
  }
  return moved;
}

/**
 * Soft-delete an image: move its jpg/png + existing .json/.txt sidecars from the
 * raw tree into the removed tree, preserving the category subpath. Reversible via
 * {@link restoreImage}. Invalidates the manifest cache and drops the hash-cache
 * entry. Serialized per image (same lock key as the autosave writers).
 */
export async function removeImage(cat: CatId, name: string): Promise<{ moved: string[] }> {
  return withFileLock(sidecarFsPath(cat, name, ".json"), async () => {
    const moved = await moveBundle(transferPairs(cat, name, REMOVED_TREE, "out"));
    if (moved.length) {
      invalidateManifest();
      dropFromCache(cat, [name]);
    }
    return { moved };
  });
}

/** Reverse {@link removeImage}: move files removed -> raw. Refuses to clobber live files. */
export async function restoreImage(cat: CatId, name: string): Promise<{ moved: string[] }> {
  return withFileLock(sidecarFsPath(cat, name, ".json"), async () => {
    const moved = await moveBundle(transferPairs(cat, name, REMOVED_TREE, "in"));
    if (moved.length) invalidateManifest();
    return { moved };
  });
}

/**
 * Manually delete an image: move its jpg/png + existing .json/.txt sidecars from
 * the raw tree into the trash tree (DELETED_ROOT), preserving the category subpath.
 * Reversible via {@link restoreDeletedImage}. Invalidates the manifest cache and
 * drops the hash-cache entry (the image left the raw set). Same lock key as the
 * autosave writers, so it can't race a pending save for the same image.
 */
export async function deleteImage(cat: CatId, name: string): Promise<{ moved: string[] }> {
  return withFileLock(sidecarFsPath(cat, name, ".json"), async () => {
    const moved = await moveBundle(transferPairs(cat, name, DELETED_TREE, "out"));
    if (moved.length) {
      invalidateManifest();
      dropFromCache(cat, [name]);
    }
    return { moved };
  });
}

/** Reverse {@link deleteImage}: move files deleted -> raw. Refuses to clobber live files. */
export async function restoreDeletedImage(cat: CatId, name: string): Promise<{ moved: string[] }> {
  return withFileLock(sidecarFsPath(cat, name, ".json"), async () => {
    const moved = await moveBundle(transferPairs(cat, name, DELETED_TREE, "in"));
    if (moved.length) invalidateManifest();
    return { moved };
  });
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

/**
 * Every box whose label === `label` across `cats`, flattened into grid-review
 * cells in global manifest order (and box order within each image). Mirrors
 * {@link getReviewQueue}'s scan but returns the boxes themselves. The id fallback
 * matches {@link readAnnotation}, so the id here keys the crop + box-op routes.
 */
export async function getReviewBoxes(label: string, cats: CatId[] = ALL_CATS): Promise<ReviewBox[]> {
  if (!label) return [];
  const manifest = filterByCats(getManifest(), cats);
  const per = await Promise.all(
    manifest.map(async (e): Promise<ReviewBox[]> => {
      const jsonPath = sidecarFsPath(e.cat, e.name, ".json");
      try {
        const data = JSON.parse(await fs.readFile(jsonPath, "utf8")) as Annotation;
        return normalizeBoxes(data.boxes)
          .filter((b) => b.label === label)
          .map((box) => ({ n: e.n, cat: e.cat, name: e.name, box }));
      } catch {
        return [];
      }
    }),
  );
  return per.flat();
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
