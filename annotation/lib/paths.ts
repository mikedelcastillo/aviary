// Server-only path resolution + traversal guards. Never import from client code.
import path from "node:path";
import { CATEGORIES, type CatId } from "./types";

/**
 * Repo root inferred from cwd when not set via env. The launcher
 * (scripts/annotation.sh) runs `next dev` from the `annotation/` dir, so the
 * repo root is one level up. Env vars (absolute) always win.
 */
const REPO_ROOT = path.resolve(process.cwd(), "..");

export const DATA_ROOT =
  process.env.AVIARY_DATA_ROOT ?? path.join(REPO_ROOT, "data", "annotation", "raw");

export const ROSTER_PATH =
  process.env.AVIARY_ROSTER ?? path.join(REPO_ROOT, "training", "roster.yaml");

/** Parent of the raw set (…/data/annotation) — home for cache + removed trees. */
const ANNOTATION_ROOT = path.dirname(DATA_ROOT);

/** On-disk perceptual-hash cache for dedupe mode (sibling of raw/). */
export const DEDUPE_CACHE_PATH =
  process.env.AVIARY_DEDUPE_CACHE ?? path.join(ANNOTATION_ROOT, ".dedupe-cache.json");

/**
 * Soft-delete root for dedupe removals (sibling of raw/). Matches the original
 * Python prototype's default so both tools, if ever run, share one removed tree.
 */
export const REMOVED_ROOT =
  process.env.AVIARY_REMOVED_ROOT ?? path.join(ANNOTATION_ROOT, "_dedup_removed");

/** Directory of valid image basenames (no path separators, jpg/jpeg/png). */
const NAME_RE = /^[A-Za-z0-9._-]+\.(jpe?g|png)$/i;

export function isValidCat(cat: string): cat is CatId {
  return CATEGORIES.some((c) => c.id === cat);
}

export function isValidName(name: string): boolean {
  return NAME_RE.test(name) && !name.includes("..");
}

function catDir(cat: CatId): string {
  const def = CATEGORIES.find((c) => c.id === cat)!;
  return path.join(DATA_ROOT, def.dir);
}

/** Absolute path to an image file, guarded against traversal. Throws if invalid. */
export function imageFsPath(cat: string, name: string): string {
  if (!isValidCat(cat) || !isValidName(name)) {
    throw new Error("invalid category or filename");
  }
  const dir = catDir(cat);
  const full = path.resolve(dir, name);
  // Ensure the resolved path stays inside the category directory.
  if (full !== path.join(dir, name) || !full.startsWith(path.resolve(dir) + path.sep)) {
    throw new Error("path traversal blocked");
  }
  return full;
}

/** Absolute path to a sidecar with the given extension (".json" / ".txt"). */
export function sidecarFsPath(cat: string, name: string, ext: string): string {
  const img = imageFsPath(cat, name);
  return img.replace(/\.(jpe?g|png)$/i, ext);
}

export function categoryDir(cat: CatId): string {
  return catDir(cat);
}

// --- Soft-delete (removed) tree --------------------------------------------
// Mirror of imageFsPath/sidecarFsPath rooted under REMOVED_ROOT, with the same
// validation + traversal containment so the move/restore endpoints are safe.

function removedCatDir(cat: CatId): string {
  const def = CATEGORIES.find((c) => c.id === cat)!;
  return path.join(REMOVED_ROOT, def.dir);
}

/** Absolute path to an image inside the removed tree, guarded against traversal. */
export function removedFsPath(cat: string, name: string): string {
  if (!isValidCat(cat) || !isValidName(name)) {
    throw new Error("invalid category or filename");
  }
  const dir = removedCatDir(cat);
  const full = path.resolve(dir, name);
  if (full !== path.join(dir, name) || !full.startsWith(path.resolve(dir) + path.sep)) {
    throw new Error("path traversal blocked");
  }
  return full;
}

/** Sidecar path (".json" / ".txt") inside the removed tree. */
export function removedSidecarFsPath(cat: string, name: string, ext: string): string {
  const img = removedFsPath(cat, name);
  return img.replace(/\.(jpe?g|png)$/i, ext);
}
