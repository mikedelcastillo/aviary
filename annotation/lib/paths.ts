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

/** Directory of trained model weights (`<series>-NNN.pt`) + benchmark results. */
export const MODELS_DIR =
  process.env.AVIARY_MODELS_DIR ?? path.join(REPO_ROOT, "data", "models");

/** Benchmark results file written by `scripts/benchmark.*` (training/scripts/benchmark.py). */
export const BENCHMARK_PATH =
  process.env.AVIARY_BENCHMARK ?? path.join(MODELS_DIR, "benchmark.json");

/** Parent of the raw set (…/data/annotation) — home for cache + removed trees. */
const ANNOTATION_ROOT = path.dirname(DATA_ROOT);

/** On-disk perceptual-hash cache for dedupe mode (sibling of raw/). */
export const DEDUPE_CACHE_PATH =
  process.env.AVIARY_DEDUPE_CACHE ?? path.join(ANNOTATION_ROOT, ".dedupe-cache.json");

/**
 * Soft-delete root for dedupe removals (sibling of raw/). The dedupe loser tree;
 * kept distinct from the manual-delete tree ({@link DELETED_ROOT}).
 */
export const REMOVED_ROOT =
  process.env.AVIARY_REMOVED_ROOT ?? path.join(ANNOTATION_ROOT, "dedup");

/**
 * Trash root for manual image deletions from Box/Label/Review (sibling of raw/).
 * Separate from {@link REMOVED_ROOT} because "manually deleted" and "deduped" are
 * different intents; both are reversible moves, never destructive unlinks.
 */
export const DELETED_ROOT =
  process.env.AVIARY_DELETED_ROOT ?? path.join(ANNOTATION_ROOT, "deleted");

/** Directory of valid image basenames (no path separators, jpg/jpeg/png). */
const NAME_RE = /^[A-Za-z0-9._-]+\.(jpe?g|png)$/i;

export function isValidCat(cat: string): cat is CatId {
  return CATEGORIES.some((c) => c.id === cat);
}

export function isValidName(name: string): boolean {
  return NAME_RE.test(name) && !name.includes("..");
}

/** Category subdirectory name (e.g. "camera_frames/day") for a valid category. */
function catSubdir(cat: CatId): string {
  return CATEGORIES.find((c) => c.id === cat)!.dir;
}

function catDir(cat: CatId): string {
  return path.join(DATA_ROOT, catSubdir(cat));
}

/**
 * Resolve `name` for category `cat` under `root`, validating the inputs and
 * containing the result to the category directory (traversal guard). Shared by
 * the raw tree ({@link imageFsPath}) and the removed tree ({@link removedFsPath})
 * so the containment check lives in exactly one place.
 */
function resolveInTree(root: string, cat: string, name: string): string {
  if (!isValidCat(cat) || !isValidName(name)) {
    throw new Error("invalid category or filename");
  }
  const dir = path.join(root, catSubdir(cat));
  const full = path.resolve(dir, name);
  if (full !== path.join(dir, name) || !full.startsWith(path.resolve(dir) + path.sep)) {
    throw new Error("path traversal blocked");
  }
  return full;
}

/** Absolute path to an image file, guarded against traversal. Throws if invalid. */
export function imageFsPath(cat: string, name: string): string {
  return resolveInTree(DATA_ROOT, cat, name);
}

/** Absolute path to a sidecar with the given extension (".json" / ".txt"). */
export function sidecarFsPath(cat: string, name: string, ext: string): string {
  return imageFsPath(cat, name).replace(/\.(jpe?g|png)$/i, ext);
}

export function categoryDir(cat: CatId): string {
  return catDir(cat);
}

// --- Soft-delete (removed) tree --------------------------------------------
// Mirror of imageFsPath/sidecarFsPath rooted under REMOVED_ROOT, sharing the
// same validation + traversal containment via resolveInTree.

/** Absolute path to an image inside the removed tree, guarded against traversal. */
export function removedFsPath(cat: string, name: string): string {
  return resolveInTree(REMOVED_ROOT, cat, name);
}

/** Sidecar path (".json" / ".txt") inside the removed tree. */
export function removedSidecarFsPath(cat: string, name: string, ext: string): string {
  return removedFsPath(cat, name).replace(/\.(jpe?g|png)$/i, ext);
}

// --- Manual-delete (trash) tree --------------------------------------------
// Mirror of imageFsPath/sidecarFsPath rooted under DELETED_ROOT, sharing the
// same validation + traversal containment via resolveInTree.

/** Absolute path to an image inside the deleted tree, guarded against traversal. */
export function deletedFsPath(cat: string, name: string): string {
  return resolveInTree(DELETED_ROOT, cat, name);
}

/** Sidecar path (".json" / ".txt") inside the deleted tree. */
export function deletedSidecarFsPath(cat: string, name: string, ext: string): string {
  return deletedFsPath(cat, name).replace(/\.(jpe?g|png)$/i, ext);
}
