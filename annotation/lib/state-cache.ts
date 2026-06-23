// Unified per-image state cache — the read-side optimization for the home page.
//
// progress / queue / box-queue / entry / label-stats each used to independently
// scan the WHOLE manifest, re-reading and re-parsing the same `<image>.json`
// sidecars every request (no warm speedup). This module reads+parses each sidecar
// ONCE and caches a compact summary keyed by the file's mtime, so every view
// derives from one shared cache and warm scans skip readFile+JSON.parse entirely.
//
// Correctness with external writers: each lookup still stats the file (cheap, and
// bounded by fs-limit), so an edit by the suggest/training scripts is detected via
// mtime and the entry is re-read. The filesystem stays the source of truth.
import type { Annotation, CatId } from "./types";
import { sidecarFsPath } from "./paths";
import { statLimited, readTextLimited } from "./fs-limit";

/** Compact, derivable summary of one image's `.json` sidecar. */
export interface JsonState {
  /** Human-confirmed in Box mode. */
  boxed: boolean;
  /** Boxed AND every box carries a label. */
  labeled: boolean;
  /** Has at least one box still missing a label. */
  hasUnlabeled: boolean;
  /** Labels of every labeled box (repeats kept, so label-stats can count). */
  boxLabels: string[];
}

const EMPTY: JsonState = { boxed: false, labeled: false, hasUnlabeled: false, boxLabels: [] };

const jsonCache = new Map<string, { mtimeMs: number; state: JsonState }>();
const suggestCache = new Map<string, { mtimeMs: number; count: number }>();
const suggestLabelsCache = new Map<string, { mtimeMs: number; labels: string[] }>();

/** Generic box proposals carry no roster label — bucket them under this name. */
const GENERIC_SUGGEST_LABEL = "bird";

// Warn at most once per corrupt file so a single bad sidecar doesn't spam logs.
const warnedCorrupt = new Set<string>();

function key(cat: CatId, name: string): string {
  return `${cat}/${name}`;
}

function isErrno(e: unknown): e is NodeJS.ErrnoException {
  return typeof e === "object" && e !== null && "code" in e;
}

/**
 * Summarized `.json` state for one image, mtime-validated and cached. A missing
 * sidecar (ENOENT) is the normal "not annotated yet" case -> EMPTY. A sidecar
 * that exists but fails to parse is CORRUPTION: we log it once and treat it as
 * EMPTY for read-only scans, but never cache that (so it re-checks), and the
 * editor's readAnnotation surfaces it instead of silently seeding over it.
 */
export async function getJsonState(cat: CatId, name: string): Promise<JsonState> {
  const p = sidecarFsPath(cat, name, ".json");
  const k = key(cat, name);
  let st;
  try {
    st = await statLimited(p);
  } catch {
    jsonCache.delete(k); // absent -> evict any stale entry
    return EMPTY;
  }
  const hit = jsonCache.get(k);
  if (hit && hit.mtimeMs === st.mtimeMs) return hit.state;

  let raw: string;
  try {
    raw = await readTextLimited(p);
  } catch (e) {
    if (isErrno(e) && e.code === "ENOENT") {
      jsonCache.delete(k);
      return EMPTY; // raced with a delete — treat as absent
    }
    return EMPTY; // transient read error — don't cache
  }
  let state: JsonState;
  try {
    const data = JSON.parse(raw) as Annotation;
    const boxes = data.boxes ?? [];
    const boxed = Boolean(data.boxed);
    const hasUnlabeled = boxes.some((b) => b.label == null);
    const boxLabels: string[] = [];
    for (const b of boxes) if (b.label) boxLabels.push(b.label);
    state = { boxed, labeled: boxed && !hasUnlabeled, hasUnlabeled, boxLabels };
  } catch {
    if (!warnedCorrupt.has(k)) {
      warnedCorrupt.add(k);
      console.warn(`[state-cache] corrupt annotation, treating as empty for scans: ${k}.json`);
    }
    return EMPTY; // present-but-corrupt — don't cache; readAnnotation will surface it
  }
  jsonCache.set(k, { mtimeMs: st.mtimeMs, state });
  return state;
}

/** Pending suggestion count from `<image>.suggest.json`, mtime-validated + cached. */
export async function getSuggestCount(cat: CatId, name: string): Promise<number> {
  const p = sidecarFsPath(cat, name, ".suggest.json");
  const k = key(cat, name);
  let st;
  try {
    st = await statLimited(p);
  } catch {
    suggestCache.delete(k);
    return 0;
  }
  const hit = suggestCache.get(k);
  if (hit && hit.mtimeMs === st.mtimeMs) return hit.count;
  let count = 0;
  try {
    const data = JSON.parse(await readTextLimited(p)) as { boxes?: unknown[] };
    count = Array.isArray(data.boxes) ? data.boxes.length : 0;
  } catch {
    return 0; // absent/garbled — suggestions are an optional layer
  }
  suggestCache.set(k, { mtimeMs: st.mtimeMs, count });
  return count;
}

/**
 * Distinct suggested LABELS for one image from `<image>.suggest.json`,
 * mtime-validated + cached (sibling of {@link getSuggestCount}). The set is the
 * union of `boxes[].label` (null/empty -> the generic bucket "bird") and
 * `labels[].label` (always a roster name). Empty array if no/garbled sidecar.
 */
export async function getSuggestLabels(cat: CatId, name: string): Promise<string[]> {
  const p = sidecarFsPath(cat, name, ".suggest.json");
  const k = key(cat, name);
  let st;
  try {
    st = await statLimited(p);
  } catch {
    suggestLabelsCache.delete(k);
    return [];
  }
  const hit = suggestLabelsCache.get(k);
  if (hit && hit.mtimeMs === st.mtimeMs) return hit.labels;
  let labels: string[] = [];
  try {
    const data = JSON.parse(await readTextLimited(p)) as {
      boxes?: Array<{ label?: string | null }>;
      labels?: Array<{ label?: string | null }>;
    };
    const set = new Set<string>();
    if (Array.isArray(data.boxes)) {
      for (const b of data.boxes) {
        const lab = b.label ? b.label : GENERIC_SUGGEST_LABEL;
        set.add(lab);
      }
    }
    if (Array.isArray(data.labels)) {
      for (const l of data.labels) if (l.label) set.add(l.label);
    }
    labels = [...set];
  } catch {
    return []; // absent/garbled — suggestions are an optional layer
  }
  suggestLabelsCache.set(k, { mtimeMs: st.mtimeMs, labels });
  return labels;
}

/** Drop one image's cached state (call after the app writes its sidecars). */
export function invalidateState(cat: CatId, name: string): void {
  jsonCache.delete(key(cat, name));
  suggestCache.delete(key(cat, name));
  suggestLabelsCache.delete(key(cat, name));
}
