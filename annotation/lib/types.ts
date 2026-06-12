// Shared data contract for the annotation tool. Pure types + constants only —
// NO node/fs imports here so this is safe to import from client components.

/** URL-safe category id (the `[cat]` route param). */
export type CatId = "day" | "ir" | "phone";

/** Which roster pill rule a category uses. */
export type PillRule = "day" | "ir" | "phone";

export interface CategoryDef {
  id: CatId;
  /** Directory relative to the data root (AVIARY_DATA_ROOT). */
  dir: string;
  /** Human title shown in the UI. */
  title: string;
  /** Short subtitle / source description. */
  subtitle: string;
  /** Roster pill rule for label mode. */
  pills: PillRule;
}

/** The three image categories, in global manifest order (day, ir, phone). */
export const CATEGORIES: readonly CategoryDef[] = [
  { id: "day", dir: "tapo/day", title: "Tapo · Day", subtitle: "Visible-light camera frames", pills: "day" },
  { id: "ir", dir: "tapo/ir", title: "Tapo · IR", subtitle: "Infrared / night frames", pills: "ir" },
  { id: "phone", dir: "phone", title: "Phone", subtitle: "Phone library photos", pills: "phone" },
] as const;

export function categoryById(id: string): CategoryDef | undefined {
  return CATEGORIES.find((c) => c.id === id);
}

// ---------------------------------------------------------------------------
// Category selection helpers. Box/Label navigation and the home-page summary
// can be scoped to a subset of categories. The selection rides in a `?cats=`
// query param (e.g. `?cats=day,ir`); the canonical "all three" state omits the
// param entirely so default URLs stay identical to the unfiltered tool.
// Pure + client-safe — no fs imports.
// ---------------------------------------------------------------------------

/** All category ids in canonical manifest order (day, ir, phone). */
export const ALL_CATS: CatId[] = CATEGORIES.map((c) => c.id);

function isCatId(v: string): v is CatId {
  return CATEGORIES.some((c) => c.id === v);
}

/**
 * Parse a `cats` query value into a valid, canonically-ordered, de-duped subset.
 * Empty / null / all-invalid input falls back to all three categories so a
 * missing or garbled param never strands the user with nothing to annotate.
 */
export function parseCats(param: string | null | undefined): CatId[] {
  if (!param) return [...ALL_CATS];
  const picked = new Set(
    param
      .split(",")
      .map((s) => s.trim())
      .filter(isCatId),
  );
  const out = ALL_CATS.filter((c) => picked.has(c));
  return out.length > 0 ? out : [...ALL_CATS];
}

/**
 * Serialize a category subset for a URL. Returns null when the selection equals
 * all three (canonical) so callers can omit the param; otherwise a canonically
 * ordered comma list, e.g. "day,ir".
 */
export function serializeCats(cats: CatId[]): string | null {
  const ordered = ALL_CATS.filter((c) => cats.includes(c));
  if (ordered.length === 0 || ordered.length === ALL_CATS.length) return null;
  return ordered.join(",");
}

/** Append `?cats=…` to a path, unless the selection is the canonical all-three. */
export function withCats(href: string, cats: CatId[]): string {
  const param = serializeCats(cats);
  return param ? `${href}?cats=${param}` : href;
}

/**
 * Parse the `?random` seed param. Returns a positive integer seed (random
 * order in effect) or null (sequential order) when absent/invalid.
 */
export function parseSeed(param: string | null | undefined): number | null {
  if (!param) return null;
  const n = Number(param);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : null;
}

/**
 * Append `?cats=…&random=…` to a path, omitting each param when not in effect so
 * default (all-cats, sequential) URLs stay identical to before. The single nav
 * helper used by Box/Label so the category scope AND shuffle seed both ride
 * along across page navigations.
 */
export function withNav(href: string, cats: CatId[], seed: number | null): string {
  const params = new URLSearchParams();
  const c = serializeCats(cats);
  if (c) params.set("cats", c);
  if (seed != null) params.set("random", String(seed));
  const qs = params.toString();
  return qs ? `${href}?${qs}` : href;
}

/**
 * Deterministic order for a list. With a null seed the original order is
 * returned unchanged; otherwise a Fisher–Yates shuffle driven by a mulberry32
 * PRNG yields a stable permutation for that seed — so every page load with the
 * same seed reproduces the same sequence, keeping prev/next coherent.
 */
export function orderBySeed<T>(items: T[], seed: number | null): T[] {
  if (seed == null) return items;
  const rng = mulberry32(seed);
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** A fresh positive seed for entering random mode. Client-side only. */
export function newSeed(): number {
  return Math.floor(Math.random() * 2_147_483_646) + 1;
}

/** Small, fast seeded PRNG. Deterministic for a given seed. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Keep only items whose `cat` is in the selected set. */
export function filterByCats<T extends { cat: CatId }>(items: T[], cats: CatId[]): T[] {
  const set = new Set(cats);
  return items.filter((it) => set.has(it.cat));
}

/**
 * Entry URL for review mode for a given label. Lands on global index 0; the
 * review page's deep-link guard snaps to the first image that actually contains
 * the label. Carries the category scope so review matches the leaderboard.
 */
export function reviewHref(label: string, cats: CatId[]): string {
  const params = new URLSearchParams({ label });
  const c = serializeCats(cats);
  if (c) params.set("cats", c);
  return `/review/0?${params.toString()}`;
}

/**
 * Grid (camera-roll) review for a label — the bird's-eye sibling of
 * {@link reviewHref}. Not tied to a manifest index; carries the same label +
 * category scope so the two views toggle in place.
 */
export function gridReviewHref(label: string, cats: CatId[]): string {
  const params = new URLSearchParams({ label });
  const c = serializeCats(cats);
  if (c) params.set("cats", c);
  return `/review/grid?${params.toString()}`;
}

/**
 * The catch-all roster label for birds that can't be confidently identified.
 * Shared between server (roster ordering) and client (review "mark unknown").
 */
export const UNKNOWN_LABEL = "unknown_bird";

/** A single bounding box in normalized YOLO geometry (center + size, 0..1). */
export interface Box {
  /** Stable client key — used for hover/queue/undo. NOT written to the YOLO .txt. */
  id: string;
  cx: number;
  cy: number;
  w: number;
  h: number;
  /** Roster label name, or null when the box is drawn but not yet labeled. */
  label: string | null;
}

/** Per-image annotation; the JSON sidecar shape on disk. */
export interface Annotation {
  /**
   * True only once a human has reviewed/confirmed this image's boxes in Box mode.
   * Auto-detection seed boxes do NOT make an image boxed.
   */
  boxed: boolean;
  boxes: Box[];
}

/** One labeled box plus the image it lives on — a single grid-review cell. */
export interface ReviewBox {
  /** Global manifest index of the image (path param for Focus Review). */
  n: number;
  cat: CatId;
  name: string;
  box: Box;
}

/** One entry in the global image manifest. */
export interface ManifestEntry {
  /** Global index across all categories (day first, then ir, then phone). */
  n: number;
  cat: CatId;
  /** Image filename including extension, e.g. "room-main_day_..._00123.jpg". */
  name: string;
}

export interface CategoryProgress {
  id: CatId;
  title: string;
  subtitle: string;
  total: number;
  boxed: number;
  labeled: number;
  /** Global index of the first image in this category (for entering modes). */
  startN: number;
}

// ---------------------------------------------------------------------------
// Dedupe mode: near-duplicate clusters. Client-safe (no fs) so both the API
// (lib/dedupe.ts) and the page (app/dedupe) share one contract.
// ---------------------------------------------------------------------------

/** Default pHash Hamming threshold — strict / near-identical only. */
export const DEDUPE_DEFAULT_THRESHOLD = 3;

/** One image within a near-duplicate cluster. */
export interface DedupeMember {
  /** Image filename including extension. */
  name: string;
  /** True when a non-empty YOLO .txt sidecar exists (has real labels). */
  hasLabels: boolean;
  /** Source jpg size in bytes (sharpness proxy for keep-selection). */
  sizeBytes: number;
  /** pHash Hamming distance to the cluster anchor. */
  dist: number;
}

/** A cluster of near-duplicate images within a single category. */
export interface DedupeCluster {
  cat: CatId;
  /** Suggested image to keep (the rest are removal candidates). */
  keepName: string;
  members: DedupeMember[];
}

/** A label option rendered as a pill in label mode. */
export interface Pill {
  /** Roster label name. */
  label: string;
  /** Global roster class index (file position) — written to the YOLO .txt. */
  globalIndex: number;
  /** Keyboard shortcut character, e.g. "1" or "q". */
  shortcut: string;
  /** species | individual | unknown (for subtle styling if desired). */
  kind: string;
}

/** Save lifecycle, surfaced in the browser tab title. */
export type SaveStatus = "idle" | "dirty" | "saving" | "saved";

// ---------------------------------------------------------------------------
// Canvas component contracts (implemented in components/Stage.tsx + hooks).
// Box mode and Label mode pages code against these; do not change the shapes
// without updating all consumers.
// ---------------------------------------------------------------------------

/** Pan/zoom transform: screen = translate(tx,ty) then scale(scale) of image px. */
export interface Transform {
  scale: number;
  tx: number;
  ty: number;
}

/** A point. Used for both screen px and image px depending on context. */
export interface Pt {
  x: number;
  y: number;
}

/** A normalized rect (top-left + size, 0..1) — used to frame a box in label mode. */
export interface NormRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Imperative handle exposed by <Stage> via ref. All conversions use the live
 * transform. Image coordinates are in source pixels (0..naturalWidth/Height).
 */
export interface StageHandle {
  /** Convert a screen/client point to image-pixel coordinates. */
  screenToImage(clientX: number, clientY: number): Pt;
  /** Convert an image-pixel point to screen/client coordinates. */
  imageToScreen(x: number, y: number): Pt;
  getTransform(): Transform;
  setTransform(t: Transform): void;
  /** Natural image size in px (0,0 until the image has loaded). */
  getNaturalSize(): { width: number; height: number };
  /** Fit the whole image within the viewport (contain). */
  fit(): void;
  /** Center + zoom so the normalized rect fills ~`fill` fraction of the viewport. */
  focusRect(rect: NormRect, fill?: number): void;
}
