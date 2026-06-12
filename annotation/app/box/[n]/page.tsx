"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Stage } from "@/components/Stage";
import { BoxLayer } from "@/components/BoxLayer";
import { DeleteToast } from "@/components/DeleteToast";
import { useAnnotation } from "@/lib/use-annotation";
import { useImageDelete } from "@/lib/use-image-delete";
import { LoadingOverlay } from "@/components/LoadingOverlay";
import { Spinner } from "@/components/Spinner";
import {
  categoryById,
  filterByCats,
  newSeed,
  orderBySeed,
  parseCats,
  parseSeed,
  serializeCats,
  withNav,
  type CatId,
  type ManifestEntry,
  type NormRect,
  type Pt,
  type StageHandle,
} from "@/lib/types";

const MIN_DRAG = 0.005;

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Convert two image-pixel points to a clamped normalized rect (top-left + size). */
function rectFromPoints(a: Pt, b: Pt, w: number, h: number): NormRect {
  const x = Math.min(a.x, b.x) / w;
  const y = Math.min(a.y, b.y) / h;
  const rw = Math.abs(b.x - a.x) / w;
  const rh = Math.abs(b.y - a.y) / h;
  const nx = clamp01(x);
  const ny = clamp01(y);
  return {
    x: nx,
    y: ny,
    w: clamp01(nx + rw) - nx,
    h: clamp01(ny + rh) - ny,
  };
}

export default function BoxPage() {
  const router = useRouter();
  const n = Number(useParams().n);
  const searchParams = useSearchParams();
  const cats = useMemo(() => parseCats(searchParams.get("cats")), [searchParams]);
  const seed = useMemo(() => parseSeed(searchParams.get("random")), [searchParams]);

  const [manifest, setManifest] = useState<ManifestEntry[] | null>(null);
  const [draft, setDraft] = useState<NormRect | null>(null);
  // In random mode, restrict navigation to images that still need boxing.
  const [boxQueue, setBoxQueue] = useState<number[] | null>(null);

  const stageRef = useRef<StageHandle>(null);
  const startRef = useRef<Pt | null>(null);

  // Load the global manifest into state and return it (so a post-delete handler
  // can navigate against the freshly-renumbered indices). Used on mount and after
  // a delete/undo, which renumber the server manifest.
  const loadManifest = useCallback(async (): Promise<ManifestEntry[]> => {
    try {
      const res = await fetch("/api/manifest");
      const data = (await res.json()) as ManifestEntry[];
      setManifest(data);
      return data;
    } catch {
      setManifest([]);
      return [];
    }
  }, []);

  // Fetch the global manifest once.
  useEffect(() => {
    void loadManifest();
  }, [loadManifest]);

  // In random mode, fetch the set of still-unboxed images so the shuffle skips
  // ones already done. Sequential mode shows everything, so no fetch needed.
  //
  // This snapshot is captured ONCE per (seed, category-scope) and deliberately
  // not refreshed on navigation. Depending on the `cats` array would re-run the
  // effect on every route change — it's a fresh array reference each render — and
  // since boxing an image drops it from the queue, the refetched list shrinks.
  // `orderBySeed` is a function of the list's contents, so a shorter list yields a
  // wholly different permutation: prev/next would jump to seemingly random images.
  // A stable string key (value-compared in the deps) freezes the worklist so the
  // seeded order stays put and prev/next walk a coherent sequence.
  const catsKey = serializeCats(cats);
  // Load (or refresh) the still-unboxed worklist for random mode. Called on
  // seed/scope change and after a delete/undo (a deletion legitimately changes
  // the worklist; the reshuffle that implies is acceptable for that rare action).
  const loadBoxQueue = useCallback(async (): Promise<number[] | null> => {
    if (seed == null) {
      setBoxQueue(null);
      return null;
    }
    const qs = catsKey ? `?cats=${catsKey}` : "";
    try {
      const res = await fetch(`/api/box-queue${qs}`);
      const data = (await res.json()) as number[];
      setBoxQueue(data);
      return data;
    } catch {
      setBoxQueue([]);
      return [];
    }
  }, [seed, catsKey]);

  useEffect(() => {
    void loadBoxQueue();
  }, [loadBoxQueue]);

  // The navigation sequence. Sequential: every in-scope image in global order.
  // Random: only still-unboxed images, in a stable seeded shuffle. The counter
  // + prev/next walk `ordered`; the path index `n` stays global.
  const filtered = useMemo(() => (manifest ? filterByCats(manifest, cats) : []), [manifest, cats]);
  const ordered = useMemo(() => {
    if (seed == null) return filtered;
    if (boxQueue == null) return []; // queue still loading
    const set = new Set(boxQueue);
    return orderBySeed(filtered.filter((e) => set.has(e.n)), seed);
  }, [filtered, seed, boxQueue]);
  const total = ordered.length;
  const globalInRange = manifest != null && Number.isInteger(n) && n >= 0 && n < manifest.length;
  const entry = globalInRange ? manifest![n] : null;
  const inCats = entry != null && cats.includes(entry.cat);
  const pos = entry && inCats ? ordered.findIndex((e) => e.n === n) : -1;

  // Deep-link guard: if `n` isn't a valid stop in the nav sequence (wrong
  // category, or — in random mode — already boxed), snap to the first in-scope
  // image at or after it (else the first in the sequence).
  useEffect(() => {
    if (manifest == null || ordered.length === 0) return;
    if (pos >= 0) return;
    const target = ordered.find((e) => e.n >= n) ?? ordered[0];
    router.replace(withNav(`/box/${target.n}`, cats, seed));
  }, [manifest, ordered, pos, n, cats, seed, router]);

  const cat: CatId | null = entry && inCats ? entry.cat : null;
  const name: string | null = entry && inCats ? entry.name : null;

  const { annotation, addBox, removeBox, replaceBoxes, setBoxed, undo, redo, loading } = useAnnotation(cat, name);

  // Manual image delete (whole image -> trash tree) with an undo toast.
  const { pending: pendingDelete, remove: deleteCurrent, undo: undoDelete, dismiss: dismissDelete } =
    useImageDelete();

  // --- Image readiness (drives the loading overlay). ------------------------
  const [ready, setReady] = useState(false);
  const onReady = useCallback(() => setReady(true), []);
  useEffect(() => {
    setReady(false);
  }, [cat, name]);

  // --- Navigation (within the selected categories) --------------------------
  const goNext = useCallback(() => {
    // Advancing confirms the image — but `boxed` must always mean "the user
    // placed boxes here", so only mark it boxed when boxes actually exist
    // (and revert a stale flag if they were all removed). On the last image
    // there's nothing to advance to, so head home.
    const hasBoxes = (annotation?.boxes.length ?? 0) > 0;
    if ((annotation?.boxed ?? false) !== hasBoxes) setBoxed(hasBoxes);
    const idx = ordered.findIndex((e) => e.n === n);
    const next = idx >= 0 ? ordered[idx + 1] : undefined;
    router.push(next ? withNav(`/box/${next.n}`, cats, seed) : "/");
  }, [ordered, n, annotation?.boxes.length, annotation?.boxed, setBoxed, router, cats, seed]);

  const goPrev = useCallback(() => {
    const idx = ordered.findIndex((e) => e.n === n);
    const prev = idx > 0 ? ordered[idx - 1] : undefined;
    if (!prev) return;
    router.push(withNav(`/box/${prev.n}`, cats, seed));
  }, [ordered, n, router, cats, seed]);

  // --- Undo/redo: history is session-global, so an edit made before advancing
  // is undone by hopping back to the image it happened on. ------------------
  const navToHistory = useCallback(
    (key: { cat: CatId; name: string }) => {
      const target = manifest?.find((e) => e.cat === key.cat && e.name === key.name);
      if (target) router.push(withNav(`/box/${target.n}`, cats, seed));
    },
    [manifest, router, cats, seed],
  );
  const handleUndo = useCallback(() => {
    const target = undo();
    if (target) navToHistory(target);
  }, [undo, navToHistory]);
  const handleRedo = useCallback(() => {
    const target = redo();
    if (target) navToHistory(target);
  }, [redo, navToHistory]);

  // Flip random/sequential order in place (no need to bounce through Home).
  const toggleRandom = useCallback(() => {
    router.replace(withNav(`/box/${n}`, cats, seed != null ? null : newSeed()));
  }, [router, n, cats, seed]);

  // Wipe every box on the current image. Emptying it reverts `boxed` so the
  // flag keeps meaning "has user boxes".
  const clearBoxes = useCallback(() => {
    if (!annotation?.boxes.length) return;
    replaceBoxes([]);
    if (annotation.boxed) setBoxed(false);
  }, [annotation?.boxes.length, annotation?.boxed, replaceBoxes, setBoxed]);

  // Delete a single box; if it was the last one, the image is no longer boxed.
  const handleRemoveBox = useCallback(
    (id: string) => {
      removeBox(id);
      if (annotation?.boxes.length === 1 && annotation.boxed) setBoxed(false);
    },
    [removeBox, annotation?.boxes.length, annotation?.boxed, setBoxed],
  );

  // --- Delete the whole image (move to trash), then advance. Undo restores it
  // and hops back. Navigation is by image identity against a freshly-refetched
  // manifest, since deleting renumbers the global indices the URL uses. --------
  const handleDelete = useCallback(async () => {
    if (!cat || !name) return;
    const idx = ordered.findIndex((e) => e.n === n);
    const nextEntry = idx >= 0 ? ordered[idx + 1] : undefined;
    const nextId = nextEntry ? { cat: nextEntry.cat, name: nextEntry.name } : null;
    await deleteCurrent(cat, name, async () => {
      const fresh = await loadManifest();
      await loadBoxQueue();
      const t = nextId ? fresh.find((e) => e.cat === nextId.cat && e.name === nextId.name) : undefined;
      router.push(t ? withNav(`/box/${t.n}`, cats, seed) : "/");
    });
  }, [cat, name, ordered, n, deleteCurrent, loadManifest, loadBoxQueue, router, cats, seed]);

  const handleDeleteUndo = useCallback(() => {
    void undoDelete(async (p) => {
      const fresh = await loadManifest();
      await loadBoxQueue();
      const t = fresh.find((e) => e.cat === p.cat && e.name === p.name);
      if (t) router.push(withNav(`/box/${t.n}`, cats, seed));
    });
  }, [undoDelete, loadManifest, loadBoxQueue, router, cats, seed]);

  // --- Drawing --------------------------------------------------------------
  const onDrawStart = useCallback((pt: Pt) => {
    startRef.current = pt;
    setDraft(null);
  }, []);

  const onDrawMove = useCallback((pt: Pt) => {
    const start = startRef.current;
    const size = stageRef.current?.getNaturalSize();
    if (!start || !size || size.width === 0 || size.height === 0) return;
    setDraft(rectFromPoints(start, pt, size.width, size.height));
  }, []);

  const onDrawEnd = useCallback(
    (pt: Pt) => {
      const start = startRef.current;
      startRef.current = null;
      const size = stageRef.current?.getNaturalSize();
      setDraft(null);
      if (!start || !size || size.width === 0 || size.height === 0) return;
      const r = rectFromPoints(start, pt, size.width, size.height);
      if (r.w > MIN_DRAG && r.h > MIN_DRAG) {
        addBox({ cx: r.x + r.w / 2, cy: r.y + r.h / 2, w: r.w, h: r.h, label: null });
        // Drawing a box is itself a confirmation — mark the image boxed right
        // away so it counts even if the user never advances (e.g. last image).
        if (!annotation?.boxed) setBoxed(true);
      }
    },
    [addBox, setBoxed, annotation?.boxed],
  );

  // --- Keyboard -------------------------------------------------------------
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || target?.isContentEditable) return;

      if (e.key === "Escape") {
        e.preventDefault();
        router.push("/");
        return;
      }

      const mod = e.metaKey || e.ctrlKey;
      if (mod && (e.key === "z" || e.key === "Z")) {
        e.preventDefault();
        if (e.shiftKey) handleRedo();
        else handleUndo();
        return;
      }
      if (mod && (e.key === "y" || e.key === "Y")) {
        e.preventDefault();
        handleRedo();
        return;
      }
      if (mod && e.key === "Backspace") {
        e.preventDefault();
        void handleDelete();
        return;
      }
      if (e.key === "ArrowRight" || e.key === " " || e.code === "Space") {
        e.preventDefault();
        goNext();
        return;
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        goPrev();
        return;
      }
      if (e.key === "f" || e.key === "F") {
        stageRef.current?.fit();
        return;
      }
      if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        toggleRandom();
        return;
      }
      if (e.key === "c" || e.key === "C") {
        e.preventDefault();
        clearBoxes();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleUndo, handleRedo, handleDelete, goNext, goPrev, toggleRandom, clearBoxes, router]);

  // --- Render guards --------------------------------------------------------
  if (
    manifest == null ||
    (seed != null && boxQueue == null) ||
    (entry != null && !inCats && filtered.length > 0)
  ) {
    // Loading, or briefly mid-redirect to an in-scope image.
    return (
      <main className="fixed inset-0 flex items-center justify-center bg-bg">
        <Spinner size={22} className="text-muted" />
      </main>
    );
  }

  if (!entry || !inCats || !cat || !name) {
    return (
      <main className="fixed inset-0 bg-bg flex flex-col items-center justify-center gap-3">
        <span className="text-sm text-muted">
          {total === 0 ? "No images in the selected categories." : `Image ${Number.isInteger(n) ? n + 1 : "?"} is out of range.`}
        </span>
        <Link href="/" className="text-xs text-faint hover:text-fg transition-colors">
          ← Home
        </Link>
      </main>
    );
  }

  const src = `/api/image/${cat}/${encodeURIComponent(name)}`;
  const title = categoryById(cat)?.title ?? cat;

  return (
    <main className="fixed inset-0 bg-bg">
      <Stage
        ref={stageRef}
        src={src}
        drawingEnabled
        onReady={onReady}
        onDrawStart={onDrawStart}
        onDrawMove={onDrawMove}
        onDrawEnd={onDrawEnd}
      >
        <BoxLayer boxes={annotation?.boxes ?? []} draft={draft} showDelete onDelete={handleRemoveBox} />
      </Stage>

      <LoadingOverlay show={!ready || loading} label="Loading image…" />

      {/* Top-left: home + context. */}
      <div className="fixed left-4 top-4 flex items-center gap-3 text-xs text-muted">
        <Link href="/" className="text-faint hover:text-fg transition-colors">
          ← Home
        </Link>
        <span className="text-border-strong">|</span>
        <span>{title}</span>
        <span className="max-w-[18rem] truncate text-faint" title={name}>
          {name}
        </span>
        <span className="text-border-strong">|</span>
        <button
          type="button"
          onClick={toggleRandom}
          title="Toggle random / sequential order (R)"
          className="text-faint hover:text-fg transition-colors"
        >
          {seed != null ? "random" : "sequential"}
        </button>
      </div>

      {/* Top-right: boxed indicator. */}
      <div className="fixed right-4 top-4 text-xs font-mono">
        {annotation?.boxed ? (
          <span className="text-box">boxed ✓</span>
        ) : (
          <span className="text-faint">unboxed</span>
        )}
      </div>

      {/* Bottom-center: nav HUD. */}
      <div className="fixed bottom-6 left-1/2 flex -translate-x-1/2 items-center gap-2">
        {(annotation?.boxes.length ?? 0) > 0 && (
          <div className="flex items-center gap-2 rounded-2xl border border-border bg-surface/85 px-3 py-3 backdrop-blur-md">
            <button
              type="button"
              onClick={clearBoxes}
              title="Clear all boxes (C)"
              className="flex cursor-pointer items-center gap-2 rounded-pill border border-danger/40 bg-danger/10 px-3 py-1.5 text-sm text-danger transition-colors hover:border-danger/70"
            >
              <kbd className="grid h-5 min-w-5 place-items-center rounded border border-danger/50 px-1 font-mono text-[11px] uppercase text-danger">
                C
              </kbd>
              <span className="font-medium">Clear</span>
            </button>
          </div>
        )}
        <div className="flex items-center gap-2 rounded-2xl border border-border bg-surface/85 px-3 py-3 backdrop-blur-md">
          <button
            type="button"
            onClick={() => void handleDelete()}
            title="Delete this image (⌘/Ctrl + Backspace)"
            className="flex cursor-pointer items-center gap-2 rounded-pill border border-danger/40 bg-danger/10 px-3 py-1.5 text-sm text-danger transition-colors hover:border-danger/70"
          >
            <kbd className="grid h-5 min-w-5 place-items-center rounded border border-danger/50 px-1 font-mono text-[11px] text-danger">
              ⌘⌫
            </kbd>
            <span className="font-medium">Delete</span>
          </button>
        </div>
        <div className="flex items-center gap-1 rounded-full border border-border bg-surface/85 px-2 py-1.5 backdrop-blur">
          <button
            type="button"
            onClick={goPrev}
            disabled={pos <= 0}
            aria-label="Previous image"
            className="flex h-7 w-7 items-center justify-center rounded-full text-muted transition-colors hover:bg-elevated hover:text-fg disabled:pointer-events-none disabled:opacity-30"
          >
            ‹
          </button>
          <span className="px-2 font-mono text-xs tabular-nums text-fg">
            {pos + 1} / {total}
          </span>
          <button
            type="button"
            onClick={goNext}
            disabled={pos < 0}
            aria-label={pos === total - 1 ? "Confirm & finish" : "Next image (confirm)"}
            className="flex h-7 w-7 items-center justify-center rounded-full text-muted transition-colors hover:bg-elevated hover:text-fg disabled:pointer-events-none disabled:opacity-30"
          >
            ›
          </button>
        </div>
      </div>

      {pendingDelete && (
        <DeleteToast
          name={pendingDelete.name}
          onUndo={handleDeleteUndo}
          onDismiss={dismissDelete}
        />
      )}
    </main>
  );
}
