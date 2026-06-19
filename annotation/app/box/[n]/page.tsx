"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Stage } from "@/components/Stage";
import { BoxLayer } from "@/components/BoxLayer";
import { SuggestionLayer } from "@/components/SuggestionLayer";
import { DeleteToast } from "@/components/DeleteToast";
import { BottomBar } from "@/components/BottomBar";
import { ActionButton } from "@/components/ActionButton";
import { UndoIcon, RedoIcon } from "@/components/icons";
import { NavCluster } from "@/components/NavCluster";
import { useAnnotation } from "@/lib/use-annotation";
import { useSuggestions } from "@/lib/use-suggestions";
import { useImageDelete } from "@/lib/use-image-delete";
import { useCoarsePointer } from "@/hooks/useCoarsePointer";
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
  const coarse = useCoarsePointer();

  const [manifest, setManifest] = useState<ManifestEntry[] | null>(null);
  const [draft, setDraft] = useState<NormRect | null>(null);
  // The set of still-unboxed images. Used ONLY as a membership Set so Space
  // jumps to the next image that still needs boxing — it does NOT drive the
  // navigation order, so it can refetch freely without disturbing prev/next.
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

  // Fetch the still-unboxed worklist for the current category scope. This only
  // feeds Space's "jump to next unboxed" — navigation order is a pure function of
  // the full manifest + seed, so a stale/shrinking queue can no longer scramble
  // prev/next. Fetched on scope change and after a delete/undo.
  const catsKey = serializeCats(cats);
  const loadBoxQueue = useCallback(async (): Promise<number[]> => {
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
  }, [catsKey]);

  useEffect(() => {
    void loadBoxQueue();
  }, [loadBoxQueue]);

  // The navigation sequence walked by prev/next: EVERY in-scope image, in global
  // order (sequential) or one fixed seeded shuffle (random). Because it shuffles
  // the full manifest — whose contents never change as you box — the order is
  // identical on every render/navigation, so Left always returns to the image you
  // just saw. The path index `n` stays global.
  const filtered = useMemo(() => (manifest ? filterByCats(manifest, cats) : []), [manifest, cats]);
  const ordered = useMemo(
    () => (seed == null ? filtered : orderBySeed(filtered, seed)),
    [filtered, seed],
  );
  // Membership Set of still-unboxed images — drives Space's "jump to next unboxed".
  const boxSet = useMemo(() => (boxQueue ? new Set(boxQueue) : null), [boxQueue]);
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

  const { annotation, addBox, removeBox, replaceBoxes, setBoxed, undo, redo, canUndo, canRedo, loading } = useAnnotation(cat, name);

  // --- Model box proposals (suggest_boxes output) -> yellow approve/reject. ---
  const { boxes: suggBoxes, acceptBox, rejectBox } = useSuggestions(cat, name);
  // Index of the keyboard-selected proposal; clamped to the (shrinking) list.
  const [activeSugg, setActiveSugg] = useState(0);
  useEffect(() => {
    setActiveSugg(0);
  }, [cat, name]);
  const activeSuggId = suggBoxes.length
    ? suggBoxes[Math.min(activeSugg, suggBoxes.length - 1)]?.id ?? null
    : null;

  // Accept a proposal: add it as a real box (carrying any suggested label so an
  // accepted box can arrive already identified), mark the image boxed, and drop
  // the proposal from the sidecar.
  const acceptSuggestion = useCallback(
    (id: string) => {
      const s = suggBoxes.find((b) => b.id === id);
      if (!s) return;
      addBox({ cx: s.cx, cy: s.cy, w: s.w, h: s.h, label: s.label ?? null });
      if (!annotation?.boxed) setBoxed(true);
      acceptBox(id);
    },
    [suggBoxes, addBox, annotation?.boxed, setBoxed, acceptBox],
  );

  const acceptActiveSuggestion = useCallback(() => {
    if (activeSuggId) acceptSuggestion(activeSuggId);
  }, [activeSuggId, acceptSuggestion]);

  const rejectActiveSuggestion = useCallback(() => {
    if (activeSuggId) rejectBox(activeSuggId);
  }, [activeSuggId, rejectBox]);

  const acceptAllSuggestions = useCallback(() => {
    if (suggBoxes.length === 0) return;
    for (const s of suggBoxes) {
      addBox({ cx: s.cx, cy: s.cy, w: s.w, h: s.h, label: s.label ?? null });
      acceptBox(s.id);
    }
    setBoxed(true);
  }, [suggBoxes, addBox, acceptBox, setBoxed]);

  const cycleSuggestion = useCallback(() => {
    if (suggBoxes.length === 0) return;
    setActiveSugg((i) => (i + 1) % suggBoxes.length);
  }, [suggBoxes.length]);

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
  // Confirm the current image on the way out — `boxed` must always mean "the user
  // placed boxes here", so set it iff boxes actually exist (reverting a stale flag).
  const confirmCurrent = useCallback(() => {
    const hasBoxes = (annotation?.boxes.length ?? 0) > 0;
    if ((annotation?.boxed ?? false) !== hasBoxes) setBoxed(hasBoxes);
  }, [annotation?.boxes.length, annotation?.boxed, setBoxed]);

  // Right arrow: step one image forward in the fixed order (clamp at the end —
  // finishing the worklist is Space's job now). Still confirms the current image.
  const goNext = useCallback(() => {
    confirmCurrent();
    const idx = ordered.findIndex((e) => e.n === n);
    const next = idx >= 0 ? ordered[idx + 1] : undefined;
    if (next) router.push(withNav(`/box/${next.n}`, cats, seed));
  }, [confirmCurrent, ordered, n, router, cats, seed]);

  const goPrev = useCallback(() => {
    const idx = ordered.findIndex((e) => e.n === n);
    const prev = idx > 0 ? ordered[idx - 1] : undefined;
    if (!prev) return;
    router.push(withNav(`/box/${prev.n}`, cats, seed));
  }, [ordered, n, router, cats, seed]);

  // Space: jump to the next image that still NEEDS boxing, in the fixed order.
  // Scan starts at idx+1 so the just-confirmed current image is never re-shown.
  // Home when nothing's left. Without a loaded queue, fall back to plain next.
  const advance = useCallback(() => {
    confirmCurrent();
    const idx = ordered.findIndex((e) => e.n === n);
    let target: ManifestEntry | undefined;
    for (let i = idx + 1; i < ordered.length; i++) {
      if (!boxSet || boxSet.has(ordered[i].n)) {
        target = ordered[i];
        break;
      }
    }
    router.push(target ? withNav(`/box/${target.n}`, cats, seed) : "/");
  }, [confirmCurrent, ordered, n, boxSet, router, cats, seed]);

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

  // Clear the current image: wipe every placed box (reverting `boxed`, so the
  // flag keeps meaning "has user boxes") AND reject every pending yellow
  // suggestion still on screen. No-op when there's nothing of either to clear.
  const clearBoxes = useCallback(() => {
    const hasBoxes = (annotation?.boxes.length ?? 0) > 0;
    if (!hasBoxes && suggBoxes.length === 0) return;
    if (hasBoxes) replaceBoxes([]);
    if (annotation?.boxed) setBoxed(false);
    for (const s of suggBoxes) rejectBox(s.id);
  }, [annotation?.boxes.length, annotation?.boxed, replaceBoxes, setBoxed, suggBoxes, rejectBox]);

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
      // --- Box-proposal review (only when proposals exist). ------------------
      if (suggBoxes.length > 0) {
        if (e.key === "Tab") {
          e.preventDefault();
          cycleSuggestion();
          return;
        }
        if (e.key === "Enter" || e.key === "y" || e.key === "Y") {
          e.preventDefault();
          acceptActiveSuggestion();
          return;
        }
        if (e.key === "n" || e.key === "N") {
          e.preventDefault();
          rejectActiveSuggestion();
          return;
        }
        if (e.key === "a" || e.key === "A") {
          e.preventDefault();
          acceptAllSuggestions();
          return;
        }
      }

      if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        advance();
        return;
      }
      if (e.key === "ArrowRight") {
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
  }, [handleUndo, handleRedo, handleDelete, advance, goNext, goPrev, toggleRandom, clearBoxes, router, suggBoxes.length, cycleSuggestion, acceptActiveSuggestion, rejectActiveSuggestion, acceptAllSuggestions]);

  // --- Render guards --------------------------------------------------------
  // Render once the manifest is in — navigation/order don't need the box queue
  // (it only feeds Space, which tolerates a late-arriving Set).
  if (
    manifest == null ||
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

  // Box/suggestion actions: Accept-all (while proposals exist) + Clear (whenever
  // there are placed boxes OR suggestions to wipe). Clear rejects suggestions too.
  const hasBoxes = (annotation?.boxes.length ?? 0) > 0;
  const hasSuggestions = suggBoxes.length > 0;
  const boxActions: ReactNode[] = [];
  if (hasSuggestions) boxActions.push(<ActionButton key="accept-all" label="Accept all" shortcut="A" variant="suggest" title="Accept all suggestions (A)" onClick={acceptAllSuggestions} />);
  if (hasBoxes || hasSuggestions) boxActions.push(<ActionButton key="clear" label="Clear" shortcut="C" variant="danger" title="Clear all boxes and suggestions (C)" onClick={clearBoxes} />);

  const bottomBarSegments: Array<ReactNode | null> = [
    total > 0 ? (
      <NavCluster
        pos={pos}
        total={total}
        onPrev={goPrev}
        onNext={goNext}
        prevDisabled={pos <= 0}
        nextDisabled={pos < 0 || pos === total - 1}
        onAdvance={advance}
        coarse={coarse}
      />
    ) : null,
    <>
      <ActionButton key="undo" label="Undo" icon={<UndoIcon />} onClick={handleUndo} disabled={!canUndo} />
      <ActionButton key="redo" label="Redo" icon={<RedoIcon />} onClick={handleRedo} disabled={!canRedo} />
    </>,
    boxActions.length > 0 ? <>{boxActions}</> : null,
    <ActionButton
      key="delete"
      label="Delete"
      shortcut="⌘⌫"
      variant="danger"
      title="Delete this image (⌘/Ctrl + Backspace)"
      onClick={() => void handleDelete()}
    />,
  ];

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
        <BoxLayer
          boxes={annotation?.boxes ?? []}
          draft={draft}
          showDelete
          onDelete={handleRemoveBox}
          alwaysShowControls={coarse}
        />
        {suggBoxes.length > 0 && (
          <SuggestionLayer
            boxes={suggBoxes}
            activeId={activeSuggId}
            onAccept={acceptSuggestion}
            onReject={rejectBox}
            alwaysShowControls={coarse}
          />
        )}
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

      {/* Box-proposal review hint (only while yellow proposals remain). On touch
          the ✓/✕ controls are always shown on each proposal, so the keyboard
          hint is redundant — just show the count. */}
      {suggBoxes.length > 0 && (
        <div className="pointer-events-none fixed left-1/2 top-4 -translate-x-1/2 rounded-pill border border-suggest/40 bg-surface/85 px-3 py-1.5 text-xs text-suggest backdrop-blur-md">
          {suggBoxes.length} suggestion{suggBoxes.length === 1 ? "" : "s"}
          {!coarse && (
            <>
              {" "}·{" "}
              <span className="font-mono">Tab</span> cycle · <span className="font-mono">Y</span> accept ·{" "}
              <span className="font-mono">N</span> reject · <span className="font-mono">A</span> all
            </>
          )}
        </div>
      )}

      {/* Bottom-center: one unified bar — stepper, box/suggestion actions, and
          the whole-image Delete. On touch, NavCluster also shows a primary Next →
          that skips to the next unboxed image. */}
      <BottomBar segments={bottomBarSegments} />

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
