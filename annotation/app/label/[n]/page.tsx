"use client";

import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Stage } from "@/components/Stage";
import { Spotlight } from "@/components/Spotlight";
import { BoxLayer } from "@/components/BoxLayer";
import { BottomBar } from "@/components/BottomBar";
import { PillGroup } from "@/components/PillGroup";
import { ActionButton } from "@/components/ActionButton";
import { UndoIcon, RedoIcon } from "@/components/icons";
import { NavCluster } from "@/components/NavCluster";
import { DeleteToast } from "@/components/DeleteToast";
import { LoadingOverlay } from "@/components/LoadingOverlay";
import { useAnnotation } from "@/lib/use-annotation";
import { useSuggestions } from "@/lib/use-suggestions";
import { useImageDelete } from "@/lib/use-image-delete";
import { useCoarsePointer } from "@/hooks/useCoarsePointer";
import {
  categoryById,
  filterByCats,
  newSeed,
  orderBySeed,
  parseCats,
  parseSeed,
  parseSuggest,
  serializeCats,
  withNav,
  type CatId,
  type ManifestEntry,
  type NormRect,
  type Pill,
  type StageHandle,
} from "@/lib/types";

type RosterData = Record<CatId, Pill[]>;

/** The in-memory "any suggestion" sentinel (matches page.tsx ALL_SUGGESTIONS). */
const ALL_SUGGESTIONS = "all";

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

export default function LabelPage() {
  const params = useParams<{ n: string }>();
  const n = Number(params.n);
  const router = useRouter();
  const searchParams = useSearchParams();
  const cats = useMemo(() => parseCats(searchParams.get("cats")), [searchParams]);
  const seed = useMemo(() => parseSeed(searchParams.get("random")), [searchParams]);
  // Suggestion filter carried from Home (and across every navigation). Empty set
  // = inactive (browse all in-scope frames as before).
  const suggest = useMemo(() => parseSuggest(searchParams.get("suggest")), [searchParams]);
  const suggestActive = suggest.size > 0;
  const coarse = useCoarsePointer();

  // --- Fetch manifest / roster / queue once on mount. -----------------------
  const [manifest, setManifest] = useState<ManifestEntry[] | null>(null);
  const [rosterData, setRosterData] = useState<RosterData | null>(null);
  const [queue, setQueue] = useState<number[] | null>(null);
  // The suggestion index (label -> global frame indices) from /api/home, fetched
  // only when the suggestion filter is active. `null` until it lands so navigation
  // can hold off on restricting (rather than wrongly skipping every frame).
  const [suggestionLabelIndex, setSuggestionLabelIndex] = useState<Record<
    string,
    number[]
  > | null>(null);

  // Load (or refresh) the manifest + label queue together, returning both so a
  // post-delete handler can navigate against the freshly-renumbered indices.
  // The roster is static for the session, so it's fetched only on mount below.
  const loadManifestQueue = useCallback(async (): Promise<{
    manifest: ManifestEntry[];
    queue: number[];
  }> => {
    try {
      const [mRes, qRes] = await Promise.all([fetch("/api/manifest"), fetch("/api/queue")]);
      const [m, q] = await Promise.all([
        mRes.json() as Promise<ManifestEntry[]>,
        qRes.json() as Promise<number[]>,
      ]);
      setManifest(m);
      setQueue(q);
      return { manifest: m, queue: q };
    } catch {
      setManifest([]);
      setQueue([]);
      return { manifest: [], queue: [] };
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const rRes = await fetch("/api/roster");
        const r = (await rRes.json()) as RosterData;
        if (!cancelled) setRosterData(r);
      } catch {
        if (!cancelled) setRosterData({ day: [], ir: [], phone: [] });
      }
    })();
    void loadManifestQueue();
    return () => {
      cancelled = true;
    };
  }, [loadManifestQueue]);

  // When the suggestion filter is active, pull the per-label frame index from
  // /api/home once so navigation can constrain to matching frames. Reuses the
  // same payload the home page caches; guards for an absent field on old servers.
  const catsKey = serializeCats(cats);
  useEffect(() => {
    if (!suggestActive) {
      setSuggestionLabelIndex(null);
      return;
    }
    let cancelled = false;
    const qs = catsKey ? `?cats=${catsKey}` : "";
    fetch(`/api/home${qs}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(String(res.status)))))
      .then((data: { suggestionLabelIndex?: Record<string, number[]> }) => {
        if (!cancelled) setSuggestionLabelIndex(data.suggestionLabelIndex ?? {});
      })
      .catch(() => {
        // On failure, treat as "no restriction" rather than stranding the user.
        if (!cancelled) setSuggestionLabelIndex({});
      });
    return () => {
      cancelled = true;
    };
  }, [suggestActive, catsKey]);

  // Resolve the suggestion selection to the concrete set of permitted frame
  // indices. `null` = filter active but index still loading (don't restrict yet);
  // a non-null Set = the frames whose suggestions include a selected label. With
  // the all-sentinel it's the union of EVERY label's frames (any suggestion).
  const suggestSet = useMemo<Set<number> | null>(() => {
    if (!suggestActive) return null; // off — no restriction
    if (suggestionLabelIndex === null) return null; // loading
    const out = new Set<number>();
    if (suggest.has(ALL_SUGGESTIONS)) {
      for (const arr of Object.values(suggestionLabelIndex)) {
        for (const m of arr) out.add(m);
      }
    } else {
      for (const label of suggest) {
        for (const m of suggestionLabelIndex[label] ?? []) out.add(m);
      }
    }
    return out;
  }, [suggestActive, suggest, suggestionLabelIndex]);

  // Selected-category subset. `queueSet` (boxed-but-unlabeled image indices) is a
  // membership Set only — it drives Space's "jump to next unlabeled", NOT the
  // navigation order.
  const filtered = useMemo(() => (manifest ? filterByCats(manifest, cats) : []), [manifest, cats]);
  const queueSet = useMemo(() => new Set(queue ?? []), [queue]);

  // The navigation sequence walked by prev/next: EVERY in-scope image, in global
  // order (sequential) or one fixed seeded shuffle (random). Shuffling the full
  // manifest — whose contents never change as you label — keeps the order stable
  // across renders/navigation, so Left always returns to the image you just saw.
  const ordered = useMemo(
    () => (seed == null ? filtered : orderBySeed(filtered, seed)),
    [seed, filtered],
  );
  const total = ordered.length;
  const globalInRange = manifest != null && Number.isInteger(n) && n >= 0 && n < manifest.length;
  const baseEntry = globalInRange ? manifest![n] : null;
  const inCats = baseEntry != null && cats.includes(baseEntry.cat);
  const pos = baseEntry && inCats ? ordered.findIndex((e) => e.n === n) : -1;

  // Deep-link guard. Nothing left to label (or no in-scope images) → home rather
  // than strand the user. Otherwise any in-scope image is a valid stop (arrows
  // browse the whole set), so only snap when `n` is out of range / wrong category
  // — to the nearest in-scope image at/after it (else the first in the sequence).
  useEffect(() => {
    if (manifest == null || queue == null) return; // still loading
    if (suggestActive && suggestSet == null) return; // suggestion index still loading
    if (queue.length === 0 || ordered.length === 0) {
      router.replace("/");
      return;
    }
    const validHere = pos >= 0 && (!suggestSet || suggestSet.has(n));
    if (validHere) return;
    const target = suggestSet
      ? ordered.find((e) => e.n >= n && suggestSet.has(e.n)) ?? ordered.find((e) => suggestSet.has(e.n))
      : ordered.find((e) => e.n >= n) ?? ordered[0];
    if (!target) {
      router.replace("/");
      return;
    }
    router.replace(withNav(`/label/${target.n}`, cats, seed, suggest));
  }, [manifest, queue, ordered, pos, n, cats, seed, suggest, suggestActive, suggestSet, router]);

  const entry = baseEntry && inCats ? baseEntry : null;
  const cat = entry?.cat ?? null;
  const name = entry?.name ?? null;

  const { annotation, setLabel, removeBox, replaceBoxes, undo, redo, canUndo, canRedo, loading } = useAnnotation(cat, name);

  // Model label suggestions (suggest_labels output) — pre-highlight the pill.
  const { labelFor, dismissLabel } = useSuggestions(cat, name);

  // Manual image delete (whole image -> trash tree) with an undo toast.
  const { pending: pendingDelete, remove: deleteCurrent, undo: undoDelete, dismiss: dismissDelete } =
    useImageDelete();

  const pills: Pill[] = useMemo(
    () => (cat && rosterData ? rosterData[cat] : []),
    [cat, rosterData],
  );

  const boxes = useMemo(() => annotation?.boxes ?? [], [annotation]);
  const unlabeled = useMemo(() => boxes.filter((b) => b.label == null), [boxes]);
  const activeBox = unlabeled[0] ?? null;

  // Model-suggested label for the active box, if one exists AND it's a pill the
  // current category offers (don't surface a label the user can't pick here).
  const suggestion = activeBox ? labelFor(activeBox.id) : undefined;
  const suggestedLabel =
    suggestion && pills.some((p) => p.label === suggestion.label) ? suggestion.label : null;

  const activeRect: NormRect | null = useMemo(() => {
    if (!activeBox) return null;
    return {
      x: activeBox.cx - activeBox.w / 2,
      y: activeBox.cy - activeBox.h / 2,
      w: activeBox.w,
      h: activeBox.h,
    };
  }, [activeBox]);

  // --- Stage handle + ready flag. -------------------------------------------
  const stageRef = useRef<StageHandle | null>(null);
  const [ready, setReady] = useState(false);
  const onReady = useCallback((h: StageHandle) => {
    stageRef.current = h;
    setReady(true);
  }, []);

  // Reset readiness whenever the image source changes.
  useEffect(() => {
    setReady(false);
  }, [cat, name]);

  // --- Frame the active box when it changes (and once the stage is ready). ---
  useEffect(() => {
    if (!ready || !activeRect) return;
    stageRef.current?.focusRect(activeRect, 0.6);
    // Re-run when the active box id changes or the stage becomes ready.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, activeBox?.id]);

  // --- Navigation helpers (scoped to selected categories). ------------------
  const goPrev = useCallback(() => {
    const idx = ordered.findIndex((e) => e.n === n);
    let prev: ManifestEntry | undefined;
    for (let i = idx - 1; i >= 0; i--) {
      if (!suggestSet || suggestSet.has(ordered[i].n)) {
        prev = ordered[i];
        break;
      }
    }
    if (prev) router.push(withNav(`/label/${prev.n}`, cats, seed, suggest));
  }, [ordered, n, suggestSet, router, cats, seed, suggest]);

  const goNext = useCallback(() => {
    const idx = ordered.findIndex((e) => e.n === n);
    let next: ManifestEntry | undefined;
    for (let i = idx + 1; i < ordered.length; i++) {
      if (!suggestSet || suggestSet.has(ordered[i].n)) {
        next = ordered[i];
        break;
      }
    }
    if (next) router.push(withNav(`/label/${next.n}`, cats, seed, suggest));
  }, [ordered, n, suggestSet, router, cats, seed, suggest]);

  // Space / image-completion: jump to the next image that still NEEDS labeling,
  // in the fixed order. Scan starts at idx+1 so the just-completed current image
  // is never re-shown. Home when nothing's left to label.
  const advance = useCallback(() => {
    const idx = ordered.findIndex((e) => e.n === n);
    let target: ManifestEntry | undefined;
    for (let i = idx + 1; i < ordered.length; i++) {
      const m = ordered[i].n;
      if (queueSet.has(m) && (!suggestSet || suggestSet.has(m))) {
        target = ordered[i];
        break;
      }
    }
    router.push(target ? withNav(`/label/${target.n}`, cats, seed, suggest) : "/");
  }, [ordered, n, queueSet, suggestSet, cats, seed, suggest, router]);

  // --- Assign a label to the active box, then advance if image is done. -----
  const pick = useCallback(
    (pill: Pill) => {
      if (!activeBox) return;
      setLabel(activeBox.id, pill.label);
      // Either accepting or overriding a suggestion resolves it — drop it so it
      // doesn't linger if the box is revisited.
      dismissLabel(activeBox.id);
      // If this was the last unlabeled box in the image, move on. The label
      // effect handles re-framing the next box within the same image.
      if (unlabeled.length <= 1) advance();
    },
    [activeBox, setLabel, dismissLabel, unlabeled.length, advance],
  );

  // Accept the model's suggested identity for the active box with one key.
  const acceptSuggestion = useCallback(() => {
    if (!activeBox || !suggestedLabel) return;
    const pill = pills.find((p) => p.label === suggestedLabel);
    if (pill) pick(pill);
  }, [activeBox, suggestedLabel, pills, pick]);

  // --- Unbox: delete an accidental box that reached labeling. ----------------
  const unbox = useCallback(() => {
    if (!activeBox) return;
    removeBox(activeBox.id);
    // Removing the last unlabeled box finishes this image.
    if (unlabeled.length <= 1) advance();
  }, [activeBox, removeBox, unlabeled.length, advance]);

  // --- Clear: wipe every box on the image, then move on (nothing left here). --
  const clearBoxes = useCallback(() => {
    if (boxes.length === 0) return;
    replaceBoxes([]);
    advance();
  }, [boxes.length, replaceBoxes, advance]);

  // --- Delete the whole image (move to trash), then advance to the next queued
  // image. Undo restores it and hops back. Navigation is by identity against a
  // freshly-refetched manifest, since deleting renumbers the global indices. ----
  const handleDelete = useCallback(async () => {
    if (!cat || !name) return;
    const idx = ordered.findIndex((e) => e.n === n);
    const nextEntry = idx >= 0 ? ordered[idx + 1] : undefined;
    const nextId = nextEntry ? { cat: nextEntry.cat, name: nextEntry.name } : null;
    await deleteCurrent(cat, name, async () => {
      const { manifest: fresh } = await loadManifestQueue();
      const t = nextId ? fresh.find((e) => e.cat === nextId.cat && e.name === nextId.name) : undefined;
      router.push(t ? withNav(`/label/${t.n}`, cats, seed, suggest) : "/");
    });
  }, [cat, name, ordered, n, deleteCurrent, loadManifestQueue, router, cats, seed, suggest]);

  const handleDeleteUndo = useCallback(() => {
    void undoDelete(async (p) => {
      const { manifest: fresh } = await loadManifestQueue();
      const t = fresh.find((e) => e.cat === p.cat && e.name === p.name);
      if (t) router.push(withNav(`/label/${t.n}`, cats, seed, suggest));
    });
  }, [undoDelete, loadManifestQueue, router, cats, seed, suggest]);

  // --- Undo/redo: history is session-global, so a mistake made before an
  // auto-advance is undone by hopping back to the image it happened on. ------
  const navToHistory = useCallback(
    (key: { cat: CatId; name: string }) => {
      const target = manifest?.find((e) => e.cat === key.cat && e.name === key.name);
      if (target) router.push(withNav(`/label/${target.n}`, cats, seed, suggest));
    },
    [manifest, router, cats, seed, suggest],
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
    router.replace(withNav(`/label/${n}`, cats, seed != null ? null : newSeed(), suggest));
  }, [router, n, cats, seed, suggest]);

  // --- Keyboard. ------------------------------------------------------------
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target)) return;

      if (e.key === "Escape") {
        e.preventDefault();
        router.push("/");
        return;
      }

      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) handleRedo();
        else handleUndo();
        return;
      }
      if (mod && e.key.toLowerCase() === "y") {
        e.preventDefault();
        handleRedo();
        return;
      }
      if (mod && e.key === "Backspace") {
        e.preventDefault();
        void handleDelete();
        return;
      }
      if (mod) return; // leave other shortcuts (copy/paste/etc.) alone

      // Accept the model's suggested label: Enter always, Y when no pill claims it.
      if (suggestedLabel && activeBox) {
        if (e.key === "Enter") {
          e.preventDefault();
          acceptSuggestion();
          return;
        }
        if (
          (e.key === "y" || e.key === "Y") &&
          !pills.some((p) => p.shortcut.toLowerCase() === "y")
        ) {
          e.preventDefault();
          acceptSuggestion();
          return;
        }
      }

      // Space skips to the next image needing labels (home after the last).
      if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        advance();
        return;
      }

      if (e.key === "ArrowLeft") {
        e.preventDefault();
        goPrev();
        return;
      }
      if (e.key === "ArrowRight") {
        e.preventDefault();
        goNext();
        return;
      }

      // [R] flips random/sequential order — unless a roster pill claims "r".
      if (
        (e.key === "r" || e.key === "R") &&
        !pills.some((p) => p.shortcut.toLowerCase() === "r")
      ) {
        e.preventDefault();
        toggleRandom();
        return;
      }

      // [C] clears every box on the image — unless a roster pill claims "c".
      if (
        (e.key === "c" || e.key === "C") &&
        boxes.length > 0 &&
        !pills.some((p) => p.shortcut.toLowerCase() === "c")
      ) {
        e.preventDefault();
        clearBoxes();
        return;
      }

      // [B] unboxes the active box (takes precedence over label shortcuts).
      if (e.key.toLowerCase() === "b" && activeBox) {
        e.preventDefault();
        unbox();
        return;
      }

      if (activeBox) {
        const key = e.key.toLowerCase();
        const match = pills.find((p) => p.shortcut.toLowerCase() === key);
        if (match) {
          e.preventDefault();
          pick(match);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeBox, boxes.length, pills, pick, unbox, clearBoxes, advance, goPrev, goNext, handleUndo, handleRedo, handleDelete, toggleRandom, router, suggestedLabel, acceptSuggestion]);

  // --- Render. --------------------------------------------------------------
  const category = cat ? categoryById(cat) : undefined;
  const imageSrc = cat && name ? `/api/image/${cat}/${encodeURIComponent(name)}` : null;
  const hasUnlabeled = unlabeled.length > 0;

  // Box-level destructive actions (omit a button when its target is absent).
  const boxActions: ReactNode[] = [];
  if (activeBox) boxActions.push(<ActionButton key="unbox" label="Unbox" shortcut="B" variant="danger" onClick={unbox} />);
  if (boxes.length > 0) boxActions.push(<ActionButton key="clear" label="Clear" shortcut="C" variant="danger" onClick={clearBoxes} />);

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
    pills.length > 0 ? (
      <PillGroup pills={pills} onPick={pick} activeLabel={activeBox?.label ?? null} suggestedLabel={suggestedLabel} />
    ) : null,
    boxActions.length > 0 ? <>{boxActions}</> : null,
    name ? (
      <ActionButton
        label="Delete"
        shortcut="⌘⌫"
        variant="danger"
        title="Delete this image (⌘/Ctrl + Backspace)"
        onClick={() => void handleDelete()}
      />
    ) : null,
  ];

  return (
    <main className="fixed inset-0 bg-bg text-fg">
      {imageSrc && (
        <Stage ref={stageRef} src={imageSrc} drawingEnabled={false} onReady={onReady}>
          <Spotlight rect={hasUnlabeled ? activeRect : null} />
          <BoxLayer
            boxes={boxes}
            activeId={activeBox?.id ?? null}
            showDelete
            deletableFor={(b) => b.label != null}
            onDelete={(id) => setLabel(id, null)}
            alwaysShowControls={coarse}
          />
        </Stage>
      )}

      <LoadingOverlay show={!!imageSrc && (!ready || loading)} label="Loading image…" />

      {/* Top-left info cluster. */}
      <div className="pointer-events-none fixed left-4 top-4 z-30 flex flex-col gap-1 text-xs text-muted">
        <Link
          href="/"
          className="pointer-events-auto w-fit text-muted transition-colors hover:text-fg"
        >
          ← Home
        </Link>
        {category && <span className="text-fg/80">{category.title}</span>}
        {name && <span className="max-w-[40ch] truncate text-faint">{name}</span>}
        <span className="text-muted">
          {hasUnlabeled
            ? `${unlabeled.length} box${unlabeled.length === 1 ? "" : "es"} left`
            : coarse
              ? "No boxes to label here"
              : "No boxes to label here — →"}
        </span>
        {suggestedLabel && (
          <span className="text-suggest">
            suggested: {suggestedLabel}
            {!coarse && <span className="font-mono"> ⏎</span>}
          </span>
        )}
        <button
          type="button"
          onClick={toggleRandom}
          title="Toggle random / sequential order (R)"
          className="pointer-events-auto w-fit text-faint transition-colors hover:text-fg"
        >
          {seed != null ? "random order" : "sequential order"}
        </button>
      </div>

      {/* Centered hint when nothing to label on this image. */}
      {imageSrc && !hasUnlabeled && (
        <div className="pointer-events-none fixed inset-0 z-10 flex items-center justify-center">
          <span className="rounded-pill border border-border bg-surface/80 px-4 py-2 text-sm text-muted backdrop-blur-md">
            {coarse ? "No boxes to label here — tap Next →" : "No boxes to label here — press → to continue"}
          </span>
        </div>
      )}

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
