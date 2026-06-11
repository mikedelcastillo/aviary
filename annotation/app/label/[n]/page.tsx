"use client";

import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Stage } from "@/components/Stage";
import { Spotlight } from "@/components/Spotlight";
import { BoxLayer } from "@/components/BoxLayer";
import { PillBar } from "@/components/PillBar";
import { LoadingOverlay } from "@/components/LoadingOverlay";
import { useAnnotation } from "@/lib/use-annotation";
import {
  categoryById,
  filterByCats,
  orderBySeed,
  parseCats,
  parseSeed,
  withNav,
  type CatId,
  type ManifestEntry,
  type NormRect,
  type Pill,
  type StageHandle,
} from "@/lib/types";

type RosterData = Record<CatId, Pill[]>;

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

  // --- Fetch manifest / roster / queue once on mount. -----------------------
  const [manifest, setManifest] = useState<ManifestEntry[] | null>(null);
  const [rosterData, setRosterData] = useState<RosterData | null>(null);
  const [queue, setQueue] = useState<number[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [mRes, rRes, qRes] = await Promise.all([
          fetch("/api/manifest"),
          fetch("/api/roster"),
          fetch("/api/queue"),
        ]);
        const [m, r, q] = await Promise.all([
          mRes.json() as Promise<ManifestEntry[]>,
          rRes.json() as Promise<RosterData>,
          qRes.json() as Promise<number[]>,
        ]);
        if (cancelled) return;
        setManifest(m);
        setRosterData(r);
        setQueue(q);
      } catch {
        if (cancelled) return;
        setManifest([]);
        setRosterData({ day: [], ir: [], phone: [] });
        setQueue([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Selected-category subset, plus the work queue (boxed-but-unlabeled images)
  // as manifest entries in global order.
  const filtered = useMemo(() => (manifest ? filterByCats(manifest, cats) : []), [manifest, cats]);
  const queueEntries = useMemo(() => {
    const set = new Set(queue ?? []);
    return filtered.filter((e) => set.has(e.n));
  }, [filtered, queue]);

  // The navigation sequence (counter + prev/next). Both modes walk ONLY the work
  // queue (boxed-but-unlabeled images) so an un-boxed image is never presented
  // for labeling. Sequential keeps global order; random applies a stable seeded
  // shuffle. Already-labeled / un-boxed images are skipped entirely.
  const ordered = useMemo(
    () => (seed == null ? queueEntries : orderBySeed(queueEntries, seed)),
    [seed, queueEntries],
  );
  const total = ordered.length;
  const globalInRange = manifest != null && Number.isInteger(n) && n >= 0 && n < manifest.length;
  const baseEntry = globalInRange ? manifest![n] : null;
  const inCats = baseEntry != null && cats.includes(baseEntry.cat);
  const pos = baseEntry && inCats ? ordered.findIndex((e) => e.n === n) : -1;

  // Deep-link guard: if `n` isn't a valid stop in the nav sequence (un-boxed,
  // wrong category, or already labeled), snap to the first queued image at/after
  // it (else the first in the sequence). With the queue loaded but empty there's
  // nothing to label — go home rather than render an un-boxed image's seed boxes.
  useEffect(() => {
    if (manifest == null || queue == null) return; // still loading
    if (ordered.length === 0) {
      router.replace("/");
      return;
    }
    if (pos >= 0) return;
    const target = ordered.find((e) => e.n >= n) ?? ordered[0];
    router.replace(withNav(`/label/${target.n}`, cats, seed));
  }, [manifest, queue, ordered, pos, n, cats, seed, router]);

  const entry = baseEntry && inCats ? baseEntry : null;
  const cat = entry?.cat ?? null;
  const name = entry?.name ?? null;

  const { annotation, setLabel, removeBox, undo, redo, loading } = useAnnotation(cat, name);

  const pills: Pill[] = useMemo(
    () => (cat && rosterData ? rosterData[cat] : []),
    [cat, rosterData],
  );

  const boxes = useMemo(() => annotation?.boxes ?? [], [annotation]);
  const unlabeled = useMemo(() => boxes.filter((b) => b.label == null), [boxes]);
  const activeBox = unlabeled[0] ?? null;

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
    const prev = idx > 0 ? ordered[idx - 1] : undefined;
    if (prev) router.push(withNav(`/label/${prev.n}`, cats, seed));
  }, [ordered, n, router, cats, seed]);

  const goNext = useCallback(() => {
    const idx = ordered.findIndex((e) => e.n === n);
    const next = idx >= 0 ? ordered[idx + 1] : undefined;
    if (next) router.push(withNav(`/label/${next.n}`, cats, seed));
  }, [ordered, n, router, cats, seed]);

  const advance = useCallback(() => {
    if (seed == null) {
      // Sequential: the next queued image after this one (unchanged behavior).
      const next = (queue ?? []).find(
        (idx) => idx > n && (manifest == null || cats.includes(manifest[idx].cat)),
      );
      router.push(next != null ? withNav(`/label/${next}`, cats, seed) : "/");
      return;
    }
    // Random: the next image in the shuffled work queue (`ordered`). Wraps to
    // the start if `n` isn't itself queued, e.g. arrived here via prev/next.
    const idx = ordered.findIndex((e) => e.n === n);
    const next = idx >= 0 ? ordered[idx + 1] : ordered[0];
    router.push(next ? withNav(`/label/${next.n}`, cats, seed) : "/");
  }, [seed, queue, n, manifest, cats, ordered, router]);

  // --- Assign a label to the active box, then advance if image is done. -----
  const pick = useCallback(
    (pill: Pill) => {
      if (!activeBox) return;
      setLabel(activeBox.id, pill.label);
      // If this was the last unlabeled box in the image, move on. The label
      // effect handles re-framing the next box within the same image.
      if (unlabeled.length <= 1) advance();
    },
    [activeBox, setLabel, unlabeled.length, advance],
  );

  // --- Unbox: delete an accidental box that reached labeling. ----------------
  const unbox = useCallback(() => {
    if (!activeBox) return;
    removeBox(activeBox.id);
    // Removing the last unlabeled box finishes this image.
    if (unlabeled.length <= 1) advance();
  }, [activeBox, removeBox, unlabeled.length, advance]);

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
        if (e.shiftKey) redo();
        else undo();
        return;
      }
      if (mod && e.key.toLowerCase() === "y") {
        e.preventDefault();
        redo();
        return;
      }
      if (mod) return; // leave other shortcuts (copy/paste/etc.) alone

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
  }, [activeBox, pills, pick, unbox, advance, goPrev, goNext, undo, redo, router]);

  // --- Render. --------------------------------------------------------------
  const category = cat ? categoryById(cat) : undefined;
  const imageSrc = cat && name ? `/api/image/${cat}/${encodeURIComponent(name)}` : null;
  const hasUnlabeled = unlabeled.length > 0;

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
            : "No boxes to label here — →"}
        </span>
      </div>

      {/* Top-right counter. */}
      {total > 0 && (
        <div className="pointer-events-none fixed right-4 top-4 z-30 font-mono text-xs text-faint">
          {pos + 1} / {total}
        </div>
      )}

      {/* Centered hint when nothing to label on this image. */}
      {imageSrc && !hasUnlabeled && (
        <div className="pointer-events-none fixed inset-0 z-10 flex items-center justify-center">
          <span className="rounded-pill border border-border bg-surface/80 px-4 py-2 text-sm text-muted backdrop-blur-md">
            No boxes to label here — press → to continue
          </span>
        </div>
      )}

      <PillBar
        pills={pills}
        onPick={pick}
        activeLabel={activeBox?.label ?? null}
        onUnbox={activeBox ? unbox : undefined}
      />
    </main>
  );
}
