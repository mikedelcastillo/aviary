"use client";

import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Stage } from "@/components/Stage";
import { Spotlight } from "@/components/Spotlight";
import { BoxLayer } from "@/components/BoxLayer";
import { LoadingOverlay } from "@/components/LoadingOverlay";
import { Spinner } from "@/components/Spinner";
import { useAnnotation } from "@/lib/use-annotation";
import {
  categoryById,
  filterByCats,
  parseCats,
  withCats,
  type ManifestEntry,
  type NormRect,
  type StageHandle,
} from "@/lib/types";

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

export default function ReviewPage() {
  const params = useParams<{ n: string }>();
  const n = Number(params.n);
  const router = useRouter();
  const searchParams = useSearchParams();
  const label = searchParams.get("label") ?? "";
  const cats = useMemo(() => parseCats(searchParams.get("cats")), [searchParams]);

  // --- Fetch manifest + review queue (images containing `label`). -----------
  const [manifest, setManifest] = useState<ManifestEntry[] | null>(null);
  const [queue, setQueue] = useState<number[] | null>(null);

  useEffect(() => {
    if (!label) {
      setManifest([]);
      setQueue([]);
      return;
    }
    let cancelled = false;
    const cparam = searchParams.get("cats");
    const qs = new URLSearchParams({ label });
    if (cparam) qs.set("cats", cparam);
    (async () => {
      try {
        const [mRes, qRes] = await Promise.all([
          fetch("/api/manifest"),
          fetch(`/api/review-queue?${qs.toString()}`),
        ]);
        const [m, q] = await Promise.all([
          mRes.json() as Promise<ManifestEntry[]>,
          qRes.json() as Promise<number[]>,
        ]);
        if (cancelled) return;
        setManifest(m);
        setQueue(q);
      } catch {
        if (cancelled) return;
        setManifest([]);
        setQueue([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [label, searchParams]);

  // Position within the review queue.
  const reviewSet = useMemo(() => new Set(queue ?? []), [queue]);
  const total = queue?.length ?? 0;
  const pos = queue ? queue.indexOf(n) : -1;
  const inQueue = reviewSet.has(n);
  const globalInRange = manifest != null && Number.isInteger(n) && n >= 0 && n < manifest.length;
  const baseEntry = globalInRange ? manifest![n] : null;

  // Deep-link guard: snap to the first reviewable image at/after `n`.
  useEffect(() => {
    if (manifest == null || queue == null || queue.length === 0) return;
    if (inQueue) return;
    const target = queue.find((idx) => idx >= n) ?? queue[0];
    const href = withCats(`/review/${target}`, cats);
    const sep = href.includes("?") ? "&" : "?";
    router.replace(`${href}${sep}label=${encodeURIComponent(label)}`);
  }, [manifest, queue, inQueue, n, cats, label, router]);

  const entry = baseEntry && inQueue ? baseEntry : null;
  const cat = entry?.cat ?? null;
  const name = entry?.name ?? null;

  const { annotation, setLabel, removeBox, undo, redo, loading } = useAnnotation(cat, name);

  const boxes = useMemo(() => annotation?.boxes ?? [], [annotation]);
  const matching = useMemo(() => boxes.filter((b) => b.label === label), [boxes, label]);
  const activeBox = matching[0] ?? null;

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
  useEffect(() => {
    setReady(false);
  }, [cat, name]);

  // Frame the active box when it changes.
  useEffect(() => {
    if (!ready || !activeRect) return;
    stageRef.current?.focusRect(activeRect, 0.6);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, activeBox?.id]);

  // --- Navigation (scoped to the review queue). -----------------------------
  const linkFor = useCallback(
    (target: number) => {
      const href = withCats(`/review/${target}`, cats);
      const sep = href.includes("?") ? "&" : "?";
      return `${href}${sep}label=${encodeURIComponent(label)}`;
    },
    [cats, label],
  );

  const goPrev = useCallback(() => {
    if (!queue || pos <= 0) return;
    router.push(linkFor(queue[pos - 1]));
  }, [queue, pos, router, linkFor]);

  const goNext = useCallback(() => {
    if (!queue || pos < 0 || pos >= queue.length - 1) return;
    router.push(linkFor(queue[pos + 1]));
  }, [queue, pos, router, linkFor]);

  const advance = useCallback(() => {
    if (!queue) return;
    const next = queue.find((idx) => idx > n);
    if (next != null) router.push(linkFor(next));
    else router.push("/");
  }, [queue, n, router, linkFor]);

  // --- Actions: only unbox (B) and unlabel (L). -----------------------------
  const unbox = useCallback(() => {
    if (!activeBox) return;
    removeBox(activeBox.id);
    if (matching.length <= 1) advance(); // handled the last match in this image
  }, [activeBox, removeBox, matching.length, advance]);

  const unlabel = useCallback(() => {
    if (!activeBox) return;
    setLabel(activeBox.id, null);
    if (matching.length <= 1) advance();
  }, [activeBox, setLabel, matching.length, advance]);

  // --- Keyboard. ------------------------------------------------------------
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target)) return;

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
      if (mod) return;

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
      if (e.key.toLowerCase() === "b" && activeBox) {
        e.preventDefault();
        unbox();
        return;
      }
      if (e.key.toLowerCase() === "l" && activeBox) {
        e.preventDefault();
        unlabel();
        return;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeBox, advance, goPrev, goNext, unbox, unlabel, undo, redo]);

  // --- Render guards. -------------------------------------------------------
  if (!label) {
    return (
      <main className="fixed inset-0 bg-bg flex flex-col items-center justify-center gap-3">
        <span className="text-sm text-muted">No label to review.</span>
        <Link href="/" className="text-xs text-faint hover:text-fg transition-colors">
          ← Home
        </Link>
      </main>
    );
  }

  if (manifest == null || queue == null || (!inQueue && total > 0)) {
    // Loading, or briefly mid-redirect to the first reviewable image.
    return (
      <main className="fixed inset-0 flex items-center justify-center bg-bg">
        <Spinner size={22} className="text-muted" />
      </main>
    );
  }

  if (total === 0 || !entry || !cat || !name) {
    return (
      <main className="fixed inset-0 bg-bg flex flex-col items-center justify-center gap-3">
        <span className="text-sm text-muted">
          No images labeled <span className="font-mono text-fg">{label}</span> in the selected categories.
        </span>
        <Link href="/" className="text-xs text-faint hover:text-fg transition-colors">
          ← Home
        </Link>
      </main>
    );
  }

  const imageSrc = `/api/image/${cat}/${encodeURIComponent(name)}`;
  const category = categoryById(cat);
  const hasMatch = matching.length > 0;

  return (
    <main className="fixed inset-0 bg-bg text-fg">
      <Stage ref={stageRef} src={imageSrc} drawingEnabled={false} onReady={onReady}>
        <Spotlight rect={hasMatch ? activeRect : null} />
        <BoxLayer boxes={boxes} activeId={activeBox?.id ?? null} />
      </Stage>

      <LoadingOverlay show={!ready || loading} label="Loading image…" />

      {/* Top-left info cluster. */}
      <div className="pointer-events-none fixed left-4 top-4 z-30 flex flex-col gap-1 text-xs text-muted">
        <Link
          href="/"
          className="pointer-events-auto w-fit text-muted transition-colors hover:text-fg"
        >
          ← Home
        </Link>
        <span className="text-fg/80">
          Reviewing <span className="font-mono text-box">{label}</span>
        </span>
        {category && <span className="text-faint">{category.title}</span>}
        {name && <span className="max-w-[40ch] truncate text-faint">{name}</span>}
        <span className="text-muted">
          {hasMatch
            ? `${matching.length} box${matching.length === 1 ? "" : "es"} here`
            : "Cleared — press → / space"}
        </span>
      </div>

      {/* Top-right counter. */}
      <div className="pointer-events-none fixed right-4 top-4 z-30 font-mono text-xs text-faint">
        {pos + 1} / {total}
      </div>

      {/* Centered hint when this image has no more matches. */}
      {!hasMatch && (
        <div className="pointer-events-none fixed inset-0 z-10 flex items-center justify-center">
          <span className="rounded-pill border border-border bg-surface/80 px-4 py-2 text-sm text-muted backdrop-blur-md">
            Nothing left to review here — press → or space
          </span>
        </div>
      )}

      {/* Bottom action bar: Unlabel [L] + Unbox [B]. */}
      <div className="pointer-events-none fixed inset-x-0 bottom-0 z-20 flex justify-center px-4 pb-6">
        <div className="pointer-events-auto flex items-center gap-2 rounded-2xl border border-border bg-surface/85 px-3 py-3 backdrop-blur-md">
          <button
            type="button"
            onClick={unlabel}
            disabled={!hasMatch}
            className="flex cursor-pointer items-center gap-2 rounded-pill border border-border bg-surface-2 px-3 py-1.5 text-sm text-muted transition-colors hover:border-border-strong hover:text-fg disabled:pointer-events-none disabled:opacity-40"
          >
            <kbd className="grid h-5 min-w-5 place-items-center rounded border border-border-strong px-1 font-mono text-[11px] uppercase text-faint">
              L
            </kbd>
            <span className="font-medium">Unlabel</span>
          </button>
          <button
            type="button"
            onClick={unbox}
            disabled={!hasMatch}
            className="flex cursor-pointer items-center gap-2 rounded-pill border border-danger/40 bg-danger/10 px-3 py-1.5 text-sm text-danger transition-colors hover:border-danger/70 disabled:pointer-events-none disabled:opacity-40"
          >
            <kbd className="grid h-5 min-w-5 place-items-center rounded border border-danger/50 px-1 font-mono text-[11px] uppercase text-danger">
              B
            </kbd>
            <span className="font-medium">Unbox</span>
          </button>
        </div>
      </div>
    </main>
  );
}
