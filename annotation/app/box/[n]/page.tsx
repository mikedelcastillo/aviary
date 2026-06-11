"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Stage } from "@/components/Stage";
import { BoxLayer } from "@/components/BoxLayer";
import { useAnnotation } from "@/lib/use-annotation";
import { LoadingOverlay } from "@/components/LoadingOverlay";
import { Spinner } from "@/components/Spinner";
import { categoryById, type CatId, type ManifestEntry, type NormRect, type Pt, type StageHandle } from "@/lib/types";

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

  const [manifest, setManifest] = useState<ManifestEntry[] | null>(null);
  const [draft, setDraft] = useState<NormRect | null>(null);

  const stageRef = useRef<StageHandle>(null);
  const startRef = useRef<Pt | null>(null);

  // Fetch the global manifest once.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/manifest");
        const data = (await res.json()) as ManifestEntry[];
        if (!cancelled) setManifest(data);
      } catch {
        if (!cancelled) setManifest([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const total = manifest?.length ?? 0;
  const inRange = manifest != null && Number.isInteger(n) && n >= 0 && n < total;
  const entry = inRange ? manifest![n] : null;

  const cat: CatId | null = entry?.cat ?? null;
  const name: string | null = entry?.name ?? null;

  const { annotation, addBox, removeBox, setBoxed, undo, redo, loading } = useAnnotation(cat, name);

  // --- Image readiness (drives the loading overlay). ------------------------
  const [ready, setReady] = useState(false);
  const onReady = useCallback(() => setReady(true), []);
  useEffect(() => {
    setReady(false);
  }, [cat, name]);

  // --- Navigation -----------------------------------------------------------
  const goNext = useCallback(() => {
    if (n + 1 >= total) return;
    setBoxed(true);
    router.push(`/box/${n + 1}`);
  }, [n, total, setBoxed, router]);

  const goPrev = useCallback(() => {
    if (n - 1 < 0) return;
    router.push(`/box/${n - 1}`);
  }, [n, router]);

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
      }
    },
    [addBox],
  );

  // --- Keyboard -------------------------------------------------------------
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || target?.isContentEditable) return;

      const mod = e.metaKey || e.ctrlKey;
      if (mod && (e.key === "z" || e.key === "Z")) {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
        return;
      }
      if (mod && (e.key === "y" || e.key === "Y")) {
        e.preventDefault();
        redo();
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
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [undo, redo, goNext, goPrev]);

  // --- Render guards --------------------------------------------------------
  if (manifest == null) {
    return (
      <main className="fixed inset-0 flex items-center justify-center bg-bg">
        <Spinner size={22} className="text-muted" />
      </main>
    );
  }

  if (!inRange || !entry || !cat || !name) {
    return (
      <main className="fixed inset-0 bg-bg flex flex-col items-center justify-center gap-3">
        <span className="text-sm text-muted">Image {Number.isInteger(n) ? n + 1 : "?"} is out of range.</span>
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
        <BoxLayer boxes={annotation?.boxes ?? []} draft={draft} showDelete onDelete={removeBox} />
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
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2">
        <div className="flex items-center gap-1 rounded-full border border-border bg-surface/85 px-2 py-1.5 backdrop-blur">
          <button
            type="button"
            onClick={goPrev}
            disabled={n === 0}
            aria-label="Previous image"
            className="flex h-7 w-7 items-center justify-center rounded-full text-muted transition-colors hover:bg-elevated hover:text-fg disabled:pointer-events-none disabled:opacity-30"
          >
            ‹
          </button>
          <span className="px-2 font-mono text-xs tabular-nums text-fg">
            {n + 1} / {total}
          </span>
          <button
            type="button"
            onClick={goNext}
            disabled={n === total - 1}
            aria-label="Next image (confirm)"
            className="flex h-7 w-7 items-center justify-center rounded-full text-muted transition-colors hover:bg-elevated hover:text-fg disabled:pointer-events-none disabled:opacity-30"
          >
            ›
          </button>
        </div>
      </div>
    </main>
  );
}
