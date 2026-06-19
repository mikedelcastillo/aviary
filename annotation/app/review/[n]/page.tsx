"use client";

import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Stage } from "@/components/Stage";
import { Spotlight } from "@/components/Spotlight";
import { BoxLayer } from "@/components/BoxLayer";
import { DeleteToast } from "@/components/DeleteToast";
import { BottomBar } from "@/components/BottomBar";
import { ActionButton } from "@/components/ActionButton";
import { UndoIcon, RedoIcon } from "@/components/icons";
import { NavCluster } from "@/components/NavCluster";
import { LoadingOverlay } from "@/components/LoadingOverlay";
import { Spinner } from "@/components/Spinner";
import { useAnnotation } from "@/lib/use-annotation";
import { useImageDelete } from "@/lib/use-image-delete";
import { useCoarsePointer } from "@/hooks/useCoarsePointer";
import {
  categoryById,
  filterByCats,
  gridReviewHref,
  parseCats,
  withCats,
  UNKNOWN_LABEL,
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
  const coarse = useCoarsePointer();

  // --- Fetch manifest + review queue (images containing `label`). -----------
  const [manifest, setManifest] = useState<ManifestEntry[] | null>(null);
  const [queue, setQueue] = useState<number[] | null>(null);

  // Load (or refresh) the manifest + review queue together, returning both so a
  // post-delete handler can navigate against the freshly-renumbered indices.
  const loadManifestQueue = useCallback(async (): Promise<{
    manifest: ManifestEntry[];
    queue: number[];
  }> => {
    if (!label) {
      setManifest([]);
      setQueue([]);
      return { manifest: [], queue: [] };
    }
    const cparam = searchParams.get("cats");
    const qs = new URLSearchParams({ label });
    if (cparam) qs.set("cats", cparam);
    try {
      const [mRes, qRes] = await Promise.all([
        fetch("/api/manifest"),
        fetch(`/api/review-queue?${qs.toString()}`),
      ]);
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
  }, [label, searchParams]);

  useEffect(() => {
    void loadManifestQueue();
  }, [loadManifestQueue]);

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

  const { annotation, setLabel, removeBox, undo, redo, canUndo, canRedo, loading } = useAnnotation(cat, name);

  // Manual image delete (whole image -> trash tree) with an undo toast.
  const { pending: pendingDelete, remove: deleteCurrent, undo: undoDelete, dismiss: dismissDelete } =
    useImageDelete();

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

  // Relabel the active box as the catch-all "unknown" kind. Pointless when we're
  // already reviewing that label, so it's a no-op there. Like (un)label, the box
  // leaves this queue, so advance once it was the last match on this image.
  const canUnknown = label !== UNKNOWN_LABEL;
  const labelUnknown = useCallback(() => {
    if (!activeBox || !canUnknown) return;
    setLabel(activeBox.id, UNKNOWN_LABEL);
    if (matching.length <= 1) advance();
  }, [activeBox, canUnknown, setLabel, matching.length, advance]);

  // --- Delete the whole image (move to trash), then advance to the next image
  // in the review queue. Undo restores it and hops back. Navigation is by
  // identity against a freshly-refetched manifest (delete renumbers indices). ---
  const handleDelete = useCallback(async () => {
    if (!cat || !name || !queue || !manifest) return;
    const nextN = queue.find((idx) => idx > n);
    const nextId = nextN != null ? { cat: manifest[nextN].cat, name: manifest[nextN].name } : null;
    await deleteCurrent(cat, name, async () => {
      const fresh = await loadManifestQueue();
      const t = nextId
        ? fresh.manifest.find((e) => e.cat === nextId.cat && e.name === nextId.name)
        : undefined;
      router.push(t ? linkFor(t.n) : "/");
    });
  }, [cat, name, queue, manifest, n, deleteCurrent, loadManifestQueue, linkFor, router]);

  const handleDeleteUndo = useCallback(() => {
    void undoDelete(async (p) => {
      const fresh = await loadManifestQueue();
      const t = fresh.manifest.find((e) => e.cat === p.cat && e.name === p.name);
      if (t) router.push(linkFor(t.n));
    });
  }, [undoDelete, loadManifestQueue, linkFor, router]);

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
      if (mod && e.key === "Backspace") {
        e.preventDefault();
        void handleDelete();
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
      if (e.key.toLowerCase() === "u" && activeBox && canUnknown) {
        e.preventDefault();
        labelUnknown();
        return;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeBox, canUnknown, advance, goPrev, goNext, unbox, unlabel, labelUnknown, undo, redo, handleDelete, router]);

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

  // Box actions: Unlabel + (Unknown) are neutral (reversible relabels); Unbox is
  // destructive. All disabled when this image has no remaining match.
  const reviewActions: ReactNode[] = [
    <ActionButton key="unlabel" label="Unlabel" shortcut="L" onClick={unlabel} disabled={!hasMatch} />,
  ];
  if (canUnknown) {
    reviewActions.push(
      <ActionButton key="unknown" label="Unknown" shortcut="U" title={`Relabel as ${UNKNOWN_LABEL} (U)`} onClick={labelUnknown} disabled={!hasMatch} />,
    );
  }
  reviewActions.push(
    <ActionButton key="unbox" label="Unbox" shortcut="B" variant="danger" onClick={unbox} disabled={!hasMatch} />,
  );

  const bottomBarSegments: Array<ReactNode | null> = [
    total > 0 ? (
      <NavCluster
        pos={pos}
        total={total}
        onPrev={goPrev}
        onNext={goNext}
        prevDisabled={pos <= 0}
        nextDisabled={pos < 0 || pos >= total - 1}
        onAdvance={advance}
        coarse={coarse}
      />
    ) : null,
    <>
      <ActionButton key="undo" label="Undo" icon={<UndoIcon />} onClick={() => undo()} disabled={!canUndo} />
      <ActionButton key="redo" label="Redo" icon={<RedoIcon />} onClick={() => redo()} disabled={!canRedo} />
    </>,
    <>{reviewActions}</>,
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
        {/* Focus (active) ⇄ Grid toggle — same label + category scope. */}
        <div className="pointer-events-auto mt-0.5 flex w-fit items-center rounded-pill border border-border bg-surface p-0.5">
          <span className="rounded-pill bg-fg px-2.5 py-1 font-medium text-bg">Focus</span>
          <Link
            href={gridReviewHref(label, cats)}
            className="rounded-pill px-2.5 py-1 text-muted transition-colors hover:text-fg"
          >
            Grid
          </Link>
        </div>
        {category && <span className="text-faint">{category.title}</span>}
        {name && <span className="max-w-[40ch] truncate text-faint">{name}</span>}
        <span className="text-muted">
          {hasMatch
            ? `${matching.length} box${matching.length === 1 ? "" : "es"} here`
            : coarse
              ? "Cleared — tap Next →"
              : "Cleared — press → / space"}
        </span>
      </div>

      {/* Centered hint when this image has no more matches. */}
      {!hasMatch && (
        <div className="pointer-events-none fixed inset-0 z-10 flex items-center justify-center">
          <span className="rounded-pill border border-border bg-surface/80 px-4 py-2 text-sm text-muted backdrop-blur-md">
            {coarse ? "Nothing left to review here — tap Next →" : "Nothing left to review here — press → or space"}
          </span>
        </div>
      )}

      {/* Bottom action bar: stepper · Unlabel/Unknown/Unbox · Delete. */}
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
