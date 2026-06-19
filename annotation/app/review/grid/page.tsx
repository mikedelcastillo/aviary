"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { CropCell } from "@/components/CropCell";
import { DeleteToast } from "@/components/DeleteToast";
import { Spinner } from "@/components/Spinner";
import { BottomBar } from "@/components/BottomBar";
import { ActionButton } from "@/components/ActionButton";
import { useImageBatchDelete } from "@/lib/use-image-batch-delete";
import { useSaveStore } from "@/lib/save-store";
import {
  parseCats,
  reviewHref,
  UNKNOWN_LABEL,
  withCats,
  type Box,
  type CatId,
  type ReviewBox,
} from "@/lib/types";

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

const cellKey = (c: ReviewBox) => `${c.cat}/${c.name}/${c.box.id}`;
const imageKey = (c: { cat: CatId; name: string }) => `${c.cat}/${c.name}`;

// Grid cell sizing: min column width in px. Persisted so the chosen density
// survives reloads. Bounds keep cells legible without overflowing the viewport.
const CELL_SIZE_MIN = 72;
const CELL_SIZE_MAX = 280;
const CELL_SIZE_STEP = 24;
const CELL_SIZE_DEFAULT = 112;
const CELL_SIZE_KEY = "grid-cell-size";
const clampCellSize = (n: number) =>
  Math.max(CELL_SIZE_MIN, Math.min(CELL_SIZE_MAX, n));

/** The op POSTed to /api/annotation/[cat]/[name]/box-op. */
type BoxOp =
  | { op: "remove"; id: string }
  | { op: "setLabel"; id: string; label: string | null }
  | { op: "add"; box: Box };

/** A forward box-op on one cell paired with the op that reverts it. */
interface CellOp {
  cell: ReviewBox;
  forward: BoxOp;
  inverse: BoxOp;
}

interface UndoEntry {
  /** Every (cell, inverse-op) pair from one bulk action; undone together. */
  items: { cell: ReviewBox; inverse: BoxOp }[];
}

// Forward/inverse op builders for the three box edits (pure — no component state).
const buildUnbox = (cell: ReviewBox): CellOp => ({
  cell,
  forward: { op: "remove", id: cell.box.id },
  inverse: { op: "add", box: cell.box },
});
const buildUnlabel = (cell: ReviewBox): CellOp => ({
  cell,
  forward: { op: "setLabel", id: cell.box.id, label: null },
  inverse: { op: "setLabel", id: cell.box.id, label: cell.box.label },
});
const buildUnknown = (cell: ReviewBox): CellOp => ({
  cell,
  forward: { op: "setLabel", id: cell.box.id, label: UNKNOWN_LABEL },
  inverse: { op: "setLabel", id: cell.box.id, label: cell.box.label },
});

export default function GridReviewPage() {
  // useSearchParams() needs a Suspense boundary on this statically-routed page.
  return (
    <Suspense
      fallback={
        <main className="fixed inset-0 flex items-center justify-center bg-bg">
          <Spinner size={22} className="text-muted" />
        </main>
      }
    >
      <GridReview />
    </Suspense>
  );
}

function GridReview() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const label = searchParams.get("label") ?? "";
  const cats = useMemo<CatId[]>(() => parseCats(searchParams.get("cats")), [searchParams]);

  // Full result set (immutable for this load) + the set of cells hidden by an
  // action. Rendering off (all − removed) keeps ordering stable across undo.
  const [all, setAll] = useState<ReviewBox[] | null>(null);
  const [removed, setRemoved] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [undoStack, setUndoStack] = useState<UndoEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [cellSize, setCellSize] = useState(CELL_SIZE_DEFAULT);
  const idleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const undoing = useRef(false);

  // Whole-image delete (moves the source image + sidecars to trash), batched so
  // one action can trash several images under a single undo toast. Separate from
  // the box-op undoStack above.
  const {
    pending: pendingDeletes,
    remove: deleteImages,
    undo: undoDelete,
    dismiss: dismissDelete,
  } = useImageBatchDelete();

  // Load the persisted cell size once on mount (kept out of the initial state to
  // avoid an SSR/CSR hydration mismatch on localStorage).
  useEffect(() => {
    const stored = Number(window.localStorage.getItem(CELL_SIZE_KEY));
    if (Number.isFinite(stored) && stored > 0) setCellSize(clampCellSize(stored));
  }, []);

  const changeCellSize = useCallback((delta: number) => {
    setCellSize((s) => {
      const next = clampCellSize(s + delta);
      window.localStorage.setItem(CELL_SIZE_KEY, String(next));
      return next;
    });
  }, []);

  // --- Fetch every box carrying `label` (scoped to cats). --------------------
  useEffect(() => {
    if (!label) {
      setAll([]);
      return;
    }
    let cancelled = false;
    const qs = new URLSearchParams({ label });
    const cparam = searchParams.get("cats");
    if (cparam) qs.set("cats", cparam);
    setAll(null);
    setRemoved(new Set());
    setSelected(new Set());
    setUndoStack([]);
    (async () => {
      try {
        const res = await fetch(`/api/review-boxes?${qs.toString()}`);
        const data = (await res.json()) as ReviewBox[];
        if (!cancelled) setAll(data);
      } catch {
        if (!cancelled) setAll([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [label, searchParams]);

  const visible = useMemo(
    () => (all ?? []).filter((c) => !removed.has(cellKey(c))),
    [all, removed],
  );

  // Selection resolved against what's actually on screen, so a stale key (e.g. a
  // cell removed by another action) can never drive a button.
  const selectedCells = useMemo(
    () => visible.filter((c) => selected.has(cellKey(c))),
    [visible, selected],
  );
  const selCount = selectedCells.length;
  const allSelected = visible.length > 0 && selCount === visible.length;

  // --- Selection. ------------------------------------------------------------
  const toggleSelect = useCallback((cell: ReviewBox) => {
    const k = cellKey(cell);
    setSelected((s) => {
      const n = new Set(s);
      if (n.has(k)) n.delete(k);
      else n.add(k);
      return n;
    });
  }, []);

  const clearSelection = useCallback(() => setSelected(new Set()), []);
  const selectAllVisible = useCallback(
    () => setSelected(new Set(visible.map(cellKey))),
    [visible],
  );

  // --- Links. ----------------------------------------------------------------
  const focusHref = useMemo(() => {
    const target = (all && all.length > 0 ? all[0].n : null);
    if (target == null) return reviewHref(label, cats); // /review/0 — guard snaps forward
    const href = withCats(`/review/${target}`, cats);
    const sep = href.includes("?") ? "&" : "?";
    return `${href}${sep}label=${encodeURIComponent(label)}`;
  }, [all, cats, label]);

  const openFocus = useCallback(
    (cell: ReviewBox) => {
      const href = withCats(`/review/${cell.n}`, cats);
      const sep = href.includes("?") ? "&" : "?";
      router.push(`${href}${sep}label=${encodeURIComponent(label)}`);
    },
    [router, cats, label],
  );

  // --- Box-op POST with save-store feedback. ---------------------------------
  const postOp = useCallback(async (cat: CatId, name: string, op: BoxOp): Promise<boolean> => {
    const setStatus = useSaveStore.getState().setStatus;
    setStatus("saving");
    try {
      const res = await fetch(`/api/annotation/${cat}/${encodeURIComponent(name)}/box-op`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(op),
      });
      if (!res.ok) throw new Error(String(res.status));
      setStatus("saved");
      if (idleTimer.current) clearTimeout(idleTimer.current);
      idleTimer.current = setTimeout(() => setStatus("idle"), 1000);
      return true;
    } catch {
      setStatus("idle");
      return false;
    }
  }, []);

  // --- Bulk box edits (optimistic hide + single grouped undo). ---------------
  // Ops run sequentially: several selected crops can belong to the same image
  // sidecar, and concurrent writes to it would race (lost updates).
  const runBatch = useCallback(
    async (ops: CellOp[]) => {
      if (ops.length === 0) return;
      setError(null);
      const keys = ops.map((o) => cellKey(o.cell));
      setRemoved((s) => {
        const n = new Set(s);
        keys.forEach((k) => n.add(k));
        return n;
      });
      clearSelection();

      const succeeded: { cell: ReviewBox; inverse: BoxOp }[] = [];
      const failedKeys: string[] = [];
      for (let i = 0; i < ops.length; i++) {
        const o = ops[i];
        const ok = await postOp(o.cell.cat, o.cell.name, o.forward);
        if (ok) succeeded.push({ cell: o.cell, inverse: o.inverse });
        else failedKeys.push(keys[i]);
      }

      if (failedKeys.length > 0) {
        setRemoved((s) => {
          const n = new Set(s);
          failedKeys.forEach((k) => n.delete(k));
          return n;
        });
        setError("Couldn't save some changes — try again.");
      }
      if (succeeded.length > 0) {
        setUndoStack((s) => [...s, { items: succeeded }]);
      }
    },
    [postOp, clearSelection],
  );

  const unboxSelected = useCallback(
    () => void runBatch(selectedCells.map(buildUnbox)),
    [runBatch, selectedCells],
  );
  const unlabelSelected = useCallback(
    () => void runBatch(selectedCells.map(buildUnlabel)),
    [runBatch, selectedCells],
  );
  // Relabel each selected box as the catch-all "unknown" kind. Hidden when we're
  // already reviewing unknown_bird (no-op there).
  const canUnknown = label !== UNKNOWN_LABEL;
  const unknownSelected = useCallback(
    () => void runBatch(selectedCells.map(buildUnknown)),
    [runBatch, selectedCells],
  );

  // --- Bulk whole-image delete (dedupe selection by image). ------------------
  const deleteSelected = useCallback(async () => {
    const seen = new Set<string>();
    const images: { cat: CatId; name: string }[] = [];
    for (const c of selectedCells) {
      const k = imageKey(c);
      if (!seen.has(k)) {
        seen.add(k);
        images.push({ cat: c.cat, name: c.name });
      }
    }
    if (images.length === 0) return;
    clearSelection();
    await deleteImages(images, (deleted) => {
      const delKeys = new Set(deleted.map(imageKey));
      setRemoved((s) => {
        const n = new Set(s);
        (all ?? []).forEach((c) => {
          if (delKeys.has(imageKey(c))) n.add(cellKey(c));
        });
        return n;
      });
    });
  }, [selectedCells, deleteImages, all, clearSelection]);

  const handleDeleteUndo = useCallback(() => {
    void undoDelete((restored) => {
      const keys = new Set(restored.map(imageKey));
      setRemoved((s) => {
        const n = new Set(s);
        (all ?? []).forEach((c) => {
          if (keys.has(imageKey(c))) n.delete(cellKey(c));
        });
        return n;
      });
    });
  }, [undoDelete, all]);

  // --- Undo (LIFO; re-show + replay the batch's inverse ops). ----------------
  const undo = useCallback(async () => {
    // Re-entrancy guard: two ⌘Z presses within one frame would otherwise both
    // read the same stale stack top and each pop one entry.
    if (undoing.current) return;
    const entry = undoStack[undoStack.length - 1];
    if (!entry) return;
    undoing.current = true;
    const keys = entry.items.map((it) => cellKey(it.cell));
    setUndoStack((s) => s.slice(0, -1));
    setRemoved((s) => {
      const n = new Set(s);
      keys.forEach((k) => n.delete(k));
      return n;
    });
    try {
      const failed: { cell: ReviewBox; inverse: BoxOp }[] = [];
      const failedKeys: string[] = [];
      for (let i = 0; i < entry.items.length; i++) {
        const it = entry.items[i];
        const ok = await postOp(it.cell.cat, it.cell.name, it.inverse);
        if (!ok) {
          failed.push(it);
          failedKeys.push(keys[i]);
        }
      }
      if (failed.length > 0) {
        // Re-hide only the cells whose revert didn't land, and re-stack just
        // those so a second undo retries exactly them (no double-apply).
        setRemoved((s) => {
          const n = new Set(s);
          failedKeys.forEach((k) => n.add(k));
          return n;
        });
        setUndoStack((s) => [...s, { items: failed }]);
        setError("Couldn't undo some changes — try again.");
      }
    } finally {
      undoing.current = false;
    }
  }, [undoStack, postOp]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target)) return;
      if (e.key === "Escape") {
        e.preventDefault();
        if (selCount > 0) clearSelection();
        else router.push("/");
        return;
      }
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "z") {
        e.preventDefault();
        void undo();
        return;
      }
      if (mod && e.key.toLowerCase() === "a") {
        e.preventDefault();
        selectAllVisible();
        return;
      }
      if (mod && e.key === "Backspace") {
        e.preventDefault();
        if (selCount > 0) void deleteSelected();
        return;
      }
      if (mod) return;
      if (selCount === 0) return;
      const k = e.key.toLowerCase();
      if (k === "b") {
        e.preventDefault();
        unboxSelected();
      } else if (k === "l") {
        e.preventDefault();
        unlabelSelected();
      } else if (k === "u" && canUnknown) {
        e.preventDefault();
        unknownSelected();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    selCount,
    canUnknown,
    clearSelection,
    selectAllVisible,
    undo,
    deleteSelected,
    unboxSelected,
    unlabelSelected,
    unknownSelected,
    router,
  ]);

  // --- Render guards. --------------------------------------------------------
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

  if (all == null) {
    return (
      <main className="fixed inset-0 flex items-center justify-center bg-bg">
        <Spinner size={22} className="text-muted" />
      </main>
    );
  }

  // Selection action bar (only while something is selected). Open shows for a
  // single selection; box edits + the trailing Delete act on the whole selection.
  const selectionSegments: Array<ReactNode | null> = [
    <div key="count" className="flex items-center gap-2">
      <span className="whitespace-nowrap px-1 text-sm text-muted">{selCount} selected</span>
      <ActionButton label="Clear" shortcut="Esc" onClick={clearSelection} />
    </div>,
    <>
      {selCount === 1 && (
        <ActionButton key="open" label="Open" onClick={() => openFocus(selectedCells[0])} />
      )}
      <ActionButton key="unlabel" label="Unlabel" shortcut="L" onClick={unlabelSelected} />
      {canUnknown && (
        <ActionButton
          key="unknown"
          label="Unknown"
          shortcut="U"
          title={`Relabel as ${UNKNOWN_LABEL} (U)`}
          onClick={unknownSelected}
        />
      )}
      <ActionButton key="unbox" label="Unbox" shortcut="B" variant="danger" onClick={unboxSelected} />
    </>,
    <ActionButton
      key="delete"
      label="Delete"
      shortcut="⌘⌫"
      variant="danger"
      title="Delete the whole image (⌘/Ctrl + Backspace)"
      onClick={() => void deleteSelected()}
    />,
  ];

  return (
    <main className="min-h-screen bg-bg text-fg">
      {/* Sticky header: home, label + toggle, density · select-all · count · undo. */}
      <header className="sticky top-0 z-20 flex flex-wrap items-center justify-between gap-3 border-b border-border bg-bg/85 px-4 py-3 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-xs text-muted transition-colors hover:text-fg">
            ← Home
          </Link>
          <span className="text-sm text-fg/80">
            Reviewing <span className="font-mono text-box">{label}</span>
          </span>
          <ViewToggle focusHref={focusHref} />
        </div>
        <div className="flex items-center gap-3">
          {/* Grid density: shrink / grow the cells. */}
          <div className="flex items-center rounded-pill border border-border bg-surface p-0.5">
            <button
              type="button"
              onClick={() => changeCellSize(-CELL_SIZE_STEP)}
              disabled={cellSize <= CELL_SIZE_MIN}
              aria-label="Smaller grid"
              title="Smaller cells"
              className="grid h-9 w-9 place-items-center rounded-pill text-base text-muted touch-manipulation transition-colors hover:text-fg disabled:pointer-events-none disabled:opacity-40"
            >
              −
            </button>
            <button
              type="button"
              onClick={() => changeCellSize(CELL_SIZE_STEP)}
              disabled={cellSize >= CELL_SIZE_MAX}
              aria-label="Larger grid"
              title="Larger cells"
              className="grid h-9 w-9 place-items-center rounded-pill text-base text-muted touch-manipulation transition-colors hover:text-fg disabled:pointer-events-none disabled:opacity-40"
            >
              +
            </button>
          </div>
          <button
            type="button"
            onClick={allSelected ? clearSelection : selectAllVisible}
            disabled={visible.length === 0}
            className="rounded-pill border border-border bg-surface px-3 py-1.5 text-xs text-muted transition-colors hover:border-border-strong hover:text-fg disabled:pointer-events-none disabled:opacity-40"
          >
            {allSelected ? "Select none" : "Select all"}
          </button>
          <span className="font-mono text-xs text-faint">
            {visible.length} box{visible.length === 1 ? "" : "es"}
          </span>
          <button
            type="button"
            onClick={() => void undo()}
            disabled={undoStack.length === 0}
            className="flex items-center gap-1.5 rounded-pill border border-border bg-surface px-3 py-1.5 text-xs text-muted transition-colors hover:border-border-strong hover:text-fg disabled:pointer-events-none disabled:opacity-40"
          >
            <kbd className="grid h-4 min-w-4 place-items-center rounded border border-border-strong px-1 font-mono text-[10px] text-faint">
              ⌘Z
            </kbd>
            Undo{undoStack.length > 0 ? ` (${undoStack.length})` : ""}
          </button>
        </div>
      </header>

      {error && (
        <div className="mx-4 mt-3 rounded-xl border border-danger/40 bg-surface px-4 py-2 text-sm text-danger">
          {error}
        </div>
      )}

      {visible.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-3 py-32 text-center">
          <span className="text-sm text-muted">
            No images labeled <span className="font-mono text-fg">{label}</span> in the selected
            categories.
          </span>
          <Link href="/" className="text-xs text-faint hover:text-fg transition-colors">
            ← Home
          </Link>
        </div>
      ) : (
        <div
          className={`grid touch-manipulation gap-2 p-4 ${selCount > 0 ? "pb-40" : ""}`}
          style={{
            gridTemplateColumns: `repeat(auto-fill, minmax(${cellSize}px, 1fr))`,
          }}
        >
          {visible.map((cell) => (
            <CropCell
              key={cellKey(cell)}
              cell={cell}
              selected={selected.has(cellKey(cell))}
              onToggle={toggleSelect}
            />
          ))}
        </div>
      )}

      {selCount > 0 && <BottomBar segments={selectionSegments} />}

      {pendingDeletes.length > 0 && (
        <DeleteToast
          name={pendingDeletes.length === 1 ? pendingDeletes[0].name : `${pendingDeletes.length} images`}
          onUndo={handleDeleteUndo}
          onDismiss={dismissDelete}
        />
      )}
    </main>
  );
}

/** Focus | Grid segmented control (Grid is the active page). */
function ViewToggle({ focusHref }: { focusHref: string }) {
  return (
    <div className="flex items-center rounded-pill border border-border bg-surface p-0.5 text-xs">
      <Link
        href={focusHref}
        className="rounded-pill px-2.5 py-1 text-muted transition-colors hover:text-fg"
      >
        Focus
      </Link>
      <span className="rounded-pill bg-fg px-2.5 py-1 font-medium text-bg">Grid</span>
    </div>
  );
}
