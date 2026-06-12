"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CropCell } from "@/components/CropCell";
import { DeleteToast } from "@/components/DeleteToast";
import { Spinner } from "@/components/Spinner";
import { useImageDelete } from "@/lib/use-image-delete";
import { useSaveStore } from "@/lib/save-store";
import { parseCats, reviewHref, withCats, type Box, type CatId, type ReviewBox } from "@/lib/types";

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

const cellKey = (c: ReviewBox) => `${c.cat}/${c.name}/${c.box.id}`;

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

interface UndoEntry {
  cell: ReviewBox;
  /** Op that reverts the forward action (re-adds the box / restores the label). */
  inverse: BoxOp;
}

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
  const [undoStack, setUndoStack] = useState<UndoEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [cellSize, setCellSize] = useState(CELL_SIZE_DEFAULT);
  const idleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const undoing = useRef(false);

  // Whole-image delete (moves the source image + sidecars to trash) with an
  // ephemeral undo toast. Separate from the box-op undoStack above.
  const {
    pending: pendingDelete,
    remove: deleteImage,
    undo: undoDelete,
    dismiss: dismissDelete,
  } = useImageDelete();

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

  // --- Forward actions (optimistic hide + revert on failure). ----------------
  const act = useCallback(
    async (cell: ReviewBox, forward: BoxOp, inverse: BoxOp) => {
      const k = cellKey(cell);
      setError(null);
      setRemoved((s) => new Set(s).add(k));
      const entry: UndoEntry = { cell, inverse };
      setUndoStack((s) => [...s, entry]);
      const ok = await postOp(cell.cat, cell.name, forward);
      if (!ok) {
        setRemoved((s) => {
          const n = new Set(s);
          n.delete(k);
          return n;
        });
        setUndoStack((s) => s.filter((e) => e !== entry));
        setError("Couldn't save that change — try again.");
      }
    },
    [postOp],
  );

  const unbox = useCallback(
    (cell: ReviewBox) => act(cell, { op: "remove", id: cell.box.id }, { op: "add", box: cell.box }),
    [act],
  );
  const unlabel = useCallback(
    (cell: ReviewBox) =>
      act(
        cell,
        { op: "setLabel", id: cell.box.id, label: null },
        { op: "setLabel", id: cell.box.id, label: cell.box.label },
      ),
    [act],
  );

  // --- Whole-image delete (hides every cell owned by the image). -------------
  // Hiding runs inside the onDeleted callback, which the hook only fires after a
  // successful move — so a failed delete leaves the cells visible.
  const handleDelete = useCallback(
    async (cell: ReviewBox) => {
      const keys = (all ?? [])
        .filter((c) => c.cat === cell.cat && c.name === cell.name)
        .map(cellKey);
      await deleteImage(cell.cat, cell.name, () => {
        setRemoved((s) => {
          const n = new Set(s);
          keys.forEach((k) => n.add(k));
          return n;
        });
      });
    },
    [all, deleteImage],
  );

  const handleDeleteUndo = useCallback(() => {
    void undoDelete((p) => {
      setRemoved((s) => {
        const n = new Set(s);
        (all ?? [])
          .filter((c) => c.cat === p.cat && c.name === p.name)
          .forEach((c) => n.delete(cellKey(c)));
        return n;
      });
    });
  }, [undoDelete, all]);

  // --- Undo (LIFO; re-show + send the inverse op). ---------------------------
  const undo = useCallback(async () => {
    // Re-entrancy guard: two ⌘Z presses within one frame would otherwise both
    // read the same stale stack top and each pop one entry, stranding the
    // second-from-top cell hidden forever with its inverse op never sent.
    if (undoing.current) return;
    const entry = undoStack[undoStack.length - 1];
    if (!entry) return;
    undoing.current = true;
    const k = cellKey(entry.cell);
    setUndoStack((s) => s.slice(0, -1));
    setRemoved((s) => {
      const n = new Set(s);
      n.delete(k);
      return n;
    });
    try {
      const ok = await postOp(entry.cell.cat, entry.cell.name, entry.inverse);
      if (!ok) {
        setRemoved((s) => new Set(s).add(k));
        setUndoStack((s) => [...s, entry]);
        setError("Couldn't undo — try again.");
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
        router.push("/");
        return;
      }
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "z") {
        e.preventDefault();
        void undo();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [undo, router]);

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

  return (
    <main className="min-h-screen bg-bg text-fg">
      {/* Sticky header: home, label + toggle, count + undo. */}
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
              className="grid h-6 w-6 place-items-center rounded-pill text-sm text-muted transition-colors hover:text-fg disabled:pointer-events-none disabled:opacity-40"
            >
              −
            </button>
            <button
              type="button"
              onClick={() => changeCellSize(CELL_SIZE_STEP)}
              disabled={cellSize >= CELL_SIZE_MAX}
              aria-label="Larger grid"
              title="Larger cells"
              className="grid h-6 w-6 place-items-center rounded-pill text-sm text-muted transition-colors hover:text-fg disabled:pointer-events-none disabled:opacity-40"
            >
              +
            </button>
          </div>
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
          className="grid gap-2 p-4"
          style={{
            gridTemplateColumns: `repeat(auto-fill, minmax(${cellSize}px, 1fr))`,
          }}
        >
          {visible.map((cell) => (
            <CropCell
              key={cellKey(cell)}
              cell={cell}
              onOpen={openFocus}
              onUnbox={unbox}
              onUnlabel={unlabel}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}

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
