"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CropCell } from "@/components/CropCell";
import { Spinner } from "@/components/Spinner";
import { useSaveStore } from "@/lib/save-store";
import { parseCats, reviewHref, withCats, type Box, type CatId, type ReviewBox } from "@/lib/types";

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

const cellKey = (c: ReviewBox) => `${c.cat}/${c.name}/${c.box.id}`;

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
  const idleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

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

  // --- Undo (LIFO; re-show + send the inverse op). ---------------------------
  const undo = useCallback(async () => {
    const entry = undoStack[undoStack.length - 1];
    if (!entry) return;
    const k = cellKey(entry.cell);
    setUndoStack((s) => s.slice(0, -1));
    setRemoved((s) => {
      const n = new Set(s);
      n.delete(k);
      return n;
    });
    const ok = await postOp(entry.cell.cat, entry.cell.name, entry.inverse);
    if (!ok) {
      setRemoved((s) => new Set(s).add(k));
      setUndoStack((s) => [...s, entry]);
      setError("Couldn't undo — try again.");
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
        <div className="grid grid-cols-[repeat(auto-fill,minmax(112px,1fr))] gap-2 p-4">
          {visible.map((cell) => (
            <CropCell
              key={cellKey(cell)}
              cell={cell}
              onOpen={openFocus}
              onUnbox={unbox}
              onUnlabel={unlabel}
            />
          ))}
        </div>
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
