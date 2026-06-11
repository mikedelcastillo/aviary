"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { CatToggle } from "@/components/CatToggle";
import { LoadingOverlay } from "@/components/LoadingOverlay";
import { useSaveStore } from "@/lib/save-store";
import {
  ALL_CATS,
  DEDUPE_DEFAULT_THRESHOLD,
  parseCats,
  serializeCats,
  type CatId,
  type DedupeCluster,
} from "@/lib/types";

const CATS_STORAGE_KEY = "aviary.cats";
const MIN_T = 0;
const MAX_T = 20;

/** A committed group, kept on a stack so removals can be undone in order. */
interface Committed {
  idx: number;
  cat: CatId;
  names: string[];
}

export default function DedupePage() {
  const router = useRouter();
  const setSaveStatus = useSaveStore((s) => s.setStatus);

  const [cats, setCats] = useState<CatId[]>(ALL_CATS);
  const [threshold, setThreshold] = useState(DEDUPE_DEFAULT_THRESHOLD);
  const [clusters, setClusters] = useState<DedupeCluster[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [coldRun, setColdRun] = useState(true); // first fetch hashes everything
  const [error, setError] = useState<string | null>(null);

  const [groupIdx, setGroupIdx] = useState(0);
  const [removeSet, setRemoveSet] = useState<Set<string>>(new Set());
  const [history, setHistory] = useState<Committed[]>([]);
  const [removedCount, setRemovedCount] = useState(0);
  const [busy, setBusy] = useState(false);

  // Restore the saved category selection after mount (shared with Box/Label).
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(CATS_STORAGE_KEY);
      if (saved) setCats(parseCats(saved));
    } catch {
      /* localStorage unavailable — keep default */
    }
    return () => setSaveStatus("idle");
  }, [setSaveStatus]);

  useEffect(() => {
    try {
      window.localStorage.setItem(CATS_STORAGE_KEY, serializeCats(cats) ?? "");
    } catch {
      /* ignore */
    }
  }, [cats]);

  // Fetch clusters whenever the category scope or threshold changes. After the
  // first run hashes are cached, so threshold changes re-cluster quickly.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    const params = new URLSearchParams();
    const c = serializeCats(cats);
    if (c) params.set("cats", c);
    params.set("threshold", String(threshold));
    (async () => {
      try {
        const res = await fetch(`/api/dedupe?${params}`);
        if (!res.ok) throw new Error(`dedupe: ${res.status}`);
        const data = (await res.json()) as DedupeCluster[];
        if (cancelled) return;
        setClusters(data);
        setGroupIdx(0);
        setHistory([]);
        setRemovedCount(0);
        setColdRun(false);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [cats, threshold]);

  const group = clusters && groupIdx < clusters.length ? clusters[groupIdx] : null;

  // Re-seed the keep/remove decision when the active group changes: every
  // non-keeper without labels defaults to "remove"; keepers and labeled frames
  // default to "keep" (warn-but-allow — labeled frames can still be toggled off).
  useEffect(() => {
    if (!group) {
      setRemoveSet(new Set());
      return;
    }
    const next = new Set<string>();
    for (const m of group.members) {
      if (m.name !== group.keepName && !m.hasLabels) next.add(m.name);
    }
    setRemoveSet(next);
  }, [group]);

  const keptCount = group ? group.members.length - removeSet.size : 0;

  const toggle = useCallback(
    (name: string) => {
      setRemoveSet((prev) => {
        const next = new Set(prev);
        if (next.has(name)) {
          next.delete(name);
        } else {
          // Never let the user mark the last remaining keeper for removal.
          if (group && group.members.length - next.size <= 1) return prev;
          next.add(name);
        }
        return next;
      });
    },
    [group],
  );

  // Whether every frame except the suggested keeper is currently marked for removal.
  const allButKeeperSelected =
    !!group &&
    group.members.length > 1 &&
    removeSet.size === group.members.length - 1 &&
    !removeSet.has(group.keepName);

  // Toggle the whole group: mark all frames except the keeper for removal, or — if
  // already in that state — clear back to keeping everything. Overrides the
  // labeled-frame default (warn-but-allow: labeled frames can be bulk-selected too).
  const selectAll = useCallback(() => {
    setRemoveSet((prev) => {
      if (!group) return prev;
      const allBut =
        group.members.length > 1 &&
        prev.size === group.members.length - 1 &&
        !prev.has(group.keepName);
      if (allBut) return new Set();
      return new Set(group.members.filter((m) => m.name !== group.keepName).map((m) => m.name));
    });
  }, [group]);

  const advance = useCallback(() => setGroupIdx((i) => i + 1), []);

  const approve = useCallback(async () => {
    if (!group || busy) return;
    const names = [...removeSet];
    if (names.length === 0) {
      advance();
      return;
    }
    setBusy(true);
    setSaveStatus("saving");
    try {
      const res = await fetch("/api/dedupe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cat: group.cat, remove: names }),
      });
      if (!res.ok) throw new Error(`commit: ${res.status}`);
      setHistory((h) => [...h, { idx: groupIdx, cat: group.cat, names }]);
      setRemovedCount((n) => n + names.length);
      setSaveStatus("saved");
      advance();
    } catch (e) {
      setError((e as Error).message);
      setSaveStatus("idle");
    } finally {
      setBusy(false);
    }
  }, [group, busy, removeSet, groupIdx, advance, setSaveStatus]);

  const undo = useCallback(async () => {
    if (busy || history.length === 0) return;
    const last = history[history.length - 1];
    setBusy(true);
    setSaveStatus("saving");
    try {
      const res = await fetch("/api/dedupe/restore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cat: last.cat, restore: last.names }),
      });
      if (!res.ok) throw new Error(`restore: ${res.status}`);
      setHistory((h) => h.slice(0, -1));
      setRemovedCount((n) => Math.max(0, n - last.names.length));
      setGroupIdx(last.idx); // return to the group so it can be re-reviewed
      setSaveStatus("saved");
    } catch (e) {
      setError((e as Error).message);
      setSaveStatus("idle");
    } finally {
      setBusy(false);
    }
  }, [busy, history, setSaveStatus]);

  const bumpThreshold = useCallback((delta: number) => {
    setThreshold((t) => Math.min(MAX_T, Math.max(MIN_T, t + delta)));
  }, []);

  // Back/undo: if a group was just committed, undo it (restores files + returns to
  // that group); otherwise step back to the previous group to re-review it.
  const back = useCallback(() => {
    if (history.length > 0) {
      undo();
    } else {
      setGroupIdx((i) => Math.max(0, i - 1));
    }
  }, [history.length, undo]);

  // Keyboard map — consistent with Box/Label modes.
  const groupRef = useRef(group);
  groupRef.current = group;
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null;
      const tag = t?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || t?.isContentEditable) return;
      switch (e.key) {
        case "Escape":
          router.push("/");
          return;
        // Proceed: delete the selected frames and advance (or just advance if none).
        case "ArrowRight":
        case " ":
        case "Enter":
          e.preventDefault();
          approve();
          return;
        // Back/undo: undo the last commit, else step to the previous group.
        case "ArrowLeft":
          e.preventDefault();
          back();
          return;
        // Select every frame except the suggested keeper (toggle).
        case "a":
        case "A":
          e.preventDefault();
          selectAll();
          return;
        // Skip this group without deleting anything.
        case "s":
        case "S":
          e.preventDefault();
          advance();
          return;
        case "[":
          e.preventDefault();
          bumpThreshold(-1);
          return;
        case "]":
          e.preventDefault();
          bumpThreshold(1);
          return;
      }
      if (e.key >= "1" && e.key <= "9") {
        const g = groupRef.current;
        if (g) {
          const idx = Number(e.key) - 1;
          const m = g.members[idx];
          if (m && !m.hasLabels) toggle(m.name);
          // Labeled frames are intentionally not toggled via number keys (click instead).
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [router, approve, advance, back, bumpThreshold, toggle, selectAll]);

  const total = clusters?.length ?? 0;
  const done = !loading && clusters !== null && groupIdx >= total;

  const loadingLabel = useMemo(
    () => (coldRun ? "Hashing images… (first run)" : "Finding near-duplicates…"),
    [coldRun],
  );

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <LoadingOverlay show={loading} label={loadingLabel} />

      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-3">
            <Link href="/" className="text-sm text-muted hover:text-fg">
              ← Home
            </Link>
            <h1 className="text-xl font-semibold tracking-tight text-fg">Dedupe</h1>
          </div>
          <p className="mt-1 text-sm text-muted">
            Review near-duplicate frames and remove redundant ones. Removed files move to{" "}
            <span className="font-mono text-faint">_dedup_removed/</span> (reversible).
          </p>
        </div>
        <ThresholdControl value={threshold} onBump={bumpThreshold} />
      </header>

      <div className="mt-6">
        <CatToggle selected={cats} totals={null} onChange={setCats} />
      </div>

      {error && (
        <div className="mt-8 rounded-xl border border-danger/40 bg-surface p-5 text-sm text-danger">
          {error}
        </div>
      )}

      {!loading && !error && clusters !== null && total === 0 && (
        <EmptyState threshold={threshold} />
      )}

      {!loading && !error && done && total > 0 && (
        <DoneState
          groups={total}
          removed={removedCount}
          canUndo={history.length > 0}
          onUndo={undo}
        />
      )}

      {group && !loading && (
        <section className="mt-6">
          <GroupProgress idx={groupIdx} total={total} cat={group.cat} removed={removedCount} />

          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
            {group.members.map((m, i) => {
              const removing = removeSet.has(m.name);
              const isKeepSuggestion = m.name === group.keepName;
              return (
                <ThumbCard
                  key={m.name}
                  cat={group.cat}
                  member={m}
                  index={i}
                  removing={removing}
                  isKeepSuggestion={isKeepSuggestion}
                  onClick={() => toggle(m.name)}
                />
              );
            })}
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={approve}
              disabled={busy}
              className="cursor-pointer rounded-pill border border-danger/40 bg-danger/10 px-4 py-2 text-sm font-medium text-danger transition-colors hover:bg-danger/20 disabled:opacity-50"
            >
              {removeSet.size > 0 ? `Remove ${removeSet.size} & next` : "Nothing to remove"}{" "}
              <kbd className="ml-1 rounded border border-danger/30 px-1 font-mono text-[11px]">→</kbd>
            </button>
            <button
              type="button"
              onClick={selectAll}
              disabled={busy}
              className="cursor-pointer rounded-pill border border-border bg-surface px-4 py-2 text-sm text-muted transition-colors hover:border-border-strong hover:text-fg disabled:opacity-50"
            >
              {allButKeeperSelected ? "Keep all" : "Select all"}{" "}
              <kbd className="ml-1 rounded border border-border px-1 font-mono text-[11px]">A</kbd>
            </button>
            <button
              type="button"
              onClick={advance}
              disabled={busy}
              className="cursor-pointer rounded-pill border border-border bg-surface px-4 py-2 text-sm text-muted transition-colors hover:border-border-strong hover:text-fg disabled:opacity-50"
            >
              Skip <kbd className="ml-1 rounded border border-border px-1 font-mono text-[11px]">S</kbd>
            </button>
            <button
              type="button"
              onClick={undo}
              disabled={busy || history.length === 0}
              className="cursor-pointer rounded-pill border border-border bg-surface px-4 py-2 text-sm text-muted transition-colors hover:border-border-strong hover:text-fg disabled:opacity-40"
            >
              Undo last <kbd className="ml-1 rounded border border-border px-1 font-mono text-[11px]">←</kbd>
            </button>
            <span className="ml-auto font-mono text-xs text-faint">
              keeping {keptCount} · removing {removeSet.size}
            </span>
          </div>

          <p className="mt-3 text-xs text-faint">
            → / Space delete selected &amp; next · ← back/undo · A select all but the keeper · S skip · 1–9 toggle a frame · green = keep, red = remove
          </p>
        </section>
      )}
    </main>
  );
}

function ThresholdControl({ value, onBump }: { value: number; onBump: (d: number) => void }) {
  return (
    <div className="flex items-center gap-2 rounded-pill border border-border bg-surface px-3 py-1.5">
      <span className="text-xs text-muted">similarity distance</span>
      <button
        type="button"
        onClick={() => onBump(-1)}
        className="cursor-pointer rounded px-1.5 font-mono text-sm text-muted hover:text-fg"
        aria-label="stricter"
      >
        −
      </button>
      <span className="w-5 text-center font-mono text-sm text-fg">{value}</span>
      <button
        type="button"
        onClick={() => onBump(1)}
        className="cursor-pointer rounded px-1.5 font-mono text-sm text-muted hover:text-fg"
        aria-label="looser"
      >
        +
      </button>
      <kbd className="rounded border border-border px-1 font-mono text-[11px] text-faint">[ ]</kbd>
    </div>
  );
}

function GroupProgress({
  idx,
  total,
  cat,
  removed,
}: {
  idx: number;
  total: number;
  cat: CatId;
  removed: number;
}) {
  const pct = total > 0 ? ((idx + 1) / total) * 100 : 0;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-sm text-fg">
          Group <span className="font-mono">{idx + 1}</span> of{" "}
          <span className="font-mono">{total}</span> ·{" "}
          <span className="font-mono text-muted">{cat}</span>
        </span>
        <span className="font-mono text-xs text-faint">{removed} removed</span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-elevated">
        <div className="h-full rounded-full bg-box transition-[width]" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ThumbCard({
  cat,
  member,
  index,
  removing,
  isKeepSuggestion,
  onClick,
}: {
  cat: CatId;
  member: { name: string; hasLabels: boolean; sizeBytes: number; dist: number };
  index: number;
  removing: boolean;
  isKeepSuggestion: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        "group relative flex cursor-pointer flex-col overflow-hidden rounded-xl border-2 bg-surface text-left transition-colors " +
        (removing ? "border-danger" : "border-box")
      }
    >
      <div className="flex h-36 items-center justify-center bg-bg">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`/api/thumb/${cat}/${encodeURIComponent(member.name)}?w=280`}
          alt={member.name}
          draggable={false}
          className={"max-h-full max-w-full object-contain " + (removing ? "opacity-50" : "")}
        />
      </div>
      <div className="absolute left-1.5 top-1.5 flex items-center gap-1">
        <span className="rounded bg-bg/70 px-1 font-mono text-[10px] text-faint">{index + 1}</span>
        {member.hasLabels && (
          <span className="rounded bg-box/20 px-1 font-mono text-[10px] text-box">labels</span>
        )}
      </div>
      <span
        className={
          "absolute right-1.5 top-1.5 rounded px-1.5 py-0.5 font-mono text-[10px] " +
          (removing ? "bg-danger/20 text-danger" : "bg-box/20 text-box")
        }
      >
        {removing ? "remove" : isKeepSuggestion ? "keep ★" : "keep"}
      </span>
      <span className="px-2 py-1 font-mono text-[10px] text-faint">
        d={member.dist} · {(member.sizeBytes / 1000).toFixed(0)}k
      </span>
    </button>
  );
}

function GroupNote({ children }: { children: ReactNode }) {
  return <p className="mt-2 text-sm text-muted">{children}</p>;
}

function EmptyState({ threshold }: { threshold: number }) {
  return (
    <div className="mt-10 rounded-2xl border border-border bg-surface p-8 text-center">
      <p className="text-sm text-fg">No near-duplicate groups at distance {threshold}.</p>
      <GroupNote>Increase the similarity distance with the + control (or the ] key) to find looser matches.</GroupNote>
    </div>
  );
}

function DoneState({
  groups,
  removed,
  canUndo,
  onUndo,
}: {
  groups: number;
  removed: number;
  canUndo: boolean;
  onUndo: () => void;
}) {
  return (
    <div className="mt-10 rounded-2xl border border-border bg-surface p-8 text-center">
      <p className="text-base font-semibold text-fg">All {groups} groups reviewed</p>
      <GroupNote>
        Removed <span className="font-mono text-fg">{removed}</span> image
        {removed === 1 ? "" : "s"} to <span className="font-mono text-faint">_dedup_removed/</span>.
      </GroupNote>
      <div className="mt-5 flex items-center justify-center gap-2">
        {canUndo && (
          <button
            type="button"
            onClick={onUndo}
            className="cursor-pointer rounded-pill border border-border bg-surface px-4 py-2 text-sm text-muted hover:border-border-strong hover:text-fg"
          >
            Undo last group <kbd className="ml-1 rounded border border-border px-1 font-mono text-[11px]">Z</kbd>
          </button>
        )}
        <Link
          href="/"
          className="rounded-pill bg-fg px-4 py-2 text-sm font-medium text-bg transition-opacity hover:opacity-90"
        >
          Home
        </Link>
      </div>
    </div>
  );
}
