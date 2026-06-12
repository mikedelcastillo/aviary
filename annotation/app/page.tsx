"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ProgressCard } from "@/components/ProgressCard";
import { CatToggle } from "@/components/CatToggle";
import {
  ALL_CATS,
  CATEGORIES,
  newSeed,
  orderBySeed,
  parseCats,
  reviewHref,
  serializeCats,
  withCats,
  withNav,
  type CatId,
  type CategoryProgress,
} from "@/lib/types";
import { useHistoryStore } from "@/lib/history-store";

const CATS_STORAGE_KEY = "aviary.cats";
const RANDOM_STORAGE_KEY = "aviary.random";

interface LabelStat {
  label: string;
  count: number;
}

function pct(n: number, total: number): number {
  return total > 0 ? Math.round((n / total) * 100) : 0;
}

interface Totals {
  total: number;
  boxed: number;
  labeled: number;
}

function sumTotals(progress: CategoryProgress[]): Totals {
  return progress.reduce<Totals>(
    (acc, p) => ({
      total: acc.total + p.total,
      boxed: acc.boxed + p.boxed,
      labeled: acc.labeled + p.labeled,
    }),
    { total: 0, boxed: 0, labeled: 0 },
  );
}

export default function Home() {
  const router = useRouter();
  const [progress, setProgress] = useState<CategoryProgress[] | null>(null);
  const [queue, setQueue] = useState<number[] | null>(null);
  const [boxQueue, setBoxQueue] = useState<number[] | null>(null);
  const [entry, setEntry] = useState<{ box: number; label: number } | null>(null);
  const [labelStats, setLabelStats] = useState<LabelStat[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cats, setCats] = useState<CatId[]>(ALL_CATS);
  const [random, setRandom] = useState(false);

  // Restore the saved selection after mount (avoids SSR hydration mismatch).
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(CATS_STORAGE_KEY);
      if (saved) setCats(parseCats(saved));
      setRandom(window.localStorage.getItem(RANDOM_STORAGE_KEY) === "1");
    } catch {
      /* localStorage unavailable — keep default */
    }
    // Home is the boundary between sessions — drop the cross-image undo timeline
    // so each Box/Label run starts with a clean history.
    useHistoryStore.getState().clear();
  }, []);

  // Persist the selection whenever it changes.
  useEffect(() => {
    try {
      window.localStorage.setItem(CATS_STORAGE_KEY, serializeCats(cats) ?? "");
    } catch {
      /* ignore */
    }
  }, [cats]);

  // Persist the randomize preference whenever it changes.
  useEffect(() => {
    try {
      window.localStorage.setItem(RANDOM_STORAGE_KEY, random ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [random]);

  // Per-category progress is selection-independent — fetch once.
  useEffect(() => {
    let cancelled = false;
    fetch("/api/progress")
      .then((res) => {
        if (!res.ok) throw new Error(`progress: ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setProgress(data as CategoryProgress[]);
      })
      .catch((e) => {
        if (!cancelled) setError((e as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Entry points, queue, and the leaderboard are scoped to the selection. Each
  // endpoint loads on its own so a slow one never blocks the others — every
  // module renders a skeleton until its own data lands. Resetting to null on
  // selection change snaps the affected modules back to their skeleton state.
  useEffect(() => {
    let cancelled = false;
    const param = serializeCats(cats);
    const qs = param ? `?cats=${param}` : "";

    setQueue(null);
    setBoxQueue(null);
    setEntry(null);
    setLabelStats(null);

    function load<T>(url: string, set: (v: T) => void): void {
      fetch(url)
        .then((res) => {
          if (!res.ok) throw new Error(`${url}: ${res.status}`);
          return res.json();
        })
        .then((data) => {
          if (!cancelled) set(data as T);
        })
        .catch((e) => {
          if (!cancelled) setError((e as Error).message);
        });
    }

    load<number[]>(`/api/queue${qs}`, setQueue);
    load<number[]>(`/api/box-queue${qs}`, setBoxQueue);
    load<{ box: number; label: number }>(`/api/entry${qs}`, setEntry);
    load<LabelStat[]>(`/api/label-stats${qs}`, setLabelStats);

    return () => {
      cancelled = true;
    };
  }, [cats]);

  const catSet = useMemo(() => new Set(cats), [cats]);
  const selectedProgress = useMemo(
    () => (progress ? progress.filter((p) => catSet.has(p.id)) : []),
    [progress, catSet],
  );
  const totals = progress ? sumTotals(selectedProgress) : null;
  const categoryTotals = useMemo(() => {
    if (!progress) return null;
    return progress.reduce(
      (acc, p) => {
        acc[p.id] = p.total;
        return acc;
      },
      {} as Record<CatId, number>,
    );
  }, [progress]);

  const queueLen = queue?.length ?? 0;
  // Mode buttons jump to the first unboxed / first unlabeled image in selection.
  const boxTarget = entry?.box ?? 0;
  const labelTarget = entry?.label ?? (queue && queue.length > 0 ? queue[0] : 0);
  const labelEmpty = queue !== null && queue.length === 0;

  // Enter a mode. In random mode, start at the FIRST image of the seeded shuffle
  // (not the global-first one) so the forward-only walk covers the whole worklist
  // and only finishes when everything is actually done. The same seed drives the
  // URL and the target, so the entry matches the page's computed sequence.
  const enterBox = () => {
    const seed = random ? newSeed() : null;
    const target =
      seed != null && boxQueue && boxQueue.length > 0 ? orderBySeed([...boxQueue], seed)[0] : boxTarget;
    router.push(withNav(`/box/${target}`, cats, seed));
  };
  const enterLabel = () => {
    if (labelEmpty) return;
    const seed = random ? newSeed() : null;
    const target =
      seed != null && queue && queue.length > 0 ? orderBySeed([...queue], seed)[0] : labelTarget;
    router.push(withNav(`/label/${target}`, cats, seed));
  };

  return (
    <main className="mx-auto max-w-5xl px-6 py-16">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-fg sm:text-3xl">
          Aviary Annotation
        </h1>
        <p className="mt-1 text-sm text-muted">
          Box and label bird detections — filesystem is the source of truth.
        </p>
        {totals ? (
          <p className="mt-3 font-mono text-xs text-faint">
            {totals.total.toLocaleString()} images · {totals.boxed.toLocaleString()} boxed (
            {pct(totals.boxed, totals.total)}%) · {totals.labeled.toLocaleString()} labeled (
            {pct(totals.labeled, totals.boxed)}%)
          </p>
        ) : (
          <div className="mt-3 h-4 w-72 max-w-full animate-pulse rounded bg-elevated" />
        )}
      </header>

      {error && (
        <div className="mt-8 rounded-xl border border-danger/40 bg-surface p-5 text-sm text-danger">
          Failed to load: {error}
        </div>
      )}

      {/* Category selector (left) + randomize toggle (right). Scopes and
          orders everything below. Both are interactive immediately — the
          per-category counts pop in once progress lands. */}
      <div className="mt-8 flex flex-wrap items-center justify-between gap-2">
        <CatToggle selected={cats} totals={categoryTotals} onChange={setCats} />
        <button
          type="button"
          role="checkbox"
          aria-checked={random}
          onClick={() => setRandom(!random)}
          title="Shuffle image order each time you enter Box/Label"
          className={
            "flex cursor-pointer items-center gap-2 rounded-pill border px-3.5 py-1.5 text-sm transition-colors " +
            (random
              ? "border-fg bg-fg text-bg"
              : "border-border bg-surface text-muted hover:border-border-strong hover:text-fg")
          }
        >
          <span
            aria-hidden
            className={
              "flex h-3.5 w-3.5 items-center justify-center rounded-[4px] border text-[9px] leading-none " +
              (random ? "border-bg/40 bg-bg/20 text-bg" : "border-border-strong text-transparent")
            }
          >
            ✓
          </span>
          <span className="font-medium">Randomize</span>
        </button>
      </div>

      {/* Primary entry points */}
      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <button
          type="button"
          onClick={enterBox}
          className="group flex cursor-pointer flex-col rounded-2xl bg-fg p-6 text-left text-bg transition-opacity hover:opacity-90"
        >
          <span className="text-lg font-semibold">Box mode</span>
          <span className="mt-1 text-sm opacity-70">Draw &amp; vet bounding boxes</span>
          {totals ? (
            <span className="mt-4 font-mono text-xs opacity-60">
              {totals.boxed}/{totals.total} boxed
            </span>
          ) : (
            <span className="mt-4 h-4 w-20 animate-pulse rounded bg-bg/20" />
          )}
        </button>

        <button
          type="button"
          onClick={enterLabel}
          aria-disabled={labelEmpty}
          title={labelEmpty ? "Nothing to label yet" : undefined}
          className={
            "group flex cursor-pointer flex-col rounded-2xl border p-6 text-left transition-colors " +
            (labelEmpty
              ? "border-border bg-surface text-faint hover:border-border-strong"
              : "border-border-strong bg-surface text-fg hover:bg-surface-2")
          }
        >
          <span className="text-lg font-semibold">Label mode</span>
          <span className="mt-1 text-sm text-muted">Assign roster labels to boxed birds</span>
          {queue === null ? (
            <span className="mt-4 h-4 w-24 animate-pulse rounded bg-elevated" />
          ) : (
            <span className="mt-4 font-mono text-xs text-faint">
              {labelEmpty ? "Nothing to label yet" : `${queueLen} in queue`}
            </span>
          )}
        </button>
      </div>

      {/* Auxiliary mode — understated, below the primary entry points. */}
      <Link
        href={withCats("/dedupe", cats)}
        className="mt-3 flex items-center justify-between rounded-xl border border-border bg-surface px-5 py-3 text-sm transition-colors hover:border-border-strong hover:bg-surface-2"
      >
        <span className="font-medium text-fg">Dedupe</span>
        <span className="text-muted">Find &amp; remove near-duplicate frames →</span>
      </Link>

      {/* Per-category progress — unselected categories dimmed. Skeleton cards
          hold the layout until progress lands. */}
      <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {progress
          ? progress.map((p) => (
              <div key={p.id} className={catSet.has(p.id) ? "" : "opacity-40 transition-opacity"}>
                <ProgressCard progress={p} />
              </div>
            ))
          : CATEGORIES.map((c) => <ProgressCardSkeleton key={c.id} />)}
      </div>

      {/* Label leaderboard — renders its own skeleton until stats arrive. */}
      <LabelLeaderboard stats={labelStats} cats={cats} />
    </main>
  );
}

function LabelLeaderboard({ stats, cats }: { stats: LabelStat[] | null; cats: CatId[] }) {
  const max = stats ? stats.reduce((m, s) => Math.max(m, s.count), 0) : 0;
  return (
    <section className="mt-12">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-base font-semibold text-fg">Labels</h2>
        <span className="font-mono text-xs text-faint">ranked by labeled count</span>
      </div>
      <div className="rounded-2xl border border-border bg-surface p-5">
        {stats === null ? (
          <LeaderboardSkeleton />
        ) : (
          <ol className="space-y-2.5">
            {stats.map((s, i) => (
              <li key={s.label} className="group flex items-center gap-3">
                <span className="w-6 text-right font-mono text-xs text-faint">{i + 1}</span>
                <span className="w-28 shrink-0 truncate text-sm text-fg">{s.label}</span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-elevated">
                  <div
                    className="h-full rounded-full bg-box transition-[width]"
                    style={{ width: max > 0 ? `${(s.count / max) * 100}%` : "0%" }}
                  />
                </div>
                <span className="w-12 shrink-0 text-right font-mono text-sm text-muted">{s.count}</span>
                {s.count > 0 ? (
                  <Link
                    href={reviewHref(s.label, cats)}
                    className="shrink-0 rounded-pill border border-border px-2.5 py-1 text-xs text-muted opacity-0 transition-all hover:border-border-strong hover:text-fg focus:opacity-100 group-hover:opacity-100"
                  >
                    Review
                  </Link>
                ) : (
                  <span className="w-[4.25rem] shrink-0" aria-hidden />
                )}
              </li>
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}

/** Placeholder mirroring {@link ProgressCard}'s layout while progress loads. */
function ProgressCardSkeleton() {
  return (
    <div className="rounded-2xl border border-border bg-surface p-5">
      <div className="flex items-baseline justify-between">
        <div className="h-5 w-24 animate-pulse rounded bg-elevated" />
        <div className="h-4 w-10 animate-pulse rounded bg-elevated" />
      </div>
      <div className="mt-1.5 h-3 w-36 animate-pulse rounded bg-elevated" />
      <div className="mt-5 space-y-4">
        {[0, 1].map((i) => (
          <div key={i}>
            <div className="mb-1.5 flex justify-between">
              <div className="h-3 w-12 animate-pulse rounded bg-elevated" />
              <div className="h-3 w-20 animate-pulse rounded bg-elevated" />
            </div>
            <div className="h-1.5 w-full animate-pulse rounded-full bg-elevated" />
          </div>
        ))}
      </div>
    </div>
  );
}

/** Placeholder rows for the leaderboard while label stats load. */
function LeaderboardSkeleton() {
  return (
    <ol className="space-y-2.5">
      {Array.from({ length: 8 }).map((_, i) => (
        <li key={i} className="flex items-center gap-3">
          <span className="w-6 text-right font-mono text-xs text-faint">{i + 1}</span>
          <div className="h-4 w-28 shrink-0 animate-pulse rounded bg-elevated" />
          <div className="h-2 flex-1 animate-pulse rounded-full bg-elevated" />
          <div className="h-4 w-12 shrink-0 animate-pulse rounded bg-elevated" />
        </li>
      ))}
    </ol>
  );
}
