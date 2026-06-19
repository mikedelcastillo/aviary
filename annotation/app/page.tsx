"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ProgressCard } from "@/components/ProgressCard";
import { CatToggle } from "@/components/CatToggle";
import { BenchmarkExplorer } from "@/components/BenchmarkExplorer";
import {
  ALL_CATS,
  CATEGORIES,
  filterByCats,
  newSeed,
  orderBySeed,
  gridReviewHref,
  parseCats,
  serializeCats,
  withCats,
  withNav,
  type CatId,
  type CategoryProgress,
  type ManifestEntry,
} from "@/lib/types";
import { useHistoryStore } from "@/lib/history-store";
import { useCoarsePointer } from "@/hooks/useCoarsePointer";

const CATS_STORAGE_KEY = "aviary.cats";
const RANDOM_STORAGE_KEY = "aviary.random";

interface LabelStat {
  label: string;
  count: number;
}

/** Mirror of the server's `/api/home` payload (lib/annotation-io.ts HomeData). */
interface HomeData {
  progress: CategoryProgress[];
  queue: number[];
  boxQueue: number[];
  entry: { box: number; label: number };
  labelStats: LabelStat[];
}

// Client-side stale-while-revalidate cache for the home payload, keyed per
// category selection and persisted across reloads. Lets a reload paint the last
// results instantly (in a "refreshing" state) instead of flashing skeletons.
const HOME_CACHE_KEY = "aviary.home";

function readHomeCache(catsKey: string): HomeData | null {
  try {
    const raw = window.localStorage.getItem(HOME_CACHE_KEY);
    if (!raw) return null;
    return (JSON.parse(raw) as Record<string, HomeData>)[catsKey] ?? null;
  } catch {
    return null;
  }
}

function writeHomeCache(catsKey: string, data: HomeData): void {
  try {
    const raw = window.localStorage.getItem(HOME_CACHE_KEY);
    const all = raw ? (JSON.parse(raw) as Record<string, HomeData>) : {};
    all[catsKey] = data;
    window.localStorage.setItem(HOME_CACHE_KEY, JSON.stringify(all));
  } catch {
    /* quota/unavailable — caching is best-effort */
  }
}

function pct(n: number, total: number): number {
  return total > 0 ? Math.round((n / total) * 100) : 0;
}

interface Totals {
  total: number;
  boxed: number;
  labeled: number;
  suggested: number;
  suggestedBoxes: number;
}

function sumTotals(progress: CategoryProgress[]): Totals {
  return progress.reduce<Totals>(
    (acc, p) => ({
      total: acc.total + p.total,
      boxed: acc.boxed + p.boxed,
      labeled: acc.labeled + p.labeled,
      suggested: acc.suggested + p.suggested,
      suggestedBoxes: acc.suggestedBoxes + p.suggestedBoxes,
    }),
    { total: 0, boxed: 0, labeled: 0, suggested: 0, suggestedBoxes: 0 },
  );
}

export default function Home() {
  const router = useRouter();
  const [progress, setProgress] = useState<CategoryProgress[] | null>(null);
  // Full global manifest — lets the entry buttons compute the FIRST to-do image
  // in the same seeded order the Box/Label pages walk, so entry lands exactly
  // where the page would (no deep-link-guard flash). Static per session.
  const [manifest, setManifest] = useState<ManifestEntry[] | null>(null);
  const [queue, setQueue] = useState<number[] | null>(null);
  const [boxQueue, setBoxQueue] = useState<number[] | null>(null);
  const [entry, setEntry] = useState<{ box: number; label: number } | null>(null);
  const [labelStats, setLabelStats] = useState<LabelStat[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cats, setCats] = useState<CatId[]>(ALL_CATS);
  const [random, setRandom] = useState(false);
  const [dedupeProgress, setDedupeProgress] = useState<{
    done: number;
    total: number;
    running: boolean;
  } | null>(null);
  // Drives the reload button's spin: stays true through the reload navigation so
  // the icon keeps spinning until the fresh page paints over it.
  const [reloading, setReloading] = useState(false);
  // True when we're showing cached (stale) home data while a fresh fetch is in flight.
  const [revalidating, setRevalidating] = useState(false);

  const reloadPage = () => {
    setReloading(true);
    window.location.reload();
  };

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

  // Full manifest is selection-independent (ignores cats) — fetch once.
  useEffect(() => {
    let cancelled = false;
    fetch("/api/manifest")
      .then((res) => {
        if (!res.ok) throw new Error(`manifest: ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setManifest(data as ManifestEntry[]);
      })
      .catch((e) => {
        if (!cancelled) setError((e as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Progress (global) + the selection-scoped queue/entry/leaderboard arrive from
  // ONE server-side pass via /api/home. Stale-while-revalidate: if we have a
  // cached payload for this selection (persisted across reloads), paint it
  // immediately and flag "revalidating" while the fresh fetch runs — no skeleton
  // flash. Only on a true cache miss do the scoped modules fall back to skeletons
  // (progress is global, so it's left in place to avoid flashing the cards).
  useEffect(() => {
    let cancelled = false;
    const param = serializeCats(cats);
    const qs = param ? `?cats=${param}` : "";
    const catsKey = param ?? "all";

    const cached = readHomeCache(catsKey);
    if (cached) {
      setProgress(cached.progress);
      setQueue(cached.queue);
      setBoxQueue(cached.boxQueue);
      setEntry(cached.entry);
      setLabelStats(cached.labelStats);
      setRevalidating(true);
    } else {
      setQueue(null);
      setBoxQueue(null);
      setEntry(null);
      setLabelStats(null);
      setRevalidating(false);
    }

    fetch(`/api/home${qs}`)
      .then((res) => {
        if (!res.ok) throw new Error(`home: ${res.status}`);
        return res.json();
      })
      .then((data: HomeData) => {
        if (cancelled) return;
        setProgress(data.progress);
        setQueue(data.queue);
        setBoxQueue(data.boxQueue);
        setEntry(data.entry);
        setLabelStats(data.labelStats);
        writeHomeCache(catsKey, data);
        setRevalidating(false);
      })
      .catch((e) => {
        if (!cancelled) {
          setError((e as Error).message);
          setRevalidating(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [cats]);

  // Warm the dedupe hash index for the selected categories and poll coverage so
  // the Dedupe card shows how much has been traversed. The GET both warms (side
  // effect) and reports {done,total,running}. Stop once fully indexed and idle.
  //
  // The first poll is DEFERRED: warming decodes every image through sharp, which
  // pegs CPU and the libuv thread pool. Firing it at mount would collide with the
  // homepage's own data fetches (notably /api/label-stats, which reads thousands
  // of sidecars) and stall them. Letting the page load first, then warming in the
  // background, keeps both healthy. Warm is idempotent so the small delay is free.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const param = serializeCats(cats);
    const qs = param ? `?cats=${param}` : "";
    const poll = async () => {
      try {
        const res = await fetch(`/api/dedupe/progress${qs}`);
        if (res.ok) {
          const snap = (await res.json()) as {
            done: number;
            total: number;
            running: boolean;
          };
          if (!cancelled) {
            setDedupeProgress(snap);
            if (snap.total > 0 && snap.done >= snap.total && !snap.running) return;
          }
        }
      } catch {
        /* transient — keep polling */
      }
      if (!cancelled) timer = setTimeout(poll, 2000);
    };
    timer = setTimeout(poll, 2500);
    return () => {
      cancelled = true;
      clearTimeout(timer);
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

  // Enter a mode. Land on the FIRST image still needing work in the seeded order
  // — computed exactly as the Box/Label page does (shuffle the full in-scope
  // manifest with the same seed, then pick the first to-do), so entry matches the
  // page's sequence and the deep-link guard never has to snap. Falls back to the
  // server entry point until the manifest lands.
  const filteredHome = useMemo(
    () => (manifest ? filterByCats(manifest, cats) : []),
    [manifest, cats],
  );
  const enterBox = () => {
    const seed = random ? newSeed() : null;
    const orderedHome = orderBySeed(filteredHome, seed);
    const set = new Set(boxQueue ?? []);
    const first = orderedHome.find((e) => set.has(e.n)) ?? orderedHome[0];
    const target = first?.n ?? boxTarget;
    router.push(withNav(`/box/${target}`, cats, seed));
  };
  const enterLabel = () => {
    if (labelEmpty) return;
    const seed = random ? newSeed() : null;
    const orderedHome = orderBySeed(filteredHome, seed);
    const set = new Set(queue ?? []);
    const first = orderedHome.find((e) => set.has(e.n)) ?? orderedHome[0];
    const target = first?.n ?? labelTarget;
    router.push(withNav(`/label/${target}`, cats, seed));
  };

  return (
    <main className="mx-auto max-w-5xl px-6 py-16">
      <header>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-fg sm:text-3xl">
              Aviary Annotation
            </h1>
            <p className="mt-1 text-sm text-muted">
              Box and label bird detections — filesystem is the source of truth.
            </p>
          </div>
          <button
            type="button"
            onClick={reloadPage}
            disabled={reloading}
            aria-label="Reload page"
            title="Reload page"
            className={
              "group flex h-11 w-11 shrink-0 cursor-pointer touch-manipulation select-none items-center " +
              "justify-center rounded-pill border border-border bg-surface text-muted outline-none " +
              "transition-[transform,background-color,border-color,color] duration-150 " +
              "hover:border-border-strong hover:bg-surface-2 hover:text-fg " +
              "focus-visible:border-border-strong focus-visible:text-fg " +
              "active:scale-90 active:bg-elevated disabled:cursor-default disabled:active:scale-100"
            }
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
              className={
                "h-5 w-5 transition-transform duration-300 ease-out " +
                (reloading ? "animate-spin" : "group-hover:rotate-180 group-active:rotate-90")
              }
            >
              <path d="M21 12a9 9 0 1 1-2.64-6.36" />
              <path d="M21 3v6h-6" />
            </svg>
          </button>
        </div>
        {totals ? (
          <p className="mt-3 font-mono text-xs text-faint">
            {totals.total.toLocaleString()} images · {totals.boxed.toLocaleString()} boxed (
            {pct(totals.boxed, totals.total)}%) · {totals.labeled.toLocaleString()} labeled (
            {pct(totals.labeled, totals.boxed)}%)
            {totals.suggestedBoxes > 0 && (
              <>
                {" "}·{" "}
                <span className="text-suggest">
                  {totals.suggested.toLocaleString()} suggested ({pct(totals.suggested, totals.total)}%)
                </span>
              </>
            )}
            {revalidating && <span className="ml-2 animate-pulse text-faint">· refreshing…</span>}
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
        className="mt-3 flex flex-col gap-2 rounded-xl border border-border bg-surface px-5 py-3 text-sm transition-colors hover:border-border-strong hover:bg-surface-2"
      >
        <div className="flex items-center justify-between">
          <span className="font-medium text-fg">Dedupe</span>
          <span className="text-muted">Find &amp; remove near-duplicate frames →</span>
        </div>
        {dedupeProgress && dedupeProgress.total > 0 && (
          <div>
            <div className="h-1.5 overflow-hidden rounded-full bg-elevated">
              <div
                className="h-full rounded-full bg-box transition-[width]"
                style={{
                  width: `${Math.round((dedupeProgress.done / dedupeProgress.total) * 100)}%`,
                }}
              />
            </div>
            <div className="mt-1 font-mono text-[11px] text-faint">
              {dedupeProgress.done.toLocaleString()} / {dedupeProgress.total.toLocaleString()} indexed
              ({Math.round((dedupeProgress.done / dedupeProgress.total) * 100)}%)
              {dedupeProgress.running && " · indexing…"}
            </div>
          </div>
        )}
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

      {/* Model benchmarks — explore how each model version scores per label.
          Self-contained: fetches /api/benchmark and renders its own empty state.
          Sits at the bottom of the page. */}
      <BenchmarkExplorer />
    </main>
  );
}

function LabelLeaderboard({ stats, cats }: { stats: LabelStat[] | null; cats: CatId[] }) {
  const max = stats ? stats.reduce((m, s) => Math.max(m, s.count), 0) : 0;
  // Touch has no hover, so the per-label Review link can't rely on a hover
  // reveal — show it permanently on coarse pointers.
  const coarse = useCoarsePointer();
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
                    href={gridReviewHref(s.label, cats)}
                    className={`shrink-0 rounded-pill border border-border px-2.5 py-1 text-xs text-muted transition-all hover:border-border-strong hover:text-fg ${
                      coarse ? "opacity-100" : "opacity-0 focus:opacity-100 group-hover:opacity-100"
                    }`}
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
