"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ProgressCard } from "@/components/ProgressCard";
import { CatToggle } from "@/components/CatToggle";
import { Spinner } from "@/components/Spinner";
import {
  ALL_CATS,
  newSeed,
  parseCats,
  reviewHref,
  serializeCats,
  withNav,
  type CatId,
  type CategoryProgress,
} from "@/lib/types";

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
    (async () => {
      try {
        const res = await fetch("/api/progress");
        if (!res.ok) throw new Error(`progress: ${res.status}`);
        const data = (await res.json()) as CategoryProgress[];
        if (!cancelled) setProgress(data);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Entry points, queue, and the leaderboard are scoped to the selection.
  useEffect(() => {
    let cancelled = false;
    const param = serializeCats(cats);
    const qs = param ? `?cats=${param}` : "";
    (async () => {
      try {
        const [queueRes, entryRes, statsRes] = await Promise.all([
          fetch(`/api/queue${qs}`),
          fetch(`/api/entry${qs}`),
          fetch(`/api/label-stats${qs}`),
        ]);
        if (!queueRes.ok) throw new Error(`queue: ${queueRes.status}`);
        if (!entryRes.ok) throw new Error(`entry: ${entryRes.status}`);
        if (!statsRes.ok) throw new Error(`label-stats: ${statsRes.status}`);

        const queueData = (await queueRes.json()) as number[];
        const entryData = (await entryRes.json()) as { box: number; label: number };
        const statsData = (await statsRes.json()) as LabelStat[];

        if (!cancelled) {
          setQueue(queueData);
          setEntry(entryData);
          setLabelStats(statsData);
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    })();
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

  return (
    <main className="mx-auto max-w-5xl px-6 py-16">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-fg sm:text-3xl">
          Aviary Annotation
        </h1>
        <p className="mt-1 text-sm text-muted">
          Box and label bird detections — filesystem is the source of truth.
        </p>
        {totals && (
          <p className="mt-3 font-mono text-xs text-faint">
            {totals.total.toLocaleString()} images · {totals.boxed.toLocaleString()} boxed (
            {pct(totals.boxed, totals.total)}%) · {totals.labeled.toLocaleString()} labeled (
            {pct(totals.labeled, totals.boxed)}%)
          </p>
        )}
      </header>

      {error && (
        <div className="mt-8 rounded-xl border border-danger/40 bg-surface p-5 text-sm text-danger">
          Failed to load progress: {error}
        </div>
      )}

      {!error && !progress && (
        <div className="mt-16 flex items-center justify-center gap-3 text-sm text-muted">
          <Spinner size={18} className="text-fg" />
          <span>Loading…</span>
        </div>
      )}

      {!error && progress && (
        <>
          {/* Category selector (left) + randomize toggle (right). Scopes and
              orders everything below. */}
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
              onClick={() => router.push(withNav(`/box/${boxTarget}`, cats, random ? newSeed() : null))}
              className="group flex cursor-pointer flex-col rounded-2xl bg-fg p-6 text-left text-bg transition-opacity hover:opacity-90"
            >
              <span className="text-lg font-semibold">Box mode</span>
              <span className="mt-1 text-sm opacity-70">Draw &amp; vet bounding boxes</span>
              {totals && (
                <span className="mt-4 font-mono text-xs opacity-60">
                  {totals.boxed}/{totals.total} boxed
                </span>
              )}
            </button>

            <button
              type="button"
              onClick={() =>
                !labelEmpty &&
                router.push(withNav(`/label/${labelTarget}`, cats, random ? newSeed() : null))
              }
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
              <span className="mt-4 font-mono text-xs text-faint">
                {labelEmpty ? "Nothing to label yet" : `${queueLen} in queue`}
              </span>
            </button>
          </div>

          {/* Per-category progress — unselected categories dimmed. */}
          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {progress.map((p) => (
              <div
                key={p.id}
                className={catSet.has(p.id) ? "" : "opacity-40 transition-opacity"}
              >
                <ProgressCard progress={p} />
              </div>
            ))}
          </div>

          {/* Label leaderboard */}
          {labelStats && <LabelLeaderboard stats={labelStats} cats={cats} />}
        </>
      )}
    </main>
  );
}

function LabelLeaderboard({ stats, cats }: { stats: LabelStat[]; cats: CatId[] }) {
  const max = stats.reduce((m, s) => Math.max(m, s.count), 0);
  return (
    <section className="mt-12">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-base font-semibold text-fg">Labels</h2>
        <span className="font-mono text-xs text-faint">ranked by labeled count</span>
      </div>
      <div className="rounded-2xl border border-border bg-surface p-5">
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
      </div>
    </section>
  );
}
