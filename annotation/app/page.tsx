"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ProgressCard } from "@/components/ProgressCard";
import { Spinner } from "@/components/Spinner";
import type { CategoryProgress } from "@/lib/types";

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

  useEffect(() => {
    let cancelled = false;

    async function load(): Promise<void> {
      try {
        const [progressRes, queueRes, entryRes, statsRes] = await Promise.all([
          fetch("/api/progress"),
          fetch("/api/queue"),
          fetch("/api/entry"),
          fetch("/api/label-stats"),
        ]);
        if (!progressRes.ok) throw new Error(`progress: ${progressRes.status}`);
        if (!queueRes.ok) throw new Error(`queue: ${queueRes.status}`);
        if (!entryRes.ok) throw new Error(`entry: ${entryRes.status}`);
        if (!statsRes.ok) throw new Error(`label-stats: ${statsRes.status}`);

        const progressData = (await progressRes.json()) as CategoryProgress[];
        const queueData = (await queueRes.json()) as number[];
        const entryData = (await entryRes.json()) as { box: number; label: number };
        const statsData = (await statsRes.json()) as LabelStat[];

        if (!cancelled) {
          setProgress(progressData);
          setQueue(queueData);
          setEntry(entryData);
          setLabelStats(statsData);
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const totals = progress ? sumTotals(progress) : null;
  const queueLen = queue?.length ?? 0;
  // Mode buttons jump to the first unboxed / first unlabeled image.
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
            {pct(totals.labeled, totals.total)}%)
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
          {/* Primary entry points */}
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            <button
              type="button"
              onClick={() => router.push(`/box/${boxTarget}`)}
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
              onClick={() => router.push(`/label/${labelTarget}`)}
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

          {/* Per-category progress */}
          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {progress.map((p) => (
              <ProgressCard key={p.id} progress={p} />
            ))}
          </div>

          {/* Label leaderboard */}
          {labelStats && <LabelLeaderboard stats={labelStats} />}
        </>
      )}
    </main>
  );
}

function LabelLeaderboard({ stats }: { stats: LabelStat[] }) {
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
            <li key={s.label} className="flex items-center gap-3">
              <span className="w-6 text-right font-mono text-xs text-faint">{i + 1}</span>
              <span className="w-28 shrink-0 truncate text-sm text-fg">{s.label}</span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-elevated">
                <div
                  className="h-full rounded-full bg-box transition-[width]"
                  style={{ width: max > 0 ? `${(s.count / max) * 100}%` : "0%" }}
                />
              </div>
              <span className="w-12 shrink-0 text-right font-mono text-sm text-muted">{s.count}</span>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
