"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  BENCH_METRICS,
  type BenchMetric,
  type BenchModel,
  type BenchScore,
  type BenchSeries,
  type BenchmarkFile,
} from "@/lib/benchmark-types";

// Stale-while-revalidate cache so a reload paints the last results instantly
// instead of flashing a skeleton (mirrors the home payload cache in page.tsx).
const CACHE_KEY = "aviary.benchmark";

function readCache(): BenchmarkFile | null | undefined {
  try {
    const raw = window.localStorage.getItem(CACHE_KEY);
    if (!raw) return undefined;
    return JSON.parse(raw) as BenchmarkFile | null;
  } catch {
    return undefined;
  }
}
function writeCache(value: BenchmarkFile | null): void {
  try {
    window.localStorage.setItem(CACHE_KEY, JSON.stringify(value));
  } catch {
    /* quota/unavailable — best-effort */
  }
}

const METRIC_LABEL: Record<BenchMetric, string> = { recall: "Recall", f1: "F1", precision: "Precision" };

// Chart geometry (viewBox units; the svg scales responsively to its container).
const W = 600;
const H = 210;
const PADL = 40;
const PADR = 16;
const PADT = 16;
const PADB = 34;
const PW = W - PADL - PADR;
const PH = H - PADT - PADB;

const xAt = (i: number, n: number): number => PADL + (n <= 1 ? PW / 2 : (i / (n - 1)) * PW);
const yAt = (frac: number): number => PADT + (1 - frac) * PH;

/**
 * Split a series into polyline point-strings broken at null gaps, so a missing
 * value (e.g. precision when tp+fp==0) leaves a visible gap instead of a straight
 * segment drawn across a version that has no data point. Runs of <2 points draw
 * no line (the dot still renders separately).
 */
function lineSegments(values: (number | null)[], n: number): string[] {
  const segments: string[] = [];
  let run: string[] = [];
  values.forEach((v, i) => {
    if (v == null) {
      if (run.length >= 2) segments.push(run.join(" "));
      run = [];
    } else {
      run.push(`${xAt(i, n)},${yAt(v)}`);
    }
  });
  if (run.length >= 2) segments.push(run.join(" "));
  return segments;
}

// Per-category line colors (overall uses the green "box" token).
const CAT_COLOR: Record<string, string> = { day: "#3b82f6", ir: "#a78bfa", phone: "#00e676" };
const OVERALL_COLOR = "#00e676";

function pctOrDash(frac: number | null | undefined): string {
  return frac == null ? "—" : String(Math.round(frac * 100));
}
function barColor(frac: number | null): string {
  if (frac == null) return "var(--color-elevated)";
  if (frac >= 0.8) return "#00e676";
  if (frac >= 0.65) return "#84cc16";
  return "#ffd54f";
}

export function BenchmarkExplorer() {
  const [file, setFile] = useState<BenchmarkFile | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const cached = readCache();
    if (cached !== undefined) {
      setFile(cached);
      setLoaded(true);
    }
    let cancelled = false;
    fetch("/api/benchmark")
      .then((res) => {
        if (!res.ok) throw new Error(`benchmark: ${res.status}`);
        return res.json();
      })
      .then((data: { benchmark: BenchmarkFile | null }) => {
        if (cancelled) return;
        setFile(data.benchmark);
        setLoaded(true);
        writeCache(data.benchmark);
      })
      .catch(() => {
        // Keep whatever the cache gave us; just stop showing the skeleton.
        if (!cancelled) setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="mt-12">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-base font-semibold text-fg">Model benchmarks</h2>
        {file && (
          <span className="font-mono text-xs text-faint">
            generated {formatWhen(file.generatedAt)}
          </span>
        )}
      </div>
      {!loaded ? (
        <BenchmarkSkeleton />
      ) : file ? (
        <Explorer file={file} />
      ) : (
        <EmptyState />
      )}
    </section>
  );
}

function EmptyState() {
  return (
    <div className="rounded-2xl border border-dashed border-border bg-surface px-5 py-10 text-center">
      <p className="text-sm font-semibold text-fg">No benchmarks yet</p>
      <p className="mt-1 text-sm text-muted">Run the benchmark to score your models.</p>
      <p className="mt-4 inline-block rounded-lg border border-border bg-bg px-3.5 py-2 font-mono text-[13px] text-box">
        scripts/benchmark.ps1 <span className="text-faint">· scripts/benchmark.sh</span>
      </p>
    </div>
  );
}

function BenchmarkSkeleton() {
  return (
    <div className="rounded-2xl border border-border bg-surface p-5">
      <div className="mb-4 flex gap-2">
        <div className="h-6 w-16 animate-pulse rounded-pill bg-elevated" />
        <div className="h-6 w-16 animate-pulse rounded-pill bg-elevated" />
      </div>
      <div className="h-40 w-full animate-pulse rounded-xl bg-elevated" />
      <div className="mt-4 space-y-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-4 w-full animate-pulse rounded bg-elevated" />
        ))}
      </div>
    </div>
  );
}

function Pill({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        "cursor-pointer rounded-pill border px-3 py-1 text-xs font-medium transition-colors " +
        (active
          ? "border-fg bg-fg text-bg"
          : "border-border bg-surface text-muted hover:border-border-strong hover:text-fg")
      }
    >
      {children}
    </button>
  );
}

function Explorer({ file }: { file: BenchmarkFile }) {
  const series = file.series.filter((s) => s.models.length > 0);
  const [seriesIdx, setSeriesIdx] = useState(0);
  const [metric, setMetric] = useState<BenchMetric>("f1");
  const [byCategory, setByCategory] = useState(false);
  const [pinned, setPinned] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const active: BenchSeries | undefined = series[seriesIdx];
  // Models ascending by version — the trend's x-axis.
  const models = useMemo(
    () => (active ? [...active.models].sort((a, b) => a.version - b.version) : []),
    [active],
  );
  const n = models.length;
  const multiCat = (active?.categories.length ?? 0) > 1;

  // Default the breakdown to the latest model whenever the series/data changes,
  // and clamp any stale hover/pin into range.
  useEffect(() => {
    setPinned(n > 0 ? n - 1 : 0);
    setHover(null);
    if (!multiCat) setByCategory(false);
  }, [seriesIdx, n, multiCat]);

  if (!active || n === 0) return <EmptyState />;

  const focused = Math.min(hover ?? pinned, n - 1);
  const model = models[focused];
  const prev = focused > 0 ? models[focused - 1] : undefined;

  const idxFromClientX = (clientX: number): number => {
    const svg = svgRef.current;
    if (!svg || n <= 1) return 0;
    const rect = svg.getBoundingClientRect();
    const vx = ((clientX - rect.left) / rect.width) * W;
    const i = Math.round(((vx - PADL) / PW) * (n - 1));
    return Math.max(0, Math.min(n - 1, i));
  };

  // The lines to plot: one "Overall" line, or one line per category.
  const lines: { key: string; color: string; values: (number | null)[] }[] = byCategory
    ? active.categories.map((cat) => ({
        key: cat,
        color: CAT_COLOR[cat] ?? OVERALL_COLOR,
        values: models.map((m) => m.byCategory[cat]?.[metric] ?? null),
      }))
    : [{ key: "Overall", color: OVERALL_COLOR, values: models.map((m) => m.overall[metric]) }];

  // Per-label breakdown rows for the focused model, sorted by the metric desc
  // (nulls last), with the delta vs the previous version.
  const rows = Object.entries(model.labels)
    .map(([label, score]) => {
      const val = score[metric];
      const prevVal = prev?.labels[label]?.[metric] ?? null;
      const delta = val != null && prevVal != null ? val - prevVal : null;
      return { label, score, val, delta, isNew: focused === 0 };
    })
    .sort((a, b) => (b.val ?? -1) - (a.val ?? -1));

  const overallDelta =
    prev && model.overall[metric] != null && prev.overall[metric] != null
      ? (model.overall[metric] as number) - (prev.overall[metric] as number)
      : null;

  return (
    <div className="rounded-2xl border border-border bg-surface p-5">
      {/* Controls */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {series.length > 1 ? (
          series.map((s, i) => (
            <Pill key={s.name} active={i === seriesIdx} onClick={() => setSeriesIdx(i)}>
              {capitalize(s.name)}
            </Pill>
          ))
        ) : (
          <span className="rounded-pill border border-border px-3 py-1 text-xs font-medium text-fg">
            {capitalize(active.name)}
          </span>
        )}
        <span className="ml-1 text-xs text-faint">{active.categories.join(" + ")}</span>

        <div className="ml-auto flex items-center gap-2">
          {multiCat && (
            <>
              <Pill active={!byCategory} onClick={() => setByCategory(false)}>
                Overall
              </Pill>
              <Pill active={byCategory} onClick={() => setByCategory(true)}>
                By category
              </Pill>
              <span className="mx-1 h-4 w-px bg-border" aria-hidden />
            </>
          )}
          {BENCH_METRICS.map((m) => (
            <Pill key={m} active={m === metric} onClick={() => setMetric(m)}>
              {METRIC_LABEL[m]}
            </Pill>
          ))}
        </div>
      </div>

      {/* Trend chart with labeled axes */}
      <div className="flex gap-2">
        <div className="flex flex-col justify-between py-[14px] pb-[34px] text-right font-mono text-[9px] text-faint">
          <span>100</span>
          <span>50</span>
          <span>0</span>
        </div>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full touch-none select-none"
          role="img"
          aria-label={`${capitalize(active.name)} ${METRIC_LABEL[metric]} across model versions`}
        >
          {/* gridlines */}
          {[0, 0.25, 0.5, 0.75, 1].map((g) => (
            <line key={g} x1={PADL} y1={yAt(g)} x2={W - PADR} y2={yAt(g)} stroke="var(--color-border)" strokeWidth={1} />
          ))}

          {/* crosshair on the focused version */}
          {n > 0 && (
            <line
              x1={xAt(focused, n)}
              y1={PADT}
              x2={xAt(focused, n)}
              y2={PADT + PH}
              stroke="var(--color-border-strong)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
          )}

          {/* lines + dots */}
          {lines.map((line) => (
            <g key={line.key}>
              {n > 1 &&
                lineSegments(line.values, n).map((seg, si) => (
                  <polyline key={si} fill="none" stroke={line.color} strokeWidth={2.5} points={seg} />
                ))}
              {line.values.map((v, i) =>
                v == null ? null : (
                  <circle key={i} cx={xAt(i, n)} cy={yAt(v)} r={i === focused ? 5.5 : 3.5} fill={line.color} />
                ),
              )}
            </g>
          ))}

          {/* value label on the focused point (overall view only) */}
          {!byCategory && model.overall[metric] != null && (
            <text
              x={xAt(focused, n)}
              y={yAt(model.overall[metric] as number) - 9}
              fill="var(--color-fg)"
              fontSize={10}
              textAnchor="middle"
              className="font-mono"
            >
              {pctOrDash(model.overall[metric])}
            </text>
          )}

          {/* x-axis version labels */}
          {models.map((m, i) => (
            <text key={m.name} x={xAt(i, n)} y={H - 12} fill="var(--color-muted)" fontSize={10} textAnchor="middle" className="font-mono">
              {m.name}
            </text>
          ))}

          {/* invisible scrub/tap overlay (pointer = mouse + touch) */}
          <rect
            x={PADL}
            y={PADT}
            width={PW}
            height={PH}
            fill="transparent"
            style={{ cursor: "crosshair" }}
            onPointerMove={(e) => setHover(idxFromClientX(e.clientX))}
            onPointerDown={(e) => {
              const i = idxFromClientX(e.clientX);
              setPinned(i);
              setHover(i);
            }}
            onPointerUp={(e) => {
              if (e.pointerType !== "mouse") setHover(null);
            }}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
      </div>

      {/* legend (category view) */}
      {byCategory && (
        <div className="mt-1 flex flex-wrap gap-3 pl-[34px] font-mono text-[11px] text-muted">
          {lines.map((line) => (
            <span key={line.key} className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: line.color }} />
              {line.key}
            </span>
          ))}
        </div>
      )}

      {/* Breakdown for the focused model */}
      <div className="mt-4 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-fg">{model.name}</h3>
        <span className="font-mono text-xs text-muted">
          overall {pctOrDash(model.overall[metric])} <DeltaChip delta={overallDelta} isNew={focused === 0} />
        </span>
      </div>

      <div className="mt-2 space-y-1.5">
        {rows.map((r) => (
          <div key={r.label} className="grid grid-cols-[5.5rem_1fr_2rem_2.75rem] items-center gap-2 text-xs">
            <span className="truncate text-fg">{r.label}</span>
            <span className="h-2.5 w-full overflow-hidden rounded-full bg-elevated">
              <span
                className="block h-full rounded-full transition-[width]"
                style={{ width: `${r.val == null ? 0 : Math.round(r.val * 100)}%`, background: barColor(r.val) }}
              />
            </span>
            <span className="text-right font-mono text-muted">{pctOrDash(r.val)}</span>
            <span className="text-right font-mono">
              {r.isNew ? <span className="text-faint">new</span> : <DeltaChip delta={r.delta} isNew={false} />}
            </span>
          </div>
        ))}
      </div>

      <p className="mt-4 font-mono text-[11px] text-faint">
        {n === 1
          ? "Run a newer model version to see the trend. "
          : "Scrub the chart (or tap a point) to break down any version. "}
        {model.images.toLocaleString()} labeled imgs · {model.gtBoxes.toLocaleString()} boxes
        {model.naBoxes > 0 && ` · ${model.naBoxes.toLocaleString()} out-of-vocab`} · conf{" "}
        {file.config.confidence} · IoU {file.config.iou}
      </p>
    </div>
  );
}

function DeltaChip({ delta, isNew }: { delta: number | null; isNew: boolean }) {
  if (isNew) return <span className="text-faint">new</span>;
  if (delta == null) return <span className="text-faint">·</span>;
  const pts = Math.round(delta * 100);
  if (pts > 0) return <span className="text-box">▲ +{pts}</span>;
  if (pts < 0) return <span className="text-danger">▼ {Math.abs(pts)}</span>;
  return <span className="text-faint">· 0</span>;
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function formatWhen(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
