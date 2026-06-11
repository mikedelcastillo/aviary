import type { CategoryProgress } from "@/lib/types";

function pct(n: number, total: number): number {
  return total > 0 ? Math.round((n / total) * 100) : 0;
}

function Bar({ value, total, color }: { value: number; total: number; color: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
      <div className="h-full rounded-full transition-[width]" style={{ width: `${pct(value, total)}%`, background: color }} />
    </div>
  );
}

export function ProgressCard({ progress }: { progress: CategoryProgress }) {
  const { title, subtitle, total, boxed, labeled } = progress;
  return (
    <div className="rounded-2xl border border-border bg-surface p-5">
      <div className="flex items-baseline justify-between">
        <h3 className="text-base font-semibold text-fg">{title}</h3>
        <span className="font-mono text-sm text-muted">{total}</span>
      </div>
      <p className="mt-0.5 text-xs text-faint">{subtitle}</p>

      <div className="mt-5 space-y-4">
        <div>
          <div className="mb-1.5 flex justify-between text-xs">
            <span className="text-muted">Boxed</span>
            <span className="font-mono text-fg">
              {boxed}/{total} · {pct(boxed, total)}%
            </span>
          </div>
          <Bar value={boxed} total={total} color="#3b82f6" />
        </div>
        <div>
          <div className="mb-1.5 flex justify-between text-xs">
            <span className="text-muted">Labeled</span>
            <span className="font-mono text-fg">
              {labeled}/{boxed} · {pct(labeled, boxed)}%
            </span>
          </div>
          <Bar value={labeled} total={boxed} color="#00e676" />
        </div>
      </div>
    </div>
  );
}
