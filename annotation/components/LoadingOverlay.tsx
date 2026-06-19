import { Spinner } from "./Spinner";

/** Centered spinner overlay shown while a screen's data/image is loading. When
 *  `progress` is supplied (and has a positive total), a thin determinate bar and
 *  count render under the label. */
export function LoadingOverlay({
  show,
  label,
  progress,
}: {
  show: boolean;
  label?: string;
  progress?: { done: number; total: number };
}) {
  const hasBar = !!progress && progress.total > 0;
  const pct = hasBar ? Math.round((progress!.done / progress!.total) * 100) : 0;
  return (
    <div
      className={`pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-bg/50 backdrop-blur-[1px] transition-opacity duration-200 ${
        show ? "opacity-100" : "opacity-0"
      }`}
    >
      {show && (
        <div
          className={
            "flex flex-col items-center gap-2 border border-border bg-surface/90 text-sm text-muted " +
            (hasBar ? "rounded-2xl px-5 py-3" : "rounded-pill px-4 py-2")
          }
        >
          <div className="flex items-center gap-3">
            <Spinner size={16} className="text-fg" />
            {label && <span>{label}</span>}
          </div>
          {hasBar && (
            <div className="w-56">
              <div className="h-2 overflow-hidden rounded-full bg-elevated">
                <div
                  className="h-full rounded-full bg-box transition-[width]"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="mt-1 text-center font-mono text-xs text-faint">
                {progress!.done.toLocaleString()} / {progress!.total.toLocaleString()} hashed ({pct}%)
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
