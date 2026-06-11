import { Spinner } from "./Spinner";

/** Centered spinner overlay shown while a screen's data/image is loading. */
export function LoadingOverlay({ show, label }: { show: boolean; label?: string }) {
  return (
    <div
      className={`pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-bg/50 backdrop-blur-[1px] transition-opacity duration-200 ${
        show ? "opacity-100" : "opacity-0"
      }`}
    >
      {show && (
        <div className="flex items-center gap-3 rounded-pill border border-border bg-surface/90 px-4 py-2 text-sm text-muted">
          <Spinner size={16} className="text-fg" />
          {label && <span>{label}</span>}
        </div>
      )}
    </div>
  );
}
