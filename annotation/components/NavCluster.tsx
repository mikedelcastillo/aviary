"use client";

export interface NavClusterProps {
  /** Zero-based position within the sequence. */
  pos: number;
  total: number;
  /** Step one image back / forward (mirrors ← / →). */
  onPrev: () => void;
  onNext: () => void;
  prevDisabled?: boolean;
  nextDisabled?: boolean;
  /**
   * Primary "Next →" — skip to the next image still needing work (mirrors
   * Space). Only rendered on touch (`coarse`), where there's no keyboard.
   */
  onAdvance?: () => void;
  /** Touch device (no keyboard) — show the primary Next button. */
  coarse?: boolean;
}

/**
 * Bottom-bar navigation: a `‹ 12 / 340 ›` prev/next stepper plus, on touch, a
 * primary `Next →` that advances to the next image needing work. Presentational
 * — the host positions it (e.g. inside an existing bottom action bar).
 */
export function NavCluster({
  pos,
  total,
  onPrev,
  onNext,
  prevDisabled,
  nextDisabled,
  onAdvance,
  coarse = false,
}: NavClusterProps) {
  // Same 32px footprint as the bar's icon buttons (ActionButton h-8) so the
  // stepper reads as a peer of the other controls, on desktop and touch alike.
  const stepBtn =
    "grid h-8 w-8 place-items-center rounded-full text-lg leading-none text-muted transition-colors hover:bg-elevated hover:text-fg disabled:pointer-events-none disabled:opacity-30";

  return (
    <div className="flex items-center gap-1">
      <div className="flex items-center gap-1">
        <button type="button" onClick={onPrev} disabled={prevDisabled} aria-label="Previous image" className={stepBtn}>
          ‹
        </button>
        <span className="px-2 font-mono tabular-nums text-xs text-fg">
          {pos + 1} / {total}
        </span>
        <button type="button" onClick={onNext} disabled={nextDisabled} aria-label="Next image" className={stepBtn}>
          ›
        </button>
      </div>

      {coarse && onAdvance && (
        <button
          type="button"
          onClick={onAdvance}
          aria-label="Skip to next image needing work"
          className="flex items-center gap-1.5 rounded-pill bg-fg px-3 py-1.5 text-sm font-semibold text-bg transition-opacity hover:opacity-90"
        >
          Next <span aria-hidden>→</span>
        </button>
      )}
    </div>
  );
}
