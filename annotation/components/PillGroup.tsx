"use client";
import type { Pill } from "@/lib/types";

export interface PillGroupProps {
  pills: Pill[];
  onPick: (pill: Pill) => void;
  /** The active box's current label — its pill gets the green active treatment. */
  activeLabel?: string | null;
  /**
   * Model-suggested label for the active box — its pill gets a yellow ring + an
   * Enter hint, so it can be approved with one key.
   */
  suggestedLabel?: string | null;
}

/**
 * The label-mode options as keyboard-shortcut pills. Active label is green,
 * model-suggested is yellow (+ ⏎ hint). Rendered as one segment of the BottomBar.
 */
export function PillGroup({ pills, onPick, activeLabel, suggestedLabel }: PillGroupProps) {
  return (
    <>
      {pills.map((p) => {
        const active = activeLabel === p.label;
        const suggested = !active && suggestedLabel === p.label;
        return (
          <button
            key={p.label}
            onClick={() => onPick(p)}
            title={suggested ? "Model suggestion — Enter / Y to accept" : undefined}
            className={`group flex items-center gap-2 rounded-pill border px-3 py-1.5 text-sm transition-colors ${
              active
                ? "border-box bg-box/15 text-fg"
                : suggested
                  ? "border-suggest bg-suggest/15 text-fg"
                  : "border-border bg-surface-2 text-muted hover:border-border-strong hover:text-fg"
            }`}
          >
            <kbd
              className={`grid h-5 min-w-5 place-items-center rounded border px-1 font-mono text-[11px] uppercase ${
                active
                  ? "border-box/60 text-box"
                  : suggested
                    ? "border-suggest/60 text-suggest"
                    : "border-border-strong text-faint group-hover:text-muted"
              }`}
            >
              {p.shortcut}
            </kbd>
            <span className="font-medium">{p.label}</span>
            {suggested && <span className="font-mono text-[11px] text-suggest">⏎</span>}
          </button>
        );
      })}
    </>
  );
}
