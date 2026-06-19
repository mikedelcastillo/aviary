"use client";
import type { ReactNode } from "react";
import type { Pill } from "@/lib/types";

export interface PillBarProps {
  pills: Pill[];
  onPick: (pill: Pill) => void;
  activeLabel?: string | null;
  /** Model-suggested label for the active box — its pill gets a yellow ring +
   * an Enter/Y hint, so it can be approved with one key. */
  suggestedLabel?: string | null;
  /** When provided, shows an "Unbox [B]" control that deletes the active box. */
  onUnbox?: () => void;
  /** When provided, shows a "Clear [C]" control that deletes every box. */
  onClear?: () => void;
  /** When provided, shows a "Delete [⌘⌫]" control that trashes the whole image. */
  onDelete?: () => void;
  /** Leading navigation group (touch stepper + Next), shown before the pills. */
  nav?: ReactNode;
}

/** Bottom label options for label mode. Each pill shows its keyboard shortcut. */
export function PillBar({ pills, onPick, activeLabel, suggestedLabel, onUnbox, onClear, onDelete, nav }: PillBarProps) {
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-20 flex justify-center px-4 pb-safe">
      <div className="pointer-events-auto flex max-w-[min(96vw,920px)] flex-wrap items-center justify-center gap-2 rounded-2xl border border-border bg-surface/85 px-3 py-3 backdrop-blur-md">
        {nav && (
          <>
            {nav}
            <span className="mx-1 h-6 w-px shrink-0 bg-border" aria-hidden />
          </>
        )}
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

        {(onUnbox || onClear || onDelete) && (
          <span className="mx-1 h-6 w-px shrink-0 bg-border" aria-hidden />
        )}
        {onUnbox && (
          <button
            onClick={onUnbox}
            className="flex cursor-pointer items-center gap-2 rounded-pill border border-danger/40 bg-danger/10 px-3 py-1.5 text-sm text-danger transition-colors hover:border-danger/70"
          >
            <kbd className="grid h-5 min-w-5 place-items-center rounded border border-danger/50 px-1 font-mono text-[11px] uppercase text-danger">
              B
            </kbd>
            <span className="font-medium">Unbox</span>
          </button>
        )}
        {onClear && (
          <button
            onClick={onClear}
            className="flex cursor-pointer items-center gap-2 rounded-pill border border-danger/40 bg-danger/10 px-3 py-1.5 text-sm text-danger transition-colors hover:border-danger/70"
          >
            <kbd className="grid h-5 min-w-5 place-items-center rounded border border-danger/50 px-1 font-mono text-[11px] uppercase text-danger">
              C
            </kbd>
            <span className="font-medium">Clear</span>
          </button>
        )}
        {onDelete && (
          <button
            onClick={onDelete}
            title="Delete this image (⌘/Ctrl + Backspace)"
            className="flex cursor-pointer items-center gap-2 rounded-pill border border-danger/40 bg-danger/10 px-3 py-1.5 text-sm text-danger transition-colors hover:border-danger/70"
          >
            <kbd className="grid h-5 min-w-5 place-items-center rounded border border-danger/50 px-1 font-mono text-[11px] text-danger">
              ⌘⌫
            </kbd>
            <span className="font-medium">Delete</span>
          </button>
        )}
      </div>
    </div>
  );
}
