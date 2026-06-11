"use client";

import type { CatId } from "@/lib/types";
import { CATEGORIES } from "@/lib/types";

interface CatToggleProps {
  /** Currently selected category ids. */
  selected: CatId[];
  /** Per-category image totals, keyed by category id (for the count badge). */
  totals: Record<CatId, number> | null;
  /** Called with the next selection whenever a chip is toggled. */
  onChange: (next: CatId[]) => void;
}

function fmt(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`;
  return String(n);
}

/**
 * A row of toggle chips — one per category — that scopes which sets the user is
 * annotating. At least one chip stays selected at all times (you cannot clear
 * the whole set). Selected chips are filled; unselected are outlined.
 */
export function CatToggle({ selected, totals, onChange }: CatToggleProps) {
  const set = new Set(selected);

  function toggle(id: CatId): void {
    const next = new Set(set);
    if (next.has(id)) {
      if (next.size === 1) return; // never allow the empty set
      next.delete(id);
    } else {
      next.add(id);
    }
    onChange(CATEGORIES.filter((c) => next.has(c.id)).map((c) => c.id));
  }

  return (
    <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Categories to annotate">
      {CATEGORIES.map((c) => {
        const on = set.has(c.id);
        const count = totals?.[c.id];
        return (
          <button
            key={c.id}
            type="button"
            role="checkbox"
            aria-checked={on}
            onClick={() => toggle(c.id)}
            className={
              "flex cursor-pointer items-center gap-2 rounded-pill border px-3.5 py-1.5 text-sm transition-colors " +
              (on
                ? "border-fg bg-fg text-bg"
                : "border-border bg-surface text-muted hover:border-border-strong hover:text-fg")
            }
          >
            <span
              aria-hidden
              className={
                "flex h-3.5 w-3.5 items-center justify-center rounded-[4px] border text-[9px] leading-none " +
                (on ? "border-bg/40 bg-bg/20 text-bg" : "border-border-strong text-transparent")
              }
            >
              ✓
            </span>
            <span className="font-medium">{c.title}</span>
            {count != null && (
              <span className={"font-mono text-xs " + (on ? "text-bg/60" : "text-faint")}>{fmt(count)}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
