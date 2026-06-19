"use client";

import type { ReviewBox } from "@/lib/types";

interface CropCellProps {
  cell: ReviewBox;
  /** Source size of the crop image (rendered into a smaller square). */
  size?: number;
  /** Whether this cell is part of the current selection. */
  selected: boolean;
  /** Toggle this cell's membership in the selection. */
  onToggle: (cell: ReviewBox) => void;
}

/**
 * One grid-review cell: the box shown whole (contained, never cropped) with a
 * thin context margin and the green box drawn on it. Tapping/clicking the cell
 * toggles its selection; every action (Open / Unlabel / Unknown / Unbox / Delete)
 * lives in the shared bottom bar that appears once anything is selected. No hover
 * affordances, so it behaves identically on touch and desktop.
 */
export function CropCell({ cell, size = 160, selected, onToggle }: CropCellProps) {
  const src =
    `/api/crop/${cell.cat}/${encodeURIComponent(cell.name)}/${encodeURIComponent(cell.box.id)}` +
    `?size=${size}`;

  return (
    <button
      type="button"
      onClick={() => onToggle(cell)}
      aria-pressed={selected}
      title={`${cell.name} — tap to ${selected ? "deselect" : "select"}`}
      className={`relative block aspect-square cursor-pointer touch-manipulation select-none overflow-hidden rounded-md border bg-surface transition-colors ${
        selected ? "border-box ring-2 ring-box" : "border-border hover:border-border-strong"
      }`}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt=""
        loading="lazy"
        draggable={false}
        className="h-full w-full object-cover"
      />

      {/* Green wash over the selected crop. */}
      <span
        aria-hidden
        className={`pointer-events-none absolute inset-0 transition-colors ${
          selected ? "bg-box/15" : "bg-transparent"
        }`}
      />

      {/* Selection badge: an empty circle when unselected, filled green when
          selected — visible on touch and desktop alike (no hover dependency). */}
      <span
        aria-hidden
        className={`pointer-events-none absolute left-1.5 top-1.5 grid h-5 w-5 place-items-center rounded-full border text-[11px] font-bold leading-none transition-all ${
          selected
            ? "border-box bg-box text-bg"
            : "border-white/55 bg-black/35 text-transparent"
        }`}
      >
        ✓
      </span>
    </button>
  );
}
