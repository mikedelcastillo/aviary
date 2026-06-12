"use client";

import type { ReviewBox } from "@/lib/types";

interface CropCellProps {
  cell: ReviewBox;
  /** Source size of the crop image (rendered into a smaller square). */
  size?: number;
  onOpen: (cell: ReviewBox) => void;
  onUnbox: (cell: ReviewBox) => void;
  onUnlabel: (cell: ReviewBox) => void;
  /** Relabel the box as the catch-all "unknown" kind. Omitted when reviewing it. */
  onUnknown?: (cell: ReviewBox) => void;
  onDelete?: (cell: ReviewBox) => void;
}

/**
 * One grid-review cell: the box shown whole (contained, never cropped) with a
 * thin context margin and the green box drawn on it. Click the cell body to open
 * Focus Review; hover to reveal inline Unbox / Unlabel actions (those stop
 * propagation so they don't also navigate).
 */
export function CropCell({ cell, size = 160, onOpen, onUnbox, onUnlabel, onUnknown, onDelete }: CropCellProps) {
  const src =
    `/api/crop/${cell.cat}/${encodeURIComponent(cell.name)}/${encodeURIComponent(cell.box.id)}` +
    `?size=${size}`;

  return (
    <div
      className="group relative aspect-square cursor-pointer select-none overflow-hidden rounded-md border border-border bg-surface transition-colors hover:border-border-strong"
      onClick={() => onOpen(cell)}
      title={`${cell.name} — click to open in Focus`}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt=""
        loading="lazy"
        draggable={false}
        className="h-full w-full object-cover"
      />

      {/* Hover action scrim. The container itself doesn't swallow clicks (cell
          body still opens Focus); only the buttons stop propagation. */}
      <div className="absolute inset-0 flex flex-col items-stretch justify-center gap-1 bg-black/55 p-1.5 opacity-0 backdrop-blur-[1px] transition-opacity group-hover:opacity-100">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onUnbox(cell);
          }}
          className="cursor-pointer rounded border border-danger/50 bg-danger/20 py-1 text-center text-[11px] font-medium text-danger transition-colors hover:border-danger/80 hover:bg-danger/30"
        >
          Unbox
        </button>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onUnlabel(cell);
          }}
          className="cursor-pointer rounded border border-border-strong bg-white/10 py-1 text-center text-[11px] font-medium text-fg transition-colors hover:bg-white/20"
        >
          Unlabel
        </button>
        {onUnknown && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onUnknown(cell);
            }}
            className="cursor-pointer rounded border border-border-strong bg-white/10 py-1 text-center text-[11px] font-medium text-fg transition-colors hover:bg-white/20"
          >
            Unknown
          </button>
        )}
        {onDelete && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(cell);
            }}
            className="cursor-pointer rounded border border-danger/60 bg-danger/30 py-1 text-center text-[11px] font-medium text-danger transition-colors hover:border-danger hover:bg-danger/45"
          >
            Delete
          </button>
        )}
      </div>
    </div>
  );
}
