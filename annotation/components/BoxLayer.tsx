"use client";
import { useState } from "react";
import { useStage } from "./Stage";
import type { Box, NormRect } from "@/lib/types";

export interface BoxLayerProps {
  boxes: Box[];
  /** A box currently being drawn (normalized), rendered dashed. */
  draft?: NormRect | null;
  /** Show the hover delete affordance (box mode). */
  showDelete?: boolean;
  onDelete?: (id: string) => void;
  /**
   * When provided (with showDelete), only boxes matching the predicate show the
   * hover ✕. Label mode uses this to expose ✕ on labeled boxes (✕ removes the
   * label rather than the box).
   */
  deletableFor?: (box: Box) => boolean;
  /** Optionally highlight one box (e.g. the active box in label mode). */
  activeId?: string | null;
  /**
   * Touch/Pencil devices have no hover, so render every box's ✕ permanently
   * (and with larger tap targets) instead of revealing it on hover.
   */
  alwaysShowControls?: boolean;
}

const GREEN = "#00e676";

/** Renders bounding boxes inside the Stage SVG (image-pixel coordinate space). */
export function BoxLayer({ boxes, draft, showDelete = false, onDelete, deletableFor, activeId, alwaysShowControls = false }: BoxLayerProps) {
  const { scale, natural } = useStage();
  const [hovered, setHovered] = useState<string | null>(null);

  // Screen-constant sizes expressed in image units. Touch devices get a wider
  // tap band and a larger ✕ glyph (no mouse precision).
  const stroke = 2 / scale;
  const hit = (alwaysShowControls ? 22 : 10) / scale;
  const xR = (alwaysShowControls ? 16 : 11) / scale;
  // Transparent tap pad so the touch hit area is ~44px even though the glyph is
  // ~32px (Apple's 44pt minimum). 0 on desktop (hover + precise cursor).
  const tapR = alwaysShowControls ? 22 / scale : 0;

  const toPx = (b: { cx: number; cy: number; w: number; h: number }) => ({
    x: (b.cx - b.w / 2) * natural.width,
    y: (b.cy - b.h / 2) * natural.height,
    w: b.w * natural.width,
    h: b.h * natural.height,
  });

  return (
    <g>
      {boxes.map((b) => {
        const r = toPx(b);
        const active = activeId === b.id;
        const isHover = hovered === b.id;
        const canDelete = showDelete && (deletableFor ? deletableFor(b) : true);
        return (
          <g key={b.id}>
            <rect
              x={r.x}
              y={r.y}
              width={r.w}
              height={r.h}
              fill="none"
              stroke={GREEN}
              strokeWidth={stroke}
              vectorEffect="non-scaling-stroke"
              opacity={active ? 1 : 0.95}
            />
            {/* Label caption, left-aligned just above the box. */}
            {b.label && (
              <text
                x={r.x}
                y={r.y - 5 / scale}
                fill={GREEN}
                stroke="#0a0a0a"
                strokeWidth={3 / scale}
                paintOrder="stroke"
                strokeLinejoin="round"
                fontFamily="ui-monospace, monospace"
                fontSize={13 / scale}
                style={{ pointerEvents: "none" }}
              >
                {b.label}
              </text>
            )}
            {/* Wide invisible hit-stroke so only the outline is interactive. */}
            {canDelete && (
              <rect
                x={r.x}
                y={r.y}
                width={r.w}
                height={r.h}
                fill="none"
                stroke="transparent"
                strokeWidth={hit}
                style={{ pointerEvents: "stroke", cursor: "pointer" }}
                onPointerEnter={() => setHovered(b.id)}
                onPointerLeave={() => setHovered((h) => (h === b.id ? null : h))}
              />
            )}
            {canDelete && (isHover || alwaysShowControls) && (
              <g
                transform={`translate(${r.x + r.w}, ${r.y})`}
                style={{ pointerEvents: "auto", cursor: "pointer" }}
                onPointerEnter={() => setHovered(b.id)}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  e.preventDefault();
                  onDelete?.(b.id);
                  setHovered(null);
                }}
              >
                {tapR > 0 && <circle r={tapR} fill="transparent" style={{ pointerEvents: "all" }} />}
                <circle r={xR} fill="#0a0a0a" stroke={GREEN} strokeWidth={stroke} vectorEffect="non-scaling-stroke" />
                <line x1={-xR * 0.45} y1={-xR * 0.45} x2={xR * 0.45} y2={xR * 0.45} stroke={GREEN} strokeWidth={stroke * 1.4} vectorEffect="non-scaling-stroke" />
                <line x1={-xR * 0.45} y1={xR * 0.45} x2={xR * 0.45} y2={-xR * 0.45} stroke={GREEN} strokeWidth={stroke * 1.4} vectorEffect="non-scaling-stroke" />
              </g>
            )}
          </g>
        );
      })}

      {draft && draft.w > 0 && draft.h > 0 && (
        <rect
          x={draft.x * natural.width}
          y={draft.y * natural.height}
          width={draft.w * natural.width}
          height={draft.h * natural.height}
          fill={GREEN}
          fillOpacity={0.12}
          stroke={GREEN}
          strokeWidth={stroke}
          strokeDasharray={`${6 / scale} ${4 / scale}`}
          vectorEffect="non-scaling-stroke"
        />
      )}
    </g>
  );
}
