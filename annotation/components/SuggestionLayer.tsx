"use client";
import { useState } from "react";
import { useStage } from "./Stage";
import type { SuggestedBox } from "@/lib/types";

export interface SuggestionLayerProps {
  boxes: SuggestedBox[];
  /** The keyboard-selected proposal (always shows its ✓/✕ controls). */
  activeId?: string | null;
  /** Accept a proposal into the real annotation. */
  onAccept: (id: string) => void;
  /** Reject (discard) a proposal. */
  onReject: (id: string) => void;
  /**
   * Touch/Pencil devices have no hover, so render every proposal's ✓/✕
   * permanently (and with larger tap targets) instead of revealing on hover.
   */
  alwaysShowControls?: boolean;
}

const YELLOW = "#ffd54f";

/** Caption for a proposal: suggested identity (if any) + detector confidence. */
function caption(b: SuggestedBox): string {
  const pct = `${Math.round(b.conf * 100)}%`;
  return b.label ? `${b.label}? · ${pct}` : pct;
}

/**
 * Renders model box PROPOSALS (suggest_boxes output) in yellow, dashed, with
 * hover/active ✓ (accept) and ✕ (reject) controls. Lives inside the Stage SVG in
 * image-pixel space, mirroring BoxLayer's screen-constant sizing.
 */
export function SuggestionLayer({ boxes, activeId, onAccept, onReject, alwaysShowControls = false }: SuggestionLayerProps) {
  const { scale, natural } = useStage();
  const [hovered, setHovered] = useState<string | null>(null);

  const stroke = 2 / scale;
  const hit = (alwaysShowControls ? 22 : 10) / scale;
  const r = (alwaysShowControls ? 16 : 11) / scale;
  // Transparent tap pad → ~44px touch target around each ~32px glyph (0 on desktop).
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
        const px = toPx(b);
        const show = activeId === b.id || hovered === b.id || alwaysShowControls;
        return (
          <g key={b.id}>
            <rect
              x={px.x}
              y={px.y}
              width={px.w}
              height={px.h}
              fill={YELLOW}
              fillOpacity={show ? 0.12 : 0.06}
              stroke={YELLOW}
              strokeWidth={stroke}
              strokeDasharray={`${6 / scale} ${4 / scale}`}
              vectorEffect="non-scaling-stroke"
              opacity={activeId === b.id ? 1 : 0.9}
            />
            {/* Caption above the box. */}
            <text
              x={px.x}
              y={px.y - 5 / scale}
              fill={YELLOW}
              stroke="#0a0a0a"
              strokeWidth={3 / scale}
              paintOrder="stroke"
              strokeLinejoin="round"
              fontFamily="ui-monospace, monospace"
              fontSize={13 / scale}
              style={{ pointerEvents: "none" }}
            >
              {caption(b)}
            </text>

            {/* Wide invisible hit-stroke so the outline reveals the controls. */}
            <rect
              x={px.x}
              y={px.y}
              width={px.w}
              height={px.h}
              fill="none"
              stroke="transparent"
              strokeWidth={hit}
              style={{ pointerEvents: "stroke", cursor: "pointer" }}
              onPointerEnter={() => setHovered(b.id)}
              onPointerLeave={() => setHovered((h) => (h === b.id ? null : h))}
            />

            {show && (
              <>
                {/* ✓ accept — top-left corner. */}
                <g
                  transform={`translate(${px.x}, ${px.y})`}
                  style={{ pointerEvents: "auto", cursor: "pointer" }}
                  onPointerEnter={() => setHovered(b.id)}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    setHovered(null);
                    onAccept(b.id);
                  }}
                >
                  {tapR > 0 && <circle r={tapR} fill="transparent" style={{ pointerEvents: "all" }} />}
                  <circle r={r} fill="#0a0a0a" stroke={YELLOW} strokeWidth={stroke} vectorEffect="non-scaling-stroke" />
                  <polyline
                    points={`${-r * 0.45},${0} ${-r * 0.1},${r * 0.4} ${r * 0.5},${-r * 0.4}`}
                    fill="none"
                    stroke={YELLOW}
                    strokeWidth={stroke * 1.6}
                    vectorEffect="non-scaling-stroke"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </g>
                {/* ✕ reject — top-right corner, or bottom-right on touch so the
                    enlarged accept/reject pads can't overlap on narrow boxes. */}
                <g
                  transform={`translate(${px.x + px.w}, ${alwaysShowControls ? px.y + px.h : px.y})`}
                  style={{ pointerEvents: "auto", cursor: "pointer" }}
                  onPointerEnter={() => setHovered(b.id)}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    setHovered(null);
                    onReject(b.id);
                  }}
                >
                  {tapR > 0 && <circle r={tapR} fill="transparent" style={{ pointerEvents: "all" }} />}
                  <circle r={r} fill="#0a0a0a" stroke={YELLOW} strokeWidth={stroke} vectorEffect="non-scaling-stroke" />
                  <line x1={-r * 0.45} y1={-r * 0.45} x2={r * 0.45} y2={r * 0.45} stroke={YELLOW} strokeWidth={stroke * 1.4} vectorEffect="non-scaling-stroke" />
                  <line x1={-r * 0.45} y1={r * 0.45} x2={r * 0.45} y2={-r * 0.45} stroke={YELLOW} strokeWidth={stroke * 1.4} vectorEffect="non-scaling-stroke" />
                </g>
              </>
            )}
          </g>
        );
      })}
    </g>
  );
}
