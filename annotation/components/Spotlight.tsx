"use client";
import { useId } from "react";
import { useStage } from "./Stage";
import type { NormRect } from "@/lib/types";

export interface SpotlightProps {
  /** The box to keep fully visible (normalized). Null dims nothing. */
  rect: NormRect | null;
  /** Dim opacity outside the rect (0..1). */
  dim?: number;
}

const GREEN = "#00e676";

/**
 * Darkens everything outside `rect` and leaves the inside fully transparent —
 * the label-mode "spotlight". Rendered inside the Stage SVG (image-px space).
 */
export function Spotlight({ rect, dim = 0.62 }: SpotlightProps) {
  const { natural, scale } = useStage();
  const maskId = useId();
  if (!rect) return null;

  const x = rect.x * natural.width;
  const y = rect.y * natural.height;
  const w = rect.w * natural.width;
  const h = rect.h * natural.height;

  // Pad the cover well beyond the image so panning never reveals an undimmed edge.
  const pad = Math.max(natural.width, natural.height) * 2;

  return (
    <g style={{ pointerEvents: "none" }}>
      <defs>
        <mask id={maskId}>
          <rect x={-pad} y={-pad} width={natural.width + pad * 2} height={natural.height + pad * 2} fill="white" />
          <rect x={x} y={y} width={w} height={h} fill="black" />
        </mask>
      </defs>
      <rect
        x={-pad}
        y={-pad}
        width={natural.width + pad * 2}
        height={natural.height + pad * 2}
        fill="#000"
        opacity={dim}
        mask={`url(#${maskId})`}
      />
      <rect x={x} y={y} width={w} height={h} fill="none" stroke={GREEN} strokeWidth={2 / scale} vectorEffect="non-scaling-stroke" />
    </g>
  );
}
