"use client";
import { useCallback, useRef, useState } from "react";
import type { NormRect, Pt, Transform } from "@/lib/types";

const MIN_SCALE = 0.02;
const MAX_SCALE = 40;

function clampScale(s: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));
}

export interface PanZoom {
  transform: Transform;
  setTransform: (t: Transform) => void;
  getTransform: () => Transform;
  /** client px -> image px (source pixels). */
  screenToImage: (clientX: number, clientY: number) => Pt;
  /** image px -> client px. */
  imageToScreen: (x: number, y: number) => Pt;
  /** Zoom by `factor` keeping the image point under (clientX,clientY) fixed. */
  zoomAt: (factor: number, clientX: number, clientY: number) => void;
  /** Pan by a screen-space delta. */
  panBy: (dx: number, dy: number) => void;
  /** Fit the whole image (contain) within the viewport. */
  fit: () => void;
  /** Center + zoom so a normalized rect fills ~`fill` of the viewport. */
  focusRect: (rect: NormRect, fill?: number) => void;
}

/**
 * Pan/zoom transform for a viewport that contains a `natural`-sized world.
 * Screen = translate(tx,ty) then scale(scale) of image px (transform-origin 0 0),
 * offset by the viewport's own client rect.
 */
export function usePanZoom(
  viewportRef: React.RefObject<HTMLElement | null>,
  natural: { width: number; height: number },
): PanZoom {
  const [transform, setTransformState] = useState<Transform>({ scale: 1, tx: 0, ty: 0 });
  const tRef = useRef(transform);
  tRef.current = transform;
  const naturalRef = useRef(natural);
  naturalRef.current = natural;

  const rect = useCallback(() => viewportRef.current?.getBoundingClientRect() ?? null, [viewportRef]);

  const setTransform = useCallback((t: Transform) => {
    const next = { scale: clampScale(t.scale), tx: t.tx, ty: t.ty };
    tRef.current = next;
    setTransformState(next);
  }, []);

  const getTransform = useCallback(() => tRef.current, []);

  const screenToImage = useCallback(
    (clientX: number, clientY: number): Pt => {
      const r = rect();
      const t = tRef.current;
      if (!r) return { x: 0, y: 0 };
      return { x: (clientX - r.left - t.tx) / t.scale, y: (clientY - r.top - t.ty) / t.scale };
    },
    [rect],
  );

  const imageToScreen = useCallback(
    (x: number, y: number): Pt => {
      const r = rect();
      const t = tRef.current;
      if (!r) return { x: 0, y: 0 };
      return { x: r.left + t.tx + x * t.scale, y: r.top + t.ty + y * t.scale };
    },
    [rect],
  );

  const zoomAt = useCallback(
    (factor: number, clientX: number, clientY: number) => {
      const r = rect();
      if (!r) return;
      const t = tRef.current;
      const newScale = clampScale(t.scale * factor);
      // Keep the image point under the cursor stationary.
      const ix = (clientX - r.left - t.tx) / t.scale;
      const iy = (clientY - r.top - t.ty) / t.scale;
      const tx = clientX - r.left - ix * newScale;
      const ty = clientY - r.top - iy * newScale;
      setTransform({ scale: newScale, tx, ty });
    },
    [rect, setTransform],
  );

  const panBy = useCallback(
    (dx: number, dy: number) => {
      const t = tRef.current;
      setTransform({ scale: t.scale, tx: t.tx + dx, ty: t.ty + dy });
    },
    [setTransform],
  );

  const fit = useCallback(() => {
    const r = rect();
    const n = naturalRef.current;
    if (!r || !n.width || !n.height) return;
    const scale = Math.min(r.width / n.width, r.height / n.height) * 0.95;
    const s = clampScale(scale);
    setTransform({ scale: s, tx: (r.width - n.width * s) / 2, ty: (r.height - n.height * s) / 2 });
  }, [rect, setTransform]);

  const focusRect = useCallback(
    (norm: NormRect, fill = 0.6) => {
      const r = rect();
      const n = naturalRef.current;
      if (!r || !n.width || !n.height) return;
      const rectPxW = Math.max(1, norm.w * n.width);
      const rectPxH = Math.max(1, norm.h * n.height);
      const scale = clampScale(Math.min((r.width * fill) / rectPxW, (r.height * fill) / rectPxH));
      const cx = (norm.x + norm.w / 2) * n.width;
      const cy = (norm.y + norm.h / 2) * n.height;
      setTransform({ scale, tx: r.width / 2 - cx * scale, ty: r.height / 2 - cy * scale });
    },
    [rect, setTransform],
  );

  return { transform, setTransform, getTransform, screenToImage, imageToScreen, zoomAt, panBy, fit, focusRect };
}
