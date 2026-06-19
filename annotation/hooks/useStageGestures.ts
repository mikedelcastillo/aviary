"use client";
import { useCallback, useEffect, useMemo, useRef } from "react";
import type { PanZoom } from "./usePanZoom";
import type { Pt } from "@/lib/types";

export interface StageGestureConfig {
  pz: PanZoom;
  /** When true, a pen (Apple Pencil) or left-mouse drag draws a box. */
  drawingEnabled: boolean;
  onDrawStart?: (pt: Pt, e: React.PointerEvent) => void;
  onDrawMove?: (pt: Pt, e: React.PointerEvent) => void;
  onDrawEnd?: (pt: Pt, e: React.PointerEvent) => void;
}

export interface StageGestureHandlers {
  onPointerDown: (e: React.PointerEvent) => void;
  onPointerMove: (e: React.PointerEvent) => void;
  onPointerUp: (e: React.PointerEvent) => void;
  onPointerCancel: (e: React.PointerEvent) => void;
}

/** Pure helpers — distance and midpoint of two screen points. */
function dist(a: Pt, b: Pt): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function mid(a: Pt, b: Pt): Pt {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/**
 * All viewport input for {@link Stage}, routed by `pointerType`:
 *
 *   pen     → draw a box (when drawingEnabled)
 *   1 touch → pan
 *   2 touch → pinch-zoom + pan
 *   mouse   → left-drag draws; space/middle-drag pans; wheel zooms
 *
 * Pen and touch are mutually exclusive: whichever type starts a gesture owns it
 * until all of its pointers lift, so a resting palm (a `touch` pointer) can't
 * pan or interrupt while the Pencil (`pen`) is drawing.
 *
 * Composes {@link usePanZoom} primitives (`panBy`, `zoomAt`) — it never touches
 * the transform directly — so the pan/zoom math lives in exactly one place.
 */
export function useStageGestures(
  viewportRef: React.RefObject<HTMLElement | null>,
  config: StageGestureConfig,
): StageGestureHandlers {
  // Keep the latest config in a ref so the handlers can stay identity-stable
  // (no re-binding on every parent render).
  const cfg = useRef(config);
  cfg.current = config;

  // Which pointer kind owns the current gesture (null = idle).
  const owner = useRef<null | "pen" | "touch" | "mouse">(null);
  // Active touch pointers, in down-order, for pan/pinch.
  const touches = useRef<Map<number, Pt>>(new Map());
  // Last pinch distance + midpoint (when two touches are down).
  const pinch = useRef<{ dist: number; mid: Pt } | null>(null);
  // Last one-finger pan point (single touch).
  const touchPan = useRef<Pt | null>(null);
  // Last mouse pan point (space/middle drag).
  const mousePan = useRef<Pt | null>(null);
  // A pen/mouse draw is in progress.
  const drawing = useRef(false);
  // Space held → left-mouse drag pans instead of draws.
  const spaceHeld = useRef(false);

  // Re-derive the pan/pinch baseline from the CURRENT touch set. Called on every
  // finger-count change (down and up) so a transition — 1↔2 fingers, or 3→2 when
  // one of a pinch's fingers lifts — never leaves a stale baseline that would
  // make the next move jump. Pinch always uses the first two touches by
  // down-order (a resting palm on the no-pen views is a rare exception).
  const syncTouchBaseline = useCallback(() => {
    const pts = [...touches.current.values()];
    if (pts.length >= 2) {
      pinch.current = { dist: dist(pts[0], pts[1]), mid: mid(pts[0], pts[1]) };
      touchPan.current = null;
    } else if (pts.length === 1) {
      touchPan.current = { ...pts[0] };
      pinch.current = null;
    } else {
      touchPan.current = null;
      pinch.current = null;
    }
  }, []);

  const capture = (e: React.PointerEvent) => {
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
  };
  const release = (e: React.PointerEvent) => {
    try {
      (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
    } catch {
      /* pointer already released (e.g. cancel) — ignore */
    }
  };

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    const { pz, drawingEnabled, onDrawStart } = cfg.current;

    if (e.pointerType === "touch") {
      // Palm rejection: ignore touches while a pen/mouse gesture owns the stage.
      if (owner.current && owner.current !== "touch") return;
      owner.current = "touch";
      touches.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
      capture(e);
      e.preventDefault();
      syncTouchBaseline();
      return;
    }

    if (e.pointerType === "pen") {
      if (!drawingEnabled) return;
      // Pen priority: an Apple Pencil coming down means the user is drawing, so
      // any active touch gesture is really their resting palm — cancel it and
      // let the pen take over (without this, a palm that lands first would grab
      // the stage and lock the Pencil out). A mouse drag is never a palm, so
      // don't interrupt one.
      if (owner.current === "mouse") return;
      if (owner.current === "touch") {
        touches.current.clear();
        pinch.current = null;
        touchPan.current = null;
      }
      owner.current = "pen";
      drawing.current = true;
      capture(e);
      e.preventDefault();
      onDrawStart?.(pz.screenToImage(e.clientX, e.clientY), e);
      return;
    }

    // mouse
    if (owner.current && owner.current !== "mouse") return;
    const panning = e.button === 1 || (e.button === 0 && spaceHeld.current);
    if (panning) {
      owner.current = "mouse";
      mousePan.current = { x: e.clientX, y: e.clientY };
      capture(e);
      e.preventDefault();
      return;
    }
    if (drawingEnabled && e.button === 0) {
      owner.current = "mouse";
      drawing.current = true;
      capture(e);
      onDrawStart?.(pz.screenToImage(e.clientX, e.clientY), e);
    }
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const { pz, onDrawMove } = cfg.current;

    if (e.pointerType === "touch") {
      if (!touches.current.has(e.pointerId)) return;
      touches.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (touches.current.size >= 2 && pinch.current) {
        const [a, b] = [...touches.current.values()];
        const nd = dist(a, b);
        const nm = mid(a, b);
        if (pinch.current.dist > 0) pz.zoomAt(nd / pinch.current.dist, nm.x, nm.y);
        pz.panBy(nm.x - pinch.current.mid.x, nm.y - pinch.current.mid.y);
        pinch.current = { dist: nd, mid: nm };
      } else if (touches.current.size === 1 && touchPan.current) {
        pz.panBy(e.clientX - touchPan.current.x, e.clientY - touchPan.current.y);
        touchPan.current = { x: e.clientX, y: e.clientY };
      }
      return;
    }

    if (drawing.current) {
      onDrawMove?.(pz.screenToImage(e.clientX, e.clientY), e);
      return;
    }
    if (mousePan.current) {
      pz.panBy(e.clientX - mousePan.current.x, e.clientY - mousePan.current.y);
      mousePan.current = { x: e.clientX, y: e.clientY };
    }
  }, []);

  const endPointer = useCallback((e: React.PointerEvent) => {
    const { pz, onDrawEnd } = cfg.current;
    release(e);

    if (e.pointerType === "touch") {
      // Untracked touch (e.g. a palm that was ignored on down, or cancelled by
      // pen priority): don't let its lift mutate gesture state or clear `owner`.
      if (!touches.current.has(e.pointerId)) return;
      touches.current.delete(e.pointerId);
      // Re-baseline so dropping a finger (2→1, or 3→2 onto a different pair)
      // continues smoothly instead of jumping on the next move.
      syncTouchBaseline();
      if (touches.current.size === 0) owner.current = null;
      return;
    }

    if (drawing.current) {
      drawing.current = false;
      owner.current = null;
      onDrawEnd?.(pz.screenToImage(e.clientX, e.clientY), e);
      return;
    }
    if (mousePan.current) {
      mousePan.current = null;
      owner.current = null;
    }
  }, []);

  // --- Wheel: ctrl/⌘ => zoom anchored at cursor; else pan. -------------------
  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const { pz } = cfg.current;
      if (e.ctrlKey || e.metaKey) {
        pz.zoomAt(Math.exp(-e.deltaY * 0.01), e.clientX, e.clientY);
      } else {
        pz.panBy(-e.deltaX, -e.deltaY);
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [viewportRef]);

  // --- Space key toggles mouse left-drag between draw and pan. ---------------
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space") spaceHeld.current = true;
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") spaceHeld.current = false;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  return useMemo(
    () => ({ onPointerDown, onPointerMove, onPointerUp: endPointer, onPointerCancel: endPointer }),
    [onPointerDown, onPointerMove, endPointer],
  );
}
