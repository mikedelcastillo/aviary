"use client";
import {
  createContext,
  forwardRef,
  useContext,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { usePanZoom } from "@/hooks/usePanZoom";
import { useStageGestures } from "@/hooks/useStageGestures";
import type { Pt, StageHandle } from "@/lib/types";

interface StageCtxValue {
  scale: number;
  natural: { width: number; height: number };
  handle: StageHandle | null;
}

const StageCtx = createContext<StageCtxValue | null>(null);

/** Access live scale / natural size / handle from inside Stage children (SVG). */
export function useStage(): StageCtxValue {
  const v = useContext(StageCtx);
  if (!v) throw new Error("useStage must be used within <Stage>");
  return v;
}

export interface StageProps {
  /** Image URL. */
  src: string;
  /** SVG content rendered in image-pixel coordinate space (0..naturalW/H). */
  children?: ReactNode;
  /** When true, left-drag on the background emits draw events (box mode). */
  drawingEnabled?: boolean;
  onDrawStart?: (pt: Pt, e: React.PointerEvent) => void;
  onDrawMove?: (pt: Pt, e: React.PointerEvent) => void;
  onDrawEnd?: (pt: Pt, e: React.PointerEvent) => void;
  /** Called once the image has loaded and the handle is ready. */
  onReady?: (h: StageHandle) => void;
  className?: string;
}

/**
 * Pan/zoom image canvas. Renders <img> + an <svg> overlay inside one
 * CSS-transformed world, so vector boxes stay registered to the image. Exposes
 * a StageHandle (ref + context) for coordinate transforms and framing.
 */
export const Stage = forwardRef<StageHandle, StageProps>(function Stage(
  { src, children, drawingEnabled = false, onDrawStart, onDrawMove, onDrawEnd, onReady, className },
  ref,
) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [natural, setNatural] = useState({ width: 0, height: 0 });
  const [fitKey, setFitKey] = useState<string | null>(null);
  const pz = usePanZoom(viewportRef, natural);

  // Reset the measured size whenever the source changes. onReady fires from the
  // effect below only when natural.width/height change; without this reset, loading
  // an image with the SAME dimensions as the previous one changes no dep, so onReady
  // never fires and the parent's `ready` flag stays false (spinner spins forever —
  // seen after Ctrl+Backspace delete, which swaps the image in place).
  const prevSrc = useRef(src);
  if (prevSrc.current !== src) {
    prevSrc.current = src;
    setNatural({ width: 0, height: 0 });
    setFitKey(null);
  }

  const handle = useMemo<StageHandle>(
    () => ({
      screenToImage: pz.screenToImage,
      imageToScreen: pz.imageToScreen,
      getTransform: pz.getTransform,
      setTransform: pz.setTransform,
      getNaturalSize: () => natural,
      fit: pz.fit,
      focusRect: pz.focusRect,
    }),
    [pz, natural],
  );

  useImperativeHandle(ref, () => handle, [handle]);

  // Center (fit) the image once its natural size is known, and whenever the
  // source changes. Kept in an effect (not an onLoad rAF) so it always runs
  // AFTER `natural` is committed to state — otherwise fit() reads a stale 0x0
  // size and the image sticks to the top-left corner.
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;
  useEffect(() => {
    if (natural.width > 0 && natural.height > 0) {
      pz.fit();
      setFitKey(src);
      onReadyRef.current?.(handle);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, natural.width, natural.height]);

  // All viewport input (pointer / wheel / space) lives in one hook: pen draws,
  // one finger pans, two fingers pinch-zoom, mouse keeps its space/middle-pan +
  // left-draw + wheel-zoom behavior. Stage just renders.
  const gestures = useStageGestures(viewportRef, {
    pz,
    drawingEnabled,
    onDrawStart,
    onDrawMove,
    onDrawEnd,
  });

  const ctx = useMemo<StageCtxValue>(
    () => ({ scale: pz.transform.scale, natural, handle }),
    [pz.transform.scale, natural, handle],
  );

  return (
    <div
      ref={viewportRef}
      className={`absolute inset-0 overflow-hidden no-select ${className ?? ""}`}
      style={{ touchAction: "none", cursor: drawingEnabled ? "crosshair" : "grab" }}
      {...gestures}
    >
      <div
        className="absolute left-0 top-0 will-change-transform"
        style={{
          transform: `translate(${pz.transform.tx}px, ${pz.transform.ty}px) scale(${pz.transform.scale})`,
          transformOrigin: "0 0",
          // Hidden until the first fit for this src to avoid a top-left flash.
          opacity: fitKey === src ? 1 : 0,
          transition: "opacity 120ms ease",
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt=""
          draggable={false}
          onLoad={(e) => {
            const img = e.currentTarget;
            setNatural({ width: img.naturalWidth, height: img.naturalHeight });
          }}
          style={{
            display: "block",
            pointerEvents: "none",
            // Force exact natural size. Tailwind preflight applies
            // `max-width:100%; height:auto` to <img>, which shrinks frames wider
            // than the viewport while the SVG overlay stays full-size — that
            // de-registers every box (they drift down-right). Override it.
            width: natural.width || undefined,
            height: natural.height || undefined,
            maxWidth: "none",
          }}
        />
        {natural.width > 0 && (
          <svg
            className="absolute left-0 top-0"
            width={natural.width}
            height={natural.height}
            viewBox={`0 0 ${natural.width} ${natural.height}`}
            style={{ overflow: "visible", pointerEvents: "none" }}
          >
            <StageCtx.Provider value={ctx}>{children}</StageCtx.Provider>
          </svg>
        )}
      </div>
    </div>
  );
});
