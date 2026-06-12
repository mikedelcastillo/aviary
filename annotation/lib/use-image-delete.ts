"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSaveStore } from "./save-store";
import type { CatId } from "./types";

export interface PendingDelete {
  cat: CatId;
  name: string;
}

/** How long the undo toast lingers before auto-dismissing (ms). */
const TOAST_MS = 6000;

/**
 * Manual image-delete with a transient undo affordance, shared by the Box/Label/
 * Review views. `remove` moves the image (+ sidecars) to the trash tree
 * (POST /api/delete) and, on success, runs the caller's `onDeleted` (refetch +
 * navigate) before surfacing the toast; `undo` restores it (POST
 * /api/delete/restore) and runs `onRestored`. The files sit safely in
 * /data/annotation/deleted regardless, so the toast is convenience, not the
 * only path back. A `busy` guard prevents overlapping delete/undo requests.
 */
export function useImageDelete() {
  const [pending, setPending] = useState<PendingDelete | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const busy = useRef(false);

  const clearTimer = useCallback(() => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  // Drop the toast timer if the view unmounts mid-countdown.
  useEffect(() => () => clearTimer(), [clearTimer]);

  const dismiss = useCallback(() => {
    clearTimer();
    setPending(null);
  }, [clearTimer]);

  const remove = useCallback(
    async (
      cat: CatId,
      name: string,
      onDeleted: (p: PendingDelete) => void | Promise<void>,
    ): Promise<boolean> => {
      if (busy.current) return false;
      busy.current = true;
      const setStatus = useSaveStore.getState().setStatus;
      setStatus("saving");
      try {
        const res = await fetch("/api/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cat, name }),
        });
        if (!res.ok) throw new Error(String(res.status));
        const p: PendingDelete = { cat, name };
        // Navigate AFTER the move lands so the refetch reflects the new manifest.
        await onDeleted(p);
        setStatus("saved");
        setPending(p);
        clearTimer();
        timer.current = setTimeout(() => setPending(null), TOAST_MS);
        return true;
      } catch {
        setStatus("idle");
        return false;
      } finally {
        busy.current = false;
      }
    },
    [clearTimer],
  );

  const undo = useCallback(
    async (onRestored: (p: PendingDelete) => void | Promise<void>): Promise<boolean> => {
      if (busy.current || !pending) return false;
      busy.current = true;
      const p = pending;
      const setStatus = useSaveStore.getState().setStatus;
      setStatus("saving");
      try {
        const res = await fetch("/api/delete/restore", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cat: p.cat, name: p.name }),
        });
        if (!res.ok) throw new Error(String(res.status));
        await onRestored(p);
        setStatus("saved");
        clearTimer();
        setPending(null);
        return true;
      } catch {
        setStatus("idle");
        return false;
      } finally {
        busy.current = false;
      }
    },
    [pending, clearTimer],
  );

  return { pending, remove, undo, dismiss };
}
