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
 * Batch sibling of {@link useImageDelete} for grid review, where one action can
 * trash several images at once. `remove` moves each image (+ sidecars) to the
 * trash tree (POST /api/delete) **sequentially** — concurrent moves could race on
 * shared manifest state — and, on success, runs `onDeleted` with the images that
 * actually landed before surfacing a single undo toast; `undo` restores them all
 * (POST /api/delete/restore). The files sit safely in /data/annotation/deleted
 * regardless, so the toast is convenience, not the only path back. A `busy` guard
 * prevents overlapping delete/undo runs.
 */
export function useImageBatchDelete() {
  const [pending, setPending] = useState<PendingDelete[]>([]);
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
    setPending([]);
  }, [clearTimer]);

  const remove = useCallback(
    async (
      images: PendingDelete[],
      onDeleted: (deleted: PendingDelete[]) => void | Promise<void>,
    ): Promise<boolean> => {
      if (busy.current || images.length === 0) return false;
      busy.current = true;
      const setStatus = useSaveStore.getState().setStatus;
      setStatus("saving");
      try {
        const deleted: PendingDelete[] = [];
        for (const img of images) {
          try {
            const res = await fetch("/api/delete", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(img),
            });
            if (res.ok) deleted.push(img);
          } catch {
            // Skip this image; others may still succeed.
          }
        }
        if (deleted.length === 0) {
          setStatus("idle");
          return false;
        }
        // Hide AFTER the moves land so the caller reflects the new manifest.
        await onDeleted(deleted);
        setStatus("saved");
        setPending(deleted);
        clearTimer();
        timer.current = setTimeout(() => setPending([]), TOAST_MS);
        return true;
      } finally {
        busy.current = false;
      }
    },
    [clearTimer],
  );

  const undo = useCallback(
    async (
      onRestored: (restored: PendingDelete[]) => void | Promise<void>,
    ): Promise<boolean> => {
      if (busy.current || pending.length === 0) return false;
      busy.current = true;
      const imgs = pending;
      const setStatus = useSaveStore.getState().setStatus;
      setStatus("saving");
      try {
        const restored: PendingDelete[] = [];
        for (const img of imgs) {
          try {
            const res = await fetch("/api/delete/restore", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(img),
            });
            if (res.ok) restored.push(img);
          } catch {
            // Skip; the file remains in trash and can be restored manually.
          }
        }
        await onRestored(restored);
        setStatus("saved");
        clearTimer();
        setPending([]);
        return true;
      } finally {
        busy.current = false;
      }
    },
    [pending, clearTimer],
  );

  return { pending, remove, undo, dismiss };
}
