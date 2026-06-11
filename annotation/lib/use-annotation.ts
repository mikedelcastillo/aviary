"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { Annotation, Box, CatId } from "./types";
import { useSaveStore } from "./save-store";
import { useHistoryStore, type HistoryKey } from "./history-store";

let boxCounter = 0;
function newId(): string {
  boxCounter += 1;
  return `b${Date.now().toString(36)}_${boxCounter}`;
}

const DEBOUNCE_MS = 600;
const SAVED_LINGER_MS = 1000;

function clone(a: Annotation): Annotation {
  return { boxed: a.boxed, boxes: a.boxes.map((b) => ({ ...b })) };
}

async function putAnnotation(cat: CatId, name: string, ann: Annotation, keepalive = false): Promise<void> {
  await fetch(`/api/annotation/${cat}/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(ann),
    keepalive,
  });
}

export interface UseAnnotation {
  annotation: Annotation | null;
  loading: boolean;
  addBox: (box: Omit<Box, "id">) => string;
  updateBox: (id: string, patch: Partial<Omit<Box, "id">>) => void;
  removeBox: (id: string) => void;
  setLabel: (id: string, label: string | null) => void;
  setBoxed: (boxed: boolean) => void;
  replaceBoxes: (boxes: Box[]) => void;
  /**
   * Undo the newest action in the session-global timeline. Returns the image the
   * action belongs to when it differs from the one currently shown — the caller
   * must navigate there (the restored snapshot is already staged and applied on
   * load). Returns null when handled in place or when there's nothing to undo.
   */
  undo: () => HistoryKey | null;
  redo: () => HistoryKey | null;
  canUndo: boolean;
  canRedo: boolean;
  saveNow: () => void;
}

/**
 * Per-image annotation state with undo/redo + debounced autosave. Loads when
 * (cat,name) change; flushes unsaved edits for the previous image before
 * switching. Save status is published to the global save store (tab title).
 */
export function useAnnotation(cat: CatId | null, name: string | null): UseAnnotation {
  const [annotation, setAnnotation] = useState<Annotation | null>(null);
  const [loading, setLoading] = useState(false);
  const [version, setVersion] = useState(0); // bump to recompute canUndo/canRedo

  const annRef = useRef<Annotation | null>(null);
  const dirtyRef = useRef(false);
  const keyRef = useRef<{ cat: CatId; name: string } | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lingerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const saveSeqRef = useRef(0);

  const setStatus = useSaveStore((s) => s.setStatus);

  const commit = useCallback(
    (next: Annotation) => {
      annRef.current = next;
      setAnnotation(next);
    },
    [],
  );

  const doSave = useCallback(async () => {
    const key = keyRef.current;
    const ann = annRef.current;
    if (!key || !ann || !dirtyRef.current) return;
    const seq = ++saveSeqRef.current;
    dirtyRef.current = false;
    setStatus("saving");
    try {
      await putAnnotation(key.cat, key.name, clone(ann));
    } catch {
      dirtyRef.current = true; // allow retry on next edit
    }
    if (seq !== saveSeqRef.current) return; // a newer save superseded us
    if (dirtyRef.current) return; // edited again while saving
    setStatus("saved");
    if (lingerRef.current) clearTimeout(lingerRef.current);
    lingerRef.current = setTimeout(() => setStatus("idle"), SAVED_LINGER_MS);
  }, [setStatus]);

  const scheduleSave = useCallback(() => {
    dirtyRef.current = true;
    setStatus("dirty");
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(doSave, DEBOUNCE_MS);
  }, [doSave, setStatus]);

  const saveNow = useCallback(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    void doSave();
  }, [doSave]);

  /** Apply a mutation: record a before/after snapshot for undo, schedule save. */
  const mutate = useCallback(
    (fn: (a: Annotation) => Annotation) => {
      const cur = annRef.current;
      const key = keyRef.current;
      if (!cur || !key) return;
      const next = fn(clone(cur));
      useHistoryStore.getState().record({
        cat: key.cat,
        name: key.name,
        before: clone(cur),
        after: clone(next),
      });
      commit(next);
      scheduleSave();
      setVersion((v) => v + 1);
    },
    [commit, scheduleSave],
  );

  /** Apply (and persist) a snapshot staged for an image by undo/redo. */
  const applyStaged = useCallback(
    (cat: CatId, name: string) => {
      const ann = useHistoryStore.getState().takeOverride(cat, name);
      if (!ann) return false;
      commit(clone(ann));
      dirtyRef.current = true;
      scheduleSave();
      setVersion((v) => v + 1);
      return true;
    },
    [commit, scheduleSave],
  );

  // --- Load on (cat,name) change; flush previous image first. -----------------
  useEffect(() => {
    // Flush unsaved edits for the outgoing image synchronously-ish.
    const prevKey = keyRef.current;
    const prevAnn = annRef.current;
    if (prevKey && prevAnn && dirtyRef.current) {
      void putAnnotation(prevKey.cat, prevKey.name, clone(prevAnn), true);
      dirtyRef.current = false;
    }
    if (debounceRef.current) clearTimeout(debounceRef.current);

    if (!cat || !name) {
      keyRef.current = null;
      commit({ boxed: false, boxes: [] });
      return;
    }

    keyRef.current = { cat, name };
    dirtyRef.current = false;

    // Cross-image undo/redo: if a restored snapshot was staged for this image
    // (we navigated here to undo an action that happened on it), apply that
    // instead of fetching, and persist it.
    const staged = useHistoryStore.getState().takeOverride(cat, name);
    if (staged) {
      commit(clone(staged));
      dirtyRef.current = true;
      scheduleSave();
      setLoading(false);
      setVersion((v) => v + 1);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setStatus("idle");
    (async () => {
      try {
        const res = await fetch(`/api/annotation/${cat}/${encodeURIComponent(name)}`);
        const data = (await res.json()) as Annotation;
        if (cancelled) return;
        commit(data);
      } catch {
        if (!cancelled) commit({ boxed: false, boxes: [] });
      } finally {
        if (!cancelled) {
          setLoading(false);
          setVersion((v) => v + 1);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [cat, name, commit, setStatus, scheduleSave]);

  // --- Flush on tab close. ----------------------------------------------------
  useEffect(() => {
    const handler = () => {
      const key = keyRef.current;
      const ann = annRef.current;
      if (key && ann && dirtyRef.current) {
        void putAnnotation(key.cat, key.name, clone(ann), true);
      }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, []);

  // --- Public mutations -------------------------------------------------------
  const addBox = useCallback(
    (box: Omit<Box, "id">) => {
      const id = newId();
      mutate((a) => ({ ...a, boxes: [...a.boxes, { ...box, id }] }));
      return id;
    },
    [mutate],
  );

  const updateBox = useCallback(
    (id: string, patch: Partial<Omit<Box, "id">>) => {
      mutate((a) => ({ ...a, boxes: a.boxes.map((b) => (b.id === id ? { ...b, ...patch } : b)) }));
    },
    [mutate],
  );

  const removeBox = useCallback(
    (id: string) => mutate((a) => ({ ...a, boxes: a.boxes.filter((b) => b.id !== id) })),
    [mutate],
  );

  const setLabel = useCallback(
    (id: string, label: string | null) =>
      mutate((a) => ({ ...a, boxes: a.boxes.map((b) => (b.id === id ? { ...b, label } : b)) })),
    [mutate],
  );

  const setBoxed = useCallback((boxed: boolean) => mutate((a) => ({ ...a, boxed })), [mutate]);

  const replaceBoxes = useCallback((boxes: Box[]) => mutate((a) => ({ ...a, boxes })), [mutate]);

  const undo = useCallback((): HistoryKey | null => {
    const r = useHistoryStore.getState().popUndo();
    if (!r) return null;
    const key = keyRef.current;
    if (key && key.cat === r.cat && key.name === r.name) {
      applyStaged(r.cat, r.name); // same image — restore in place
      return null;
    }
    return { cat: r.cat, name: r.name }; // caller navigates; staged snapshot applies on load
  }, [applyStaged]);

  const redo = useCallback((): HistoryKey | null => {
    const r = useHistoryStore.getState().popRedo();
    if (!r) return null;
    const key = keyRef.current;
    if (key && key.cat === r.cat && key.name === r.name) {
      applyStaged(r.cat, r.name);
      return null;
    }
    return { cat: r.cat, name: r.name };
  }, [applyStaged]);

  void version;
  const canUndo = useHistoryStore((s) => s.past.length > 0);
  const canRedo = useHistoryStore((s) => s.future.length > 0);

  return {
    annotation,
    loading,
    addBox,
    updateBox,
    removeBox,
    setLabel,
    setBoxed,
    replaceBoxes,
    undo,
    redo,
    canUndo,
    canRedo,
    saveNow,
  };
}
