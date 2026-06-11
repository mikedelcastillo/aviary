"use client";
import { create } from "zustand";
import type { Annotation, CatId } from "./types";

/**
 * Session-global undo/redo so history survives image navigation. Each record is a
 * full before/after snapshot tagged with the image it belongs to, letting Cmd+Z
 * reach back across an auto-advance: it navigates to the recorded image and stages
 * the restored snapshot via {@link HistoryState.overrides}, which `useAnnotation`
 * applies on load instead of fetching from disk.
 *
 * Cleared when returning Home so each Box/Label session has its own bounded
 * timeline (mode switches always pass through Home).
 */
export interface HistoryKey {
  cat: CatId;
  name: string;
}

export interface HistoryRecord extends HistoryKey {
  before: Annotation;
  after: Annotation;
}

function keyOf(cat: CatId, name: string): string {
  return `${cat}/${name}`;
}

const MAX = 200;

interface HistoryState {
  past: HistoryRecord[];
  future: HistoryRecord[];
  /** Restored snapshots staged for an image, keyed by `cat/name`. */
  overrides: Map<string, Annotation>;
  /** Record an applied mutation; clears the redo stack. */
  record: (r: HistoryRecord) => void;
  /** Pop the newest undo entry, staging `before` for its image. Null if empty. */
  popUndo: () => HistoryRecord | null;
  /** Pop the newest redo entry, staging `after` for its image. Null if empty. */
  popRedo: () => HistoryRecord | null;
  /** Consume a staged snapshot for an image (removing it), or null. */
  takeOverride: (cat: CatId, name: string) => Annotation | null;
  clear: () => void;
}

export const useHistoryStore = create<HistoryState>((set, get) => ({
  past: [],
  future: [],
  overrides: new Map(),

  record: (r) =>
    set((s) => {
      const past = [...s.past, r];
      if (past.length > MAX) past.shift();
      return { past, future: [] };
    }),

  popUndo: () => {
    const s = get();
    if (s.past.length === 0) return null;
    const r = s.past[s.past.length - 1];
    const overrides = new Map(s.overrides);
    overrides.set(keyOf(r.cat, r.name), r.before);
    set({ past: s.past.slice(0, -1), future: [...s.future, r], overrides });
    return r;
  },

  popRedo: () => {
    const s = get();
    if (s.future.length === 0) return null;
    const r = s.future[s.future.length - 1];
    const overrides = new Map(s.overrides);
    overrides.set(keyOf(r.cat, r.name), r.after);
    set({ future: s.future.slice(0, -1), past: [...s.past, r], overrides });
    return r;
  },

  takeOverride: (cat, name) => {
    const s = get();
    const k = keyOf(cat, name);
    const ann = s.overrides.get(k);
    if (!ann) return null;
    const overrides = new Map(s.overrides);
    overrides.delete(k);
    set({ overrides });
    return ann;
  },

  clear: () => set({ past: [], future: [], overrides: new Map() }),
}));
