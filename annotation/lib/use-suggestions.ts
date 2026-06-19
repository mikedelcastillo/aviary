"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { CatId, LabelSuggestion, SuggestedBox, Suggestions } from "./types";

const EMPTY: Suggestions = { boxes: [], labels: [] };

export interface UseSuggestions {
  /** Box proposals still awaiting accept/reject (yellow boxes in Box mode). */
  boxes: SuggestedBox[];
  /** Label suggestion for a human box id, if any (Label mode pre-highlight). */
  labelFor: (boxId: string) => LabelSuggestion | undefined;
  /** Accept a box proposal: drop it locally + server-side (caller adds the real box). */
  acceptBox: (id: string) => void;
  /** Reject a box proposal: drop it locally + server-side. */
  rejectBox: (id: string) => void;
  /** Dismiss a label suggestion once its box is labeled (accepted or overridden). */
  dismissLabel: (boxId: string) => void;
}

/**
 * Per-image model suggestions with optimistic accept/reject. Loads when
 * (cat,name) change; mutations update local state immediately and fire-and-forget
 * the DELETE so the sidecar shrinks. Suggestions are additive and non-critical,
 * so a failed sync is swallowed (the worst case is a stale yellow box on reload).
 */
export function useSuggestions(cat: CatId | null, name: string | null): UseSuggestions {
  const [suggestions, setSuggestions] = useState<Suggestions>(EMPTY);
  const keyRef = useRef<{ cat: CatId; name: string } | null>(null);

  useEffect(() => {
    if (!cat || !name) {
      keyRef.current = null;
      setSuggestions(EMPTY);
      return;
    }
    keyRef.current = { cat, name };
    let cancelled = false;
    setSuggestions(EMPTY);
    (async () => {
      try {
        const res = await fetch(`/api/suggestions/${cat}/${encodeURIComponent(name)}`);
        const data = (await res.json()) as Suggestions;
        if (!cancelled) setSuggestions({ boxes: data.boxes ?? [], labels: data.labels ?? [], boxModel: data.boxModel, labelModel: data.labelModel });
      } catch {
        if (!cancelled) setSuggestions(EMPTY);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [cat, name]);

  const del = useCallback((query: string) => {
    const key = keyRef.current;
    if (!key) return;
    void fetch(`/api/suggestions/${key.cat}/${encodeURIComponent(key.name)}?${query}`, {
      method: "DELETE",
      keepalive: true,
    }).catch(() => {});
  }, []);

  const dropBox = useCallback(
    (id: string) => {
      setSuggestions((s) => ({ ...s, boxes: s.boxes.filter((b) => b.id !== id) }));
      del(`box=${encodeURIComponent(id)}`);
    },
    [del],
  );

  const dismissLabel = useCallback(
    (boxId: string) => {
      setSuggestions((s) => ({ ...s, labels: s.labels.filter((l) => l.boxId !== boxId) }));
      del(`label=${encodeURIComponent(boxId)}`);
    },
    [del],
  );

  const labelFor = useCallback(
    (boxId: string) => suggestions.labels.find((l) => l.boxId === boxId),
    [suggestions.labels],
  );

  return {
    boxes: suggestions.boxes,
    labelFor,
    acceptBox: dropBox,
    rejectBox: dropBox,
    dismissLabel,
  };
}
