"use client";
import { useEffect, useState } from "react";

/**
 * True when the device exposes a coarse (touch) pointer — i.e. an iPad/phone,
 * even with a keyboard or mouse also attached (`any-pointer`, not `pointer`).
 *
 * Drives every touch affordance: persistent box controls, on-screen Next/step
 * buttons, and hiding keyboard-shortcut hints. Starts `false` so SSR and the
 * first client render agree (a plain desktop stays untouched); flips on mount
 * and reacts if the available pointers change (e.g. a tablet docks a mouse).
 */
export function useCoarsePointer(): boolean {
  const [coarse, setCoarse] = useState(false);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia("(any-pointer: coarse)");
    const update = () => setCoarse(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  return coarse;
}
