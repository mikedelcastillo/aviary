"use client";
import { useEffect } from "react";
import { useCoarsePointer } from "@/hooks/useCoarsePointer";

/**
 * Mirrors {@link useCoarsePointer} onto `<html class="coarse-pointer">` so plain
 * CSS can react to touch devices (e.g. globals.css hides every <kbd> shortcut
 * badge app-wide). Renders nothing.
 */
export function PointerModeClass() {
  const coarse = useCoarsePointer();
  useEffect(() => {
    document.documentElement.classList.toggle("coarse-pointer", coarse);
  }, [coarse]);
  return null;
}
