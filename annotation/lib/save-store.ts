"use client";
import { create } from "zustand";
import type { SaveStatus } from "./types";

interface SaveState {
  status: SaveStatus;
  setStatus: (s: SaveStatus) => void;
}

/** Save lifecycle, surfaced on-screen by <SaveIndicator> (no tab-title updates). */
export const useSaveStore = create<SaveState>((set) => ({
  status: "idle",
  setStatus: (status) => set({ status }),
}));
