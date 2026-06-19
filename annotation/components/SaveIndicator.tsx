"use client";
import { useSaveStore } from "@/lib/save-store";
import { Spinner } from "./Spinner";

/** Floating bottom-right chip that reflects autosave state (replaces tab-title). */
export function SaveIndicator() {
  const status = useSaveStore((s) => s.status);
  if (status === "idle" || status === "dirty") return null;

  const saving = status === "saving";
  return (
    <div className="pointer-events-none fixed bottom-safe-sm right-5 z-50">
      <div
        className={`flex items-center gap-2 rounded-pill border px-3 py-1.5 text-xs backdrop-blur-md transition-colors ${
          saving ? "border-border bg-surface/90 text-muted" : "border-box/40 bg-surface/90 text-box"
        }`}
      >
        {saving ? (
          <Spinner size={13} className="text-fg" />
        ) : (
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M5 12.5l4 4 10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
        <span>{saving ? "Saving…" : "Saved"}</span>
      </div>
    </div>
  );
}
