"use client";

interface DeleteToastProps {
  /** Basename of the just-deleted image, shown for context. */
  name: string;
  onUndo: () => void;
  onDismiss: () => void;
}

/**
 * Transient bottom-left toast offering one-click undo after a manual delete.
 * Sits clear of the bottom-center nav HUD / action bars and the bottom-right
 * <SaveIndicator>. Rendered by a view only while a delete is pending.
 */
export function DeleteToast({ name, onUndo, onDismiss }: DeleteToastProps) {
  return (
    <div className="fixed bottom-6 left-4 z-40 flex items-center gap-3 rounded-2xl border border-border bg-surface/90 px-3.5 py-2.5 text-sm shadow-lg backdrop-blur-md">
      <span className="flex items-center gap-2 text-muted">
        <svg
          width="15"
          height="15"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="shrink-0 text-danger"
          aria-hidden
        >
          <path d="M3 6h18" />
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
        </svg>
        <span>Deleted</span>
        <span className="inline-block max-w-[14rem] truncate font-mono text-xs text-faint" title={name}>
          {name}
        </span>
      </span>
      <button
        type="button"
        onClick={onUndo}
        className="flex cursor-pointer items-center gap-1.5 rounded-pill border border-border-strong bg-surface-2 px-3 py-1 text-sm font-medium text-fg transition-colors hover:border-fg/40"
      >
        Undo
      </button>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss"
        className="flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded-full text-faint transition-colors hover:bg-elevated hover:text-fg"
      >
        ×
      </button>
    </div>
  );
}
