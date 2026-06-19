"use client";
import type { ReactNode } from "react";

/** Visual tier for a bottom-bar action. */
export type ActionVariant = "neutral" | "danger" | "suggest";

export interface ActionButtonProps {
  /** Button text (and aria-label for icon-only buttons), e.g. "Clear". */
  label: string;
  /** Keyboard shortcut shown in the kbd badge, e.g. "C" or "⌘⌫". Omit for icon-only. */
  shortcut?: string;
  onClick: () => void;
  /**
   * neutral — safe action (default); danger — destructive/irreversible (red);
   * suggest — suggestion-related (yellow).
   */
  variant?: ActionVariant;
  disabled?: boolean;
  title?: string;
  /**
   * When set, renders an icon-only square button (no kbd badge, no text). `label`
   * becomes the aria-label. Used for Undo/Redo.
   */
  icon?: ReactNode;
}

const BTN: Record<ActionVariant, string> = {
  neutral:
    "border-border bg-surface-2 text-muted hover:border-border-strong hover:text-fg",
  danger: "border-danger/40 bg-danger/10 text-danger hover:border-danger/70",
  suggest: "border-suggest/50 bg-suggest/10 text-suggest hover:border-suggest",
};

const KBD: Record<ActionVariant, string> = {
  neutral: "border-border-strong text-faint",
  danger: "border-danger/50 text-danger",
  suggest: "border-suggest/60 text-suggest",
};

/**
 * The single button primitive for the bottom bar. One shape/size across every
 * mode; the `variant` carries the only meaningful difference (safe vs
 * destructive vs suggestion). Renders a kbd shortcut badge + label.
 */
export function ActionButton({ label, shortcut, onClick, variant = "neutral", disabled, title, icon }: ActionButtonProps) {
  if (icon) {
    return (
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        aria-label={label}
        title={title ?? label}
        className={`grid h-8 w-8 cursor-pointer place-items-center rounded-pill border transition-colors disabled:pointer-events-none disabled:opacity-40 ${BTN[variant]}`}
      >
        {icon}
      </button>
    );
  }
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`flex cursor-pointer items-center gap-2 rounded-pill border px-3 py-1.5 text-sm transition-colors disabled:pointer-events-none disabled:opacity-40 ${BTN[variant]}`}
    >
      {shortcut && (
        <kbd
          className={`grid h-5 min-w-5 place-items-center rounded border px-1 font-mono text-[11px] uppercase ${KBD[variant]}`}
        >
          {shortcut}
        </kbd>
      )}
      <span className="font-medium">{label}</span>
    </button>
  );
}
