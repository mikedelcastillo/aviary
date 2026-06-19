"use client";
import { Fragment, type ReactNode } from "react";

export interface BottomBarProps {
  /**
   * Ordered bar segments (nav, pills, action groups). Falsy entries are dropped,
   * and a vertical divider is drawn between each surviving segment — so a mode
   * only passes the segments it has, in order, and grouping/dividers fall out.
   */
  segments: Array<ReactNode | null | false | undefined>;
}

/**
 * The one bottom-bar shell for every mode: a single centered, blurred rounded
 * island fixed to the bottom. Label/Box/Review each compose it from a list of
 * segments (NavCluster, PillGroup, ActionButton groups); this draws the island
 * and the dividers so there's never more than one floating bar.
 */
export function BottomBar({ segments }: BottomBarProps) {
  const visible = segments.filter(Boolean) as ReactNode[];
  if (visible.length === 0) return null;
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-20 flex justify-center px-4 pb-safe">
      <div className="pointer-events-auto flex max-w-[min(96vw,920px)] flex-wrap items-center justify-center gap-2 rounded-2xl border border-border bg-surface/85 px-3 py-3 backdrop-blur-md">
        {visible.map((seg, i) => (
          <Fragment key={i}>
            {i > 0 && <span className="mx-1 h-6 w-px shrink-0 bg-border" aria-hidden />}
            {seg}
          </Fragment>
        ))}
      </div>
    </div>
  );
}
