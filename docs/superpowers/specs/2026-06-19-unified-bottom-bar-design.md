# Unified bottom bar — design

**Date:** 2026-06-19
**Area:** `annotation/` (Next.js annotation tool) — label, box, and review focus modes.

## Problem

Successive sessions evolved the bottom "pill bar" independently in each mode, producing several detached islands of buttons and three divergent implementations:

- **Label mode** — already one island via the shared `PillBar` component: `nav · pills · Unbox · Clear · Delete`.
- **Review mode** — also one island, but **hand-rolled separately** from `PillBar`: `nav · Unlabel · Unknown · Unbox · Delete`.
- **Box mode** — the worst offender: **2–3 detached islands** floating side by side — a `Clear` card, a separate `Delete` card, and the `NavCluster` stepper (plus a `Next →` button on touch).

The result is visual fragmentation and style drift, with no single source of truth for "the bottom bar".

## Goal

One shared component set renders the bottom bar in all three modes as a **single rounded island**, with one consistent button vocabulary. Retain all functionality; merge where it simplifies. Plus one new behavior: box-mode **Clear** also rejects on-screen model suggestions.

## Decisions (all confirmed via interactive mockups)

1. **Button style — two-tier.** One consistent button system (same shape/size/`kbd` badge). Three semantic variants:
   - `neutral` — safe actions (Unlabel, Unknown).
   - `danger` — destructive / irreversible (Unbox, Clear, Delete) — red tint as a warning.
   - `suggest` — suggestion-related (the model-suggested label pill in label mode; Accept-all in box mode) — yellow.
   - The active-label pill keeps its green `active` treatment.

2. **Navigation placement — stepper always in the bar.** The `‹ pos/total ›` stepper is the bar's left segment in **every** mode, on desktop and touch. Its `‹ ›` arrows are clickable; the touch `Next →` primary still appears on coarse pointers. The **top-right position counter is removed** from label & review (now shown in the bar). Box mode keeps its top-right `boxed ✓ / unboxed` **status** (not a position counter).

3. **Image-level `Delete` is always its own trailing segment** (after a divider) in all modes — it deletes the whole image, a different class of action than box/label edits. This aligns label mode (which today groups Delete with Unbox/Clear) with review mode.

4. **Box-mode `Clear` (C) also rejects every on-screen suggestion.** In addition to wiping placed boxes and reverting the `boxed` flag, Clear now rejects all yellow model proposals. Clear is shown whenever there are placed boxes **or** suggestions (today: only when boxes exist). Single-suggestion reject (`N` / per-box ✕) is unchanged.

5. **`Accept all` (A) surfaced as a button.** The existing keyboard-only accept-all becomes a `suggest`-styled button in box mode, rendered only while suggestions are present, paired with Clear (accept-all vs reject-all).

## Architecture

### New / changed components (`annotation/components/`)

- **`ActionButton.tsx`** (new) — the single button primitive.
  - Props: `label: string`, `shortcut: string`, `onClick: () => void`, `variant?: 'neutral' | 'danger' | 'suggest'` (default `neutral`), `disabled?: boolean`, `title?: string`.
  - Renders the `kbd` shortcut badge + label with variant-appropriate colors, reusing the exact Tailwind tokens already in the codebase (`border-danger/40 bg-danger/10 text-danger`, `border-suggest bg-suggest/15`, `border-border bg-surface-2 text-muted`, `rounded-pill`, etc.).

- **`BottomBar.tsx`** (new) — the island shell.
  - Renders the fixed, centered, blurred `rounded-2xl border border-border bg-surface/85` container (the existing wrapper markup from `PillBar`), positioned `fixed inset-x-0 bottom-0 z-20 … pb-safe`.
  - Prop: `segments: Array<ReactNode | null | false>`. Filters falsy segments and renders a `<span className="… w-px … bg-border">` divider between each surviving segment. Pages pass `null` for an empty/absent segment.

- **`PillGroup.tsx`** (new) — label-mode pills, extracted verbatim from today's `PillBar` (active = green, model-suggested = yellow + ⏎ hint). Props: `pills, onPick, activeLabel, suggestedLabel`.

- **`NavCluster.tsx`** (modified) — drop its own `rounded-full border bg-surface/85` chrome so it reads as a borderless **segment** inside the island (it's no longer free-floating). The coarse `Next →` primary stays filled. Behavior/props unchanged otherwise.

- **`PillBar.tsx`** (deleted) — replaced by `BottomBar` + `PillGroup` + `ActionButton`.

### Page rewiring

Each page composes a `BottomBar` with an explicit `segments` array. Keyboard-shortcut **logic is unchanged** — only the rendering moves into the shared components.

- **`app/label/[n]/page.tsx`**
  - Segments: `NavCluster` (always; gated on `total > 0`) · `PillGroup` (when pills exist) · `[Unbox(B) when activeBox, Clear(C) when boxes>0]` · `[Delete(⌘⌫) when name]`.
  - Remove the top-right `pos+1 / total` counter.

- **`app/box/[n]/page.tsx`**
  - Replace the three islands with one `BottomBar`.
  - Segments: `NavCluster` (always) · `[Accept all(A) when suggBoxes>0, Clear(C) when boxes>0 || suggBoxes>0]` · `[Delete(⌘⌫)]`.
  - `clearBoxes` extended: if placed boxes exist → `replaceBoxes([])` + revert `boxed`; additionally reject every current suggestion (`for (const s of suggBoxes) rejectBox(s.id)`); early-return only when there are neither boxes nor suggestions.
  - Keep the top-right `boxed ✓ / unboxed` status and the top-center suggestion hint.

- **`app/review/[n]/page.tsx`**
  - Replace the hand-rolled island with `BottomBar`.
  - Segments: `NavCluster` (always; `total > 0`) · `[Unlabel(L), Unknown(U) when canUnknown, Unbox(B)]` (disabled when `!hasMatch`) · `[Delete(⌘⌫)]`.
  - Remove the top-right `pos+1 / total` counter.

## Out of scope

Grid review (`/review/grid`), the home page, and all keyboard-shortcut *logic*. No functionality is removed.

## Verification

No test runner exists in `annotation/` (Next.js app). Verify via: `npx tsc --noEmit` (typecheck), `npm run lint`, `npm run build`, an adversarial multi-agent code review of the diff, and a manual run of the dev server across the three modes (placing boxes, accepting/rejecting/clearing suggestions, navigating, deleting).
