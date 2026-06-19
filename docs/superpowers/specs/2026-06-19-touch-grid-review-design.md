# Touch-first grid review — design

**Date:** 2026-06-19
**Area:** `annotation/` — grid review (`/review/grid`), the camera-roll sibling of Focus review.

## Problem

Grid review surfaces every box carrying a label as a wall of crops. Each `CropCell`
hides its actions — **Unbox / Unlabel / Unknown / Delete** — behind a `group-hover`
scrim. Hover never fires on a touch device, so on iPad the grid is *open-in-Focus
only*: you can scroll and tap into Focus, but none of the four actions are reachable.
The grid is also the natural place to triage in bulk, which per-cell hover buttons
can't express even on desktop.

## Goal

1. **Grid becomes the default review view** (Home → Review lands on the grid).
2. **Replace hover with a selection model** that works identically on touch and
   desktop, lets you scroll freely, and acts on one *or many* crops at once.
   No functionality is removed — every action survives, now reachable on touch.

## Decisions (confirmed via interactive mockups)

1. **Tap selects.** Tapping/clicking a cell toggles its selection (✓ badge + ring).
   The hover scrim and all four inline buttons are removed. Same behavior on mouse
   and touch — one code path, no more grid styling island.
2. **One shared action bar.** When ≥1 cell is selected, the unified `BottomBar`
   appears and acts on the whole selection. Segments:
   `[ {n} selected · Clear ] │ [ Open · Unlabel(L) · Unknown(U) · Unbox(B) ] │ [ Delete(⌘⌫) ]`.
   - **Open** (neutral) shows only when exactly one cell is selected → opens it in Focus.
   - **Unlabel / Unknown** neutral (Unknown hidden when already reviewing `unknown_bird`).
   - **Unbox** danger. **Delete** danger, its own trailing segment (image-delete is a
     different class of action — matches the unified-bottom-bar rule).
3. **Open in Focus = "Open" in the bar when one is selected.** No per-cell chrome;
   biggest tap target; cleanest scroll.
4. **Full bulk delete.** Delete acts on the entire selection, **deduped by image**
   (`cat`+`name`) — one tap can trash several images. The existing trash tree +
   restore endpoint make it fully undoable.
5. **Select all in the header.** A header toggle selects/clears all visible cells
   (Select-all must live outside the bar, which is hidden when nothing is selected).
   **Clear** lives in the bar.

## Architecture

### `components/CropCell.tsx` — stripped to presentational
Props shrink to `{ cell, size?, selected, onToggle }`. Renders the crop + a
selection ring (`box` green) + a ✓ badge when selected. A `<button>` with
`aria-pressed`; tap toggles. No action buttons, no `onOpen/onUnbox/...`.

### `lib/use-image-batch-delete.ts` — new
`useImageDelete` is single-image (a `busy` guard rejects overlapping calls;
`pending` holds one image). The grid needs to trash several images under **one**
undo toast, so this hook generalizes it: `remove(images[], onDeleted)` and
`undo(onRestored)` over the same `/api/delete` + `/api/delete/restore` endpoints,
deleting **sequentially** (avoids server-side races on shared manifest state) and
tracking `pending: PendingDelete[]`. `useImageDelete` is untouched for
Focus/Label/Box modes.

### `app/review/grid/page.tsx` — selection + bulk
- New state `selected: Set<cellKey>`; `selectedCells` derived against `visible`
  (stale keys can't act).
- **Undo batching:** `UndoEntry` becomes `{ items: { cell, inverse }[] }`. A bulk
  action pushes **one** grouped entry; one ⌘Z / header-Undo reverses the whole
  batch (today's single-op path is just a batch of one). Box-ops run **sequentially**
  because several selected crops may share one image sidecar (concurrent writes
  would race). Partial failures re-show only the failed cells and re-stack only
  the failed items.
- **Bulk delete** dedupes the selection by image and calls the batch hook; the
  `DeleteToast` shows `"{n} images"` (or the single name) and restores all on undo.
- **Keyboard** (desktop): with a selection, `L`/`U`/`B` and `⌘⌫` act on it; `Esc`
  clears the selection (else Home); `⌘A` selects all visible; `⌘Z` undo stays.
- Selection clears on data refetch and after every action; the grid gets bottom
  padding while the bar is visible so the last row isn't obscured.

### `app/page.tsx` — default
Leaderboard **Review** link → `gridReviewHref(label, cats)` (was `reviewHref`).
The Focus⇄Grid toggle stays on both pages, so Focus is one tap away.

## Out of scope
Focus/Label/Box modes and their keyboard-shortcut logic. Server/API unchanged.

## Verification
No test runner in `annotation/`. Verify via `npx tsc --noEmit`, `npm run lint`,
`npm run build`, an adversarial multi-agent review of the diff, and a manual run
across touch (selection, bulk Unbox/Unlabel/Unknown, Open, bulk Delete + undo)
and desktop (click-select, keyboard shortcuts).
