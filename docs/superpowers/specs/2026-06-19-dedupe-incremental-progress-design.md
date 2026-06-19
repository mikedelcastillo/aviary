# Dedupe incremental hashing progress — design

**Date:** 2026-06-19
**Status:** Approved (pending spec review)
**Area:** `annotation/` (Next.js annotation tool)

## Problem

With tens of thousands of photos, opening the dedupe page "takes forever to load"
with no feedback. `GET /api/dedupe` runs the entire job synchronously before
returning anything: `getDedupeClusters` → `clusterCategory` → `ensureHashes`,
which decodes every image through `sharp` to compute a perceptual hash on a cold
run. The client shows a static `LoadingOverlay` for the whole multi-minute wait.

We want a visible progress bar (and the groundwork for incremental indexing) on
**both** the dedupe page and the homepage's Dedupe entry, so the user can see how
much of the photo set has been traversed.

## Decisions (locked)

1. **Dedupe page outcome:** live **progress bar, clusters delivered at the end**.
   Clustering is global/greedy (each image attaches to the nearest existing anchor
   across the whole set), so streaming stable clusters mid-run is out of scope. We
   surface honest hashing progress and deliver the cluster batch when hashing +
   clustering finish.
2. **Homepage bar:** **persistent coverage % + auto-warm**. The homepage always
   shows "X / Y indexed" derived from the on-disk hash cache, and opening the
   homepage proactively warms the index so it's ready before the user opens dedupe.
3. **Transport:** **poll a shared progress endpoint**. Both pages poll
   `/api/dedupe/progress`; the dedupe page fires its normal cluster fetch in
   parallel. One mechanism, reused everywhere.
4. **Auto-warm scope:** warm the **selected** categories (consistent with the rest
   of the homepage, which is cat-scoped).
5. **Background concurrency:** make `ensureHashes` concurrency a **parameter**;
   background warm runs gentler (default 4) so it doesn't bog the server during
   Box/Label work. A foreground dedupe run may use the existing default (8).

## Core idea

Hashing is the cost, it's already cached to disk, and the cache is resumable.
So "progress" is defined as: **how many of a category's images currently have a
cached perceptual hash** (`countCached`) versus the total file count. This single
metric drives both bars. Both pages read it from one shared, server-side indexer;
neither page re-implements hashing.

`countCached` is cheap: after `loadDisk()`, count how many of the category's image
names are present in the in-memory `mem` map. No `stat`, no `sharp`. (Membership,
not mtime/size validity — slightly optimistic if a file changed in place, which is
acceptable for a progress bar; the job re-validates and recomputes regardless.)

## Architecture

Single Node process → module-level state is shared across requests.

### 1. New `annotation/lib/hash-indexer.ts` — singleton indexer

Owns a per-category **deduplicated job**:

```
jobs: Map<CatId, Promise<Map<name, HashInfo>>>
```

- `ensureCatHashes(cat): Promise<Map<name, HashInfo>>` — return the in-flight job
  for `cat` or start one wrapping `ensureHashes(cat, listImages(cat), concurrency)`.
  The job removes itself from `jobs` on completion (guarding against a newer job
  having replaced it).
- `warm(cats, concurrency = 4)` — fire-and-forget `ensureCatHashes` per cat.
  Idempotent: if a job is already running (whether started by auto-warm or by a
  real dedupe run), all callers share it. **No double-hashing.**
- `snapshot(cats): Promise<{ done, total, running }>` — computed cheaply with **no
  image decoding**: `done = Σ countCached(cat, names)`, `total = Σ names.length`,
  `running = cats.some(c => jobs.has(c))`. Awaits `loadDisk()` once.
- `invalidateCat(cat)` — drop a stale in-flight job (called after a removal).

### 2. `annotation/lib/hash-cache.ts`

- Add `countCached(cat, names): Promise<number>` — `await loadDisk()`, then count
  names present in `mem`. Cheap map lookups.
- Parameterize `ensureHashes(cat, names, onProgress?, concurrency = 8)` so the
  indexer can request gentler concurrency. (The existing `onProgress` hook stays;
  the counter does not depend on it, but it remains available.)

### 3. `annotation/lib/dedupe.ts`

`clusterCategory` awaits `ensureCatHashes(cat)` instead of calling `ensureHashes`
directly. An in-progress auto-warm and the dedupe run therefore reuse the exact
same work and feed the same counters. Clustering still runs to completion →
**clusters delivered at the end**, behavior otherwise unchanged.

### 4. New `GET /api/dedupe/progress?cats=…`

Returns `snapshot(cats)` **and** kicks `warm(cats)` as a side effect (idempotent).
This single endpoint is what both pages poll, and is what makes the homepage
auto-warm. `runtime = "nodejs"`, `dynamic = "force-dynamic"`.

### 5. `POST /api/dedupe` (existing removal)

Also call `invalidateCat(cat)` so a warm job in flight doesn't hand back a stale
hash map after files move to `dedup/`.

## Data flow

**Homepage** (`annotation/app/page.tsx`):
- On mount and on cat-change, poll `GET /api/dedupe/progress?cats=` (~2s). It warms
  the selected categories and returns coverage.
- Render a thin bar + "X / Y indexed (NN%)" inside the existing Dedupe `Link` card.
- **Stop** polling when `done === total && !running`; resume on cat change.

**Dedupe page** (`annotation/app/dedupe/page.tsx`):
- Keep the existing cluster `fetch` (clusters at end).
- *In parallel*, while `loading`, poll `GET /api/dedupe/progress?cats=` (~750ms)
  and show a real progress bar. Stop when the cluster fetch resolves.

## UI

Add an optional `progress?: { done: number; total: number }` prop to
`annotation/components/LoadingOverlay.tsx` that renders a thin bar + count under
the spinner label, reusing the bar styling already used by `GroupProgress` in the
dedupe page. The homepage gets an inline bar in the Dedupe card.

## Edge cases

- **New files added** after a completed run → `jobs` is empty, the next run
  re-hashes over the fresh `listImages(cat)`, and the count reflects it.
- **Removals mid-warm** → `invalidateCat(cat)` from the POST handler.
- **Fully indexed** → `done === total`, polling stops, zero background work.
- **Auto-warm CPU load** during Box/Label → gentle background concurrency (4).
- **`listImages` cost on each poll** (a `readdir` per cat) is tens of ms at 50k
  files; acceptable at the chosen poll intervals. Memoizing the total is a possible
  later optimization, not required now.

## Files touched

New:
- `annotation/lib/hash-indexer.ts`
- `annotation/app/api/dedupe/progress/route.ts`

Edited:
- `annotation/lib/hash-cache.ts` (add `countCached`, parameterize concurrency)
- `annotation/lib/dedupe.ts` (route hashing through the indexer)
- `annotation/app/api/dedupe/route.ts` (POST invalidates the cat's job)
- `annotation/components/LoadingOverlay.tsx` (optional progress bar)
- `annotation/app/dedupe/page.tsx` (poll progress + render bar)
- `annotation/app/page.tsx` (warm + poll + bar in the Dedupe card)

## Testing

`tests/` is Python (server-side). Confirm whether `annotation/` has a JS test
runner (e.g. vitest); if present, unit-test `countCached` and the indexer's job
deduplication (two concurrent `ensureCatHashes(cat)` calls share one underlying
`ensureHashes`). Regardless, verify via `tsc --noEmit` / `next build` and a manual
run against a large category: cold-load shows the bar advancing, warm-load is
near-instant, and the homepage coverage reaches 100%.

## Out of scope

- Streaming/partial cluster display before hashing completes.
- Reworking the greedy clustering algorithm.
- Cross-process coordination (single dev server assumed).
