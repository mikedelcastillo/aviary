# Dedupe Incremental Hashing Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show live perceptual-hash progress on the dedupe page and a persistent "X / Y indexed" coverage bar on the homepage Dedupe card, backed by a shared server-side hash indexer.

**Architecture:** A singleton module (`hash-indexer.ts`) owns a per-category deduplicated hashing job wrapping the existing `ensureHashes`. Both the homepage and the dedupe page poll a new `GET /api/dedupe/progress` endpoint that returns `{done, total, running}` (computed cheaply from cached-hash membership, no image decoding) and idempotently warms the index. The dedupe page keeps delivering clusters in one batch at the end; the bar just makes the wait visible.

**Tech Stack:** Next.js (App Router), React 19, TypeScript, Tailwind v4. No JS test runner is configured, so verification uses `npx tsc --noEmit`, `npm run lint`, and targeted manual checks against a running dev server.

---

## Notes for the implementer

- All commands run from the `annotation/` directory unless stated otherwise.
- The dev server is a single Node process, so module-level `Map`s in `lib/` are shared across requests. This is the whole mechanism — do not add cross-process coordination.
- "Cheap" progress means: count how many of a category's image names are present in the in-memory hash cache (`mem` map). No `fs.stat`, no `sharp`.
- Commit after every task. Branch is `feat/dedupe-incremental-progress` (already created).

---

## File Structure

New files:
- `annotation/lib/hash-indexer.ts` — singleton indexer: per-cat dedup'd job, `warm`, `snapshot`, `invalidateCat`.
- `annotation/app/api/dedupe/progress/route.ts` — `GET` returns snapshot + warms.

Modified files:
- `annotation/lib/hash-cache.ts` — add `countCached`; parameterize `ensureHashes` concurrency.
- `annotation/lib/dedupe.ts` — route hashing through the indexer.
- `annotation/app/api/dedupe/route.ts` — `POST` invalidates the affected cat's job.
- `annotation/components/LoadingOverlay.tsx` — optional progress-bar prop.
- `annotation/app/dedupe/page.tsx` — poll progress while loading, render the bar.
- `annotation/app/page.tsx` — warm + poll, render the coverage bar in the Dedupe card.

---

## Task 1: Add `countCached` and parameterize concurrency in the hash cache

**Files:**
- Modify: `annotation/lib/hash-cache.ts`

- [ ] **Step 1: Parameterize `ensureHashes` concurrency**

In `annotation/lib/hash-cache.ts`, change the `ensureHashes` signature to accept a `concurrency` argument and use it. Replace the signature block:

```ts
export async function ensureHashes(
  cat: CatId,
  names: string[],
  onProgress?: (done: number, total: number) => void,
): Promise<Map<string, HashInfo>> {
```

with:

```ts
export async function ensureHashes(
  cat: CatId,
  names: string[],
  onProgress?: (done: number, total: number) => void,
  concurrency = 8,
): Promise<Map<string, HashInfo>> {
```

Then, further down in the same function, remove the line:

```ts
  const CONCURRENCY = 8;
```

and change the worker-spawn line from:

```ts
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, names.length) }, worker));
```

to:

```ts
  await Promise.all(Array.from({ length: Math.min(concurrency, names.length) }, worker));
```

- [ ] **Step 2: Add `countCached`**

At the end of `annotation/lib/hash-cache.ts` (after `dropFromCache`), add:

```ts
/**
 * Count how many of the given names currently have a cached hash entry — the
 * cheap progress metric for dedupe (membership only; no stat, no decode). Loads
 * the on-disk snapshot first so a cold server reports real coverage.
 */
export async function countCached(cat: CatId, names: string[]): Promise<number> {
  await loadDisk();
  let n = 0;
  for (const name of names) if (mem.has(key(cat, name))) n++;
  return n;
}
```

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add annotation/lib/hash-cache.ts
git commit -m "feat(dedupe): add countCached and parameterize hash concurrency"
```

---

## Task 2: Create the singleton hash indexer

**Files:**
- Create: `annotation/lib/hash-indexer.ts`

- [ ] **Step 1: Write the indexer module**

Create `annotation/lib/hash-indexer.ts` with exactly:

```ts
// Singleton hashing coordinator for dedupe. One Node process → this module's
// `jobs` map is shared across requests, so the homepage auto-warm and an actual
// dedupe run reuse the SAME per-category hashing work (no double-hashing) and
// report against one source of truth.
import { listImages } from "./annotation-io";
import { ensureHashes, countCached, type HashInfo } from "./hash-cache";
import type { CatId } from "./types";

// Gentle background concurrency so warming doesn't starve Box/Label work.
const WARM_CONCURRENCY = 4;
// Foreground (dedupe page) may hash faster.
const DEDUPE_CONCURRENCY = 8;

// In-flight hashing job per category. Absent once the job settles.
const jobs = new Map<CatId, Promise<Map<string, HashInfo>>>();

/**
 * Start (or join) the hashing job for one category. Deduplicated: concurrent
 * callers share a single underlying `ensureHashes`. The job removes itself on
 * completion (guarding against a newer job having replaced it). `concurrency`
 * only takes effect when THIS call actually starts the job.
 */
export function ensureCatHashes(
  cat: CatId,
  concurrency = WARM_CONCURRENCY,
): Promise<Map<string, HashInfo>> {
  let job = jobs.get(cat);
  if (!job) {
    const names = listImages(cat);
    job = ensureHashes(cat, names, undefined, concurrency).finally(() => {
      if (jobs.get(cat) === job) jobs.delete(cat);
    });
    jobs.set(cat, job);
  }
  return job;
}

/** Foreground variant used by the dedupe clustering path (faster concurrency). */
export function ensureCatHashesForeground(cat: CatId): Promise<Map<string, HashInfo>> {
  return ensureCatHashes(cat, DEDUPE_CONCURRENCY);
}

/** Idempotently kick background hashing for the given categories. Fire-and-forget. */
export function warm(cats: CatId[]): void {
  for (const cat of cats) void ensureCatHashes(cat);
}

/**
 * Cheap progress snapshot across categories: `done` = cached-hash membership,
 * `total` = current file count, `running` = any job in flight. No image decoding.
 */
export async function snapshot(
  cats: CatId[],
): Promise<{ done: number; total: number; running: boolean }> {
  let done = 0;
  let total = 0;
  let running = false;
  for (const cat of cats) {
    const names = listImages(cat);
    total += names.length;
    done += await countCached(cat, names);
    if (jobs.has(cat)) running = true;
  }
  return { done, total, running };
}

/** Drop any in-flight job for a category (call after files move out of raw/). */
export function invalidateCat(cat: CatId): void {
  jobs.delete(cat);
}
```

- [ ] **Step 2: Type-check**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add annotation/lib/hash-indexer.ts
git commit -m "feat(dedupe): add singleton hash indexer"
```

---

## Task 3: Route clustering through the indexer

**Files:**
- Modify: `annotation/lib/dedupe.ts`

- [ ] **Step 1: Update imports**

In `annotation/lib/dedupe.ts`, change:

```ts
import { ensureHashes, type HashInfo } from "./hash-cache";
```

to:

```ts
import { type HashInfo } from "./hash-cache";
import { ensureCatHashesForeground } from "./hash-indexer";
```

- [ ] **Step 2: Use the shared job in `clusterCategory`**

In the same file, inside `clusterCategory`, change:

```ts
  const hashes = await ensureHashes(cat, names);
```

to:

```ts
  const hashes = await ensureCatHashesForeground(cat);
```

(The indexer reads `listImages(cat)` itself, so the local `names` const above it is still used for the clustering loop and stays as-is.)

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: no errors. (`HashInfo` is still referenced later in `clusterCategory`, so its import must remain.)

- [ ] **Step 4: Commit**

```bash
git add annotation/lib/dedupe.ts
git commit -m "feat(dedupe): cluster via shared hash indexer job"
```

---

## Task 4: Progress endpoint + invalidate on removal

**Files:**
- Create: `annotation/app/api/dedupe/progress/route.ts`
- Modify: `annotation/app/api/dedupe/route.ts`

- [ ] **Step 1: Create the progress route**

Create `annotation/app/api/dedupe/progress/route.ts` with exactly:

```ts
import { snapshot, warm } from "@/lib/hash-indexer";
import { parseCats } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * GET /api/dedupe/progress?cats=day,ir -> { done, total, running }.
 * Also idempotently warms the index for the requested categories (the side
 * effect that makes the homepage auto-warm). Returns coverage cheaply.
 */
export async function GET(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const cats = parseCats(url.searchParams.get("cats"));
  warm(cats);
  const snap = await snapshot(cats);
  return Response.json(snap);
}
```

- [ ] **Step 2: Invalidate the cat's job on removal**

In `annotation/app/api/dedupe/route.ts`, add to the imports at the top:

```ts
import { invalidateCat } from "@/lib/hash-indexer";
```

Then in the `POST` handler, inside the `try` block, immediately after the `for (const name of remove)` loop completes (before `} catch`), add:

```ts
    invalidateCat(cat as CatId);
```

So the block reads:

```ts
    // Sequential: each removeImage invalidates the manifest after itself.
    for (const name of remove) {
      const res = await removeImage(cat as CatId, name);
      moved.push({ name, files: res.moved });
    }
    invalidateCat(cat as CatId);
```

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Manual smoke test**

Run: `npm run dev`, then in another shell:
`curl "http://localhost:5000/api/dedupe/progress?cats=day"`
Expected: a JSON object like `{"done":N,"total":M,"running":true|false}` where `total` matches the number of images in the `day` category. A second immediate call should show `done` equal or higher (warming in progress). Stop the dev server when done.

- [ ] **Step 5: Commit**

```bash
git add annotation/app/api/dedupe/progress/route.ts annotation/app/api/dedupe/route.ts
git commit -m "feat(dedupe): add progress endpoint and invalidate job on removal"
```

---

## Task 5: Progress-bar prop on LoadingOverlay

**Files:**
- Modify: `annotation/components/LoadingOverlay.tsx`

- [ ] **Step 1: Replace the component**

Replace the entire contents of `annotation/components/LoadingOverlay.tsx` with:

```tsx
import { Spinner } from "./Spinner";

/** Centered spinner overlay shown while a screen's data/image is loading. When
 *  `progress` is supplied (and has a positive total), a thin determinate bar and
 *  count render under the label. */
export function LoadingOverlay({
  show,
  label,
  progress,
}: {
  show: boolean;
  label?: string;
  progress?: { done: number; total: number };
}) {
  const hasBar = !!progress && progress.total > 0;
  const pct = hasBar ? Math.round((progress!.done / progress!.total) * 100) : 0;
  return (
    <div
      className={`pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-bg/50 backdrop-blur-[1px] transition-opacity duration-200 ${
        show ? "opacity-100" : "opacity-0"
      }`}
    >
      {show && (
        <div
          className={
            "flex flex-col items-center gap-2 border border-border bg-surface/90 text-sm text-muted " +
            (hasBar ? "rounded-2xl px-5 py-3" : "rounded-pill px-4 py-2")
          }
        >
          <div className="flex items-center gap-3">
            <Spinner size={16} className="text-fg" />
            {label && <span>{label}</span>}
          </div>
          {hasBar && (
            <div className="w-56">
              <div className="h-2 overflow-hidden rounded-full bg-elevated">
                <div
                  className="h-full rounded-full bg-box transition-[width]"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="mt-1 text-center font-mono text-xs text-faint">
                {progress!.done.toLocaleString()} / {progress!.total.toLocaleString()} hashed ({pct}%)
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Type-check**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add annotation/components/LoadingOverlay.tsx
git commit -m "feat(dedupe): optional progress bar on LoadingOverlay"
```

---

## Task 6: Poll progress on the dedupe page

**Files:**
- Modify: `annotation/app/dedupe/page.tsx`

- [ ] **Step 1: Add progress state**

In `annotation/app/dedupe/page.tsx`, add a state hook next to the existing dedupe state (right after the `const [error, setError] = useState<string | null>(null);` line near the top of `DedupePage`):

```ts
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
```

- [ ] **Step 2: Add the polling effect**

In the same component, add this effect immediately after the existing cluster-fetch `useEffect` (the one with the dependency array `[cats, threshold]`):

```ts
  // While a load is in flight, poll the shared indexer for hashing progress so
  // the overlay shows a moving bar instead of a static spinner. Reuses the same
  // endpoint the homepage warms with.
  useEffect(() => {
    if (!loading) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const params = new URLSearchParams();
    const c = serializeCats(cats);
    if (c) params.set("cats", c);
    const poll = async () => {
      try {
        const res = await fetch(`/api/dedupe/progress?${params}`);
        if (res.ok) {
          const snap = (await res.json()) as { done: number; total: number };
          if (!cancelled) setProgress({ done: snap.done, total: snap.total });
        }
      } catch {
        /* transient — keep polling */
      }
      if (!cancelled) timer = setTimeout(poll, 750);
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [loading, cats]);
```

- [ ] **Step 3: Pass progress to the overlay**

In the same file, change the overlay render line from:

```tsx
      <LoadingOverlay show={loading} label={loadingLabel} />
```

to:

```tsx
      <LoadingOverlay show={loading} label={loadingLabel} progress={progress ?? undefined} />
```

- [ ] **Step 4: Type-check and lint**

Run: `npx tsc --noEmit`
Expected: no errors.
Run: `npm run lint`
Expected: no new errors for `app/dedupe/page.tsx`.

- [ ] **Step 5: Manual check**

Run `npm run dev`, open `http://localhost:5000/dedupe` on a category whose hashes are NOT yet cached (or delete the dedupe hash cache first to force a cold run — its path is `DEDUPE_CACHE_PATH` in `lib/paths.ts`). Expected: the overlay shows a progress bar with "N / M hashed (P%)" that advances, then clusters appear when it finishes. Stop the dev server when done.

- [ ] **Step 6: Commit**

```bash
git add annotation/app/dedupe/page.tsx
git commit -m "feat(dedupe): show live hashing progress bar on dedupe page"
```

---

## Task 7: Homepage coverage bar + auto-warm

**Files:**
- Modify: `annotation/app/page.tsx`

- [ ] **Step 1: Add dedupe-progress state**

In `annotation/app/page.tsx`, add next to the other `useState` hooks in `Home` (e.g. right after `const [random, setRandom] = useState(false);`):

```ts
  const [dedupeProgress, setDedupeProgress] = useState<{
    done: number;
    total: number;
    running: boolean;
  } | null>(null);
```

- [ ] **Step 2: Add the warm + poll effect**

In the same component, add this effect after the existing selection-scoped `useEffect` (the one that loads queue/box-queue/entry/label-stats with dependency `[cats]`):

```ts
  // Warm the dedupe hash index for the selected categories and poll coverage so
  // the Dedupe card shows how much has been traversed. The GET both warms (side
  // effect) and reports {done,total,running}. Stop once fully indexed and idle.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const param = serializeCats(cats);
    const qs = param ? `?cats=${param}` : "";
    const poll = async () => {
      try {
        const res = await fetch(`/api/dedupe/progress${qs}`);
        if (res.ok) {
          const snap = (await res.json()) as {
            done: number;
            total: number;
            running: boolean;
          };
          if (!cancelled) {
            setDedupeProgress(snap);
            if (snap.total > 0 && snap.done >= snap.total && !snap.running) return;
          }
        }
      } catch {
        /* transient — keep polling */
      }
      if (!cancelled) timer = setTimeout(poll, 2000);
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [cats]);
```

- [ ] **Step 3: Render the bar in the Dedupe card**

In the same file, replace the existing Dedupe `Link` block:

```tsx
      <Link
        href={withCats("/dedupe", cats)}
        className="mt-3 flex items-center justify-between rounded-xl border border-border bg-surface px-5 py-3 text-sm transition-colors hover:border-border-strong hover:bg-surface-2"
      >
        <span className="font-medium text-fg">Dedupe</span>
        <span className="text-muted">Find &amp; remove near-duplicate frames →</span>
      </Link>
```

with:

```tsx
      <Link
        href={withCats("/dedupe", cats)}
        className="mt-3 flex flex-col gap-2 rounded-xl border border-border bg-surface px-5 py-3 text-sm transition-colors hover:border-border-strong hover:bg-surface-2"
      >
        <div className="flex items-center justify-between">
          <span className="font-medium text-fg">Dedupe</span>
          <span className="text-muted">Find &amp; remove near-duplicate frames →</span>
        </div>
        {dedupeProgress && dedupeProgress.total > 0 && (
          <div>
            <div className="h-1.5 overflow-hidden rounded-full bg-elevated">
              <div
                className="h-full rounded-full bg-box transition-[width]"
                style={{
                  width: `${Math.round((dedupeProgress.done / dedupeProgress.total) * 100)}%`,
                }}
              />
            </div>
            <div className="mt-1 font-mono text-[11px] text-faint">
              {dedupeProgress.done.toLocaleString()} / {dedupeProgress.total.toLocaleString()} indexed
              ({Math.round((dedupeProgress.done / dedupeProgress.total) * 100)}%)
              {dedupeProgress.running && " · indexing…"}
            </div>
          </div>
        )}
      </Link>
```

- [ ] **Step 4: Type-check and lint**

Run: `npx tsc --noEmit`
Expected: no errors.
Run: `npm run lint`
Expected: no new errors for `app/page.tsx`.

- [ ] **Step 5: Manual check**

Run `npm run dev`, open `http://localhost:5000/`. Expected: the Dedupe card shows an "N / M indexed (P%)" bar; on a cold cache it advances over time and reads "· indexing…" while running, reaching 100% when done. Toggling categories rescopes the count. Stop the dev server when done.

- [ ] **Step 6: Commit**

```bash
git add annotation/app/page.tsx
git commit -m "feat(dedupe): homepage coverage bar with auto-warm"
```

---

## Task 8: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full type-check**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 2: Lint**

Run: `npm run lint`
Expected: no new errors introduced by this branch.

- [ ] **Step 3: Production build**

Run: `npm run build`
Expected: build succeeds; the new `/api/dedupe/progress` route appears in the route list.

- [ ] **Step 4: End-to-end manual check**

With `npm run dev`:
1. Delete the dedupe hash cache (`DEDUPE_CACHE_PATH` from `lib/paths.ts`) to force a cold run.
2. Load the homepage — the Dedupe card bar starts climbing from a low number.
3. Open `/dedupe` — the overlay shows the hashing bar advancing, then clusters render.
4. Return home — the card now reads ~100% indexed and polling stops (no "· indexing…").
5. Remove a group in dedupe, reload — counts remain consistent (the affected cat re-hashes only the changed set).

---

## Self-Review

- **Spec coverage:** progress metric (Task 1 `countCached`); singleton indexer + warm/snapshot/invalidate (Task 2); clusters-at-end via shared job (Task 3); progress endpoint + POST invalidation (Task 4); shared bar UI (Task 5); dedupe-page poll/bar (Task 6); homepage coverage + auto-warm, selected-cat scope (Task 7); type-check/build/manual verification (Task 8). Gentle background concurrency (4) and foreground (8) handled in Task 2 via `WARM_CONCURRENCY`/`DEDUPE_CONCURRENCY`. All spec sections map to a task.
- **Placeholders:** none — every code step shows full code; verification steps give exact commands and expected output.
- **Type consistency:** `ensureCatHashes`/`ensureCatHashesForeground`/`warm`/`snapshot`/`invalidateCat` (Task 2) are the exact names imported in Tasks 3, 4. `countCached(cat, names): Promise<number>` (Task 1) matches its use in Task 2's `snapshot`. The progress shape `{done, total, running}` is consistent across the endpoint (Task 4), dedupe page (Task 6, reads `done`/`total`), and homepage (Task 7, reads all three). `LoadingOverlay`'s `progress?: { done; total }` prop (Task 5) matches the value passed in Task 6.
