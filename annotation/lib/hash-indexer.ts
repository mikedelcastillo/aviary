// Singleton hashing coordinator for dedupe. One Node process → this module's
// `jobs` map is shared across requests, so the homepage auto-warm and an actual
// dedupe run reuse the SAME per-category hashing work (no double-hashing) and
// report against one source of truth.
import { listImages } from "./annotation-io";
import { ensureHashes, countCached, type HashInfo } from "./hash-cache";
import type { CatId } from "./types";

// Gentle background concurrency so warming doesn't starve other requests (the
// homepage's label-stats/progress scans share CPU and the libuv thread pool).
const WARM_CONCURRENCY = 2;
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
