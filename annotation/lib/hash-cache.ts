// Server-only perceptual-hash cache — the cost lever for dedupe mode.
//
// Hashing ~1.5k images decodes every file through sharp (seconds, cold). We cache
// each hash keyed by "cat/name" and validate with the file's mtime + size, so:
//   - re-running dedupe is near-instant (warm cache),
//   - changing the threshold re-clusters WITHOUT re-hashing (clustering only reads
//     the cached hashes).
// Two layers: a module-level in-memory Map (per server run) backed by an on-disk
// JSON snapshot (survives dev-server restarts).
import { promises as fs } from "node:fs";
import { DEDUPE_CACHE_PATH, imageFsPath } from "./paths";
import type { CatId } from "./types";
import { phashFromFile } from "./phash";

interface MemRec {
  mtimeMs: number;
  size: number;
  hash: bigint;
}

const mem = new Map<string, MemRec>(); // key: `${cat}/${name}`
let diskLoaded = false;

function key(cat: CatId, name: string): string {
  return `${cat}/${name}`;
}

// --- on-disk snapshot -------------------------------------------------------

interface DiskShape {
  version: number;
  entries: Record<string, { mtimeMs: number; size: number; hash: string }>;
}

async function loadDisk(): Promise<void> {
  if (diskLoaded) return;
  diskLoaded = true; // mark first so a parse failure doesn't retry every call
  try {
    const raw = await fs.readFile(DEDUPE_CACHE_PATH, "utf8");
    const data = JSON.parse(raw) as DiskShape;
    if (data.version !== 1 || !data.entries) return;
    for (const [k, v] of Object.entries(data.entries)) {
      mem.set(k, { mtimeMs: v.mtimeMs, size: v.size, hash: BigInt(`0x${v.hash}`) });
    }
  } catch {
    // No cache yet (or unreadable) — start empty.
  }
}

async function flushDisk(): Promise<void> {
  const entries: DiskShape["entries"] = {};
  for (const [k, v] of mem) {
    entries[k] = { mtimeMs: v.mtimeMs, size: v.size, hash: v.hash.toString(16).padStart(16, "0") };
  }
  const payload: DiskShape = { version: 1, entries };
  const tmp = `${DEDUPE_CACHE_PATH}.tmp-${process.pid}`;
  await fs.writeFile(tmp, JSON.stringify(payload), "utf8");
  await fs.rename(tmp, DEDUPE_CACHE_PATH);
}

// --- public API -------------------------------------------------------------

export interface HashInfo {
  hash: bigint;
  size: number;
}

/**
 * Hash one image, using mem -> disk -> compute. Stats the file to validate the
 * cached entry (mtime + size). Returns null if the file is missing/unreadable.
 */
export async function getHash(cat: CatId, name: string): Promise<HashInfo | null> {
  await loadDisk();
  const k = key(cat, name);
  let stat;
  try {
    stat = await fs.stat(imageFsPath(cat, name));
  } catch {
    mem.delete(k);
    return null;
  }
  const cached = mem.get(k);
  if (cached && cached.mtimeMs === stat.mtimeMs && cached.size === stat.size) {
    return { hash: cached.hash, size: cached.size };
  }
  let hash: bigint;
  try {
    hash = await phashFromFile(imageFsPath(cat, name));
  } catch {
    return null; // undecodable image — skip, don't poison the cache
  }
  mem.set(k, { mtimeMs: stat.mtimeMs, size: stat.size, hash });
  return { hash, size: stat.size };
}

/**
 * Ensure hashes for a list of images in a category. Hashes only stale/missing
 * entries (bounded concurrency) and persists the disk cache once if anything
 * changed. Missing/undecodable files are skipped (absent from the result map).
 */
export async function ensureHashes(
  cat: CatId,
  names: string[],
  onProgress?: (done: number, total: number) => void,
): Promise<Map<string, HashInfo>> {
  await loadDisk();
  const out = new Map<string, HashInfo>();
  const before = mem.size;
  let dirty = false;
  let done = 0;

  const CONCURRENCY = 8;
  let cursor = 0;
  async function worker(): Promise<void> {
    while (cursor < names.length) {
      const name = names[cursor++];
      const k = key(cat, name);
      const had = mem.get(k);
      const info = await getHash(cat, name);
      if (info) {
        out.set(name, info);
        // getHash mutates mem; detect a recompute (new key or changed record).
        const now = mem.get(k);
        if (!had || had.hash !== now!.hash || had.mtimeMs !== now!.mtimeMs) dirty = true;
      } else if (had) {
        dirty = true; // a previously-cached file vanished -> getHash evicted it
      }
      done++;
      onProgress?.(done, names.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, names.length) }, worker));

  if (dirty || mem.size !== before) {
    try {
      await flushDisk();
    } catch {
      // Best-effort cache; a flush failure must not fail the request.
    }
  }
  return out;
}

/** Evict images from the in-memory cache (call after moving files out of raw/). */
export function dropFromCache(cat: CatId, names: string[]): void {
  for (const name of names) mem.delete(key(cat, name));
}
